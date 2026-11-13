import json
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import polars as pl
from loguru import logger as log
from SimpleITK import ImageSeriesReader, WriteImage

from sarcosegv2._types import StrPath
from sarcosegv2.data_models import SeriesMetadata


def find_study_input(input_dir: StrPath, study_uid: str) -> Path | None:
    """
    Try to find input study data, expecting either a single `<study_uid>.nii.gz`, `<study_uid>.nii` or `<study_uid>/` with multiple `*.dcm` files.
    Uses the whole `StudyInstanceUID` DICOM tag value as pattern.

    Parameters
    ----------
    input_dir : Path
        Directory path containing input study files.
        Expects NifTI file .../<uid>.nii.gz, .../<uid>.nii or DICOM directory .../<uid>/.
    study_uid : str
        Identifying StudyInstanceUID pattern to find input study.

    Returns
    -------
    Path or None
        Path pointing to the NifTI file or directory. `None` if not found.
    """

    found = Path(input_dir).glob(study_uid + "*")

    for p in found:
        if p.is_file() and ".nii" in p.name:
            log.debug(f"Found study as NifTI file: `{p}`")
            return p
        if p.is_dir() and len(list(p.rglob("*"))) > 0:
            log.debug(f"Found study as DICOM directory: `{p}`")
            return p
        else:
            log.warning(f"No study file/directory found for {study_uid}")


def read_study_cases_list(path: StrPath) -> pl.DataFrame:
    """
    Read study cases file, filter rows for non-empty StudyInstanceUID column values.
    Expects at least one column (CSV) or key (JSON) named `study_uid` with non-empty value.

    Returns
    -------
    polars.DataFrame
        DataFrame with empty or `null` values filtered out.

    Raises
    ------
    NoDataError
        Input CSV file has no data.
    ComputeError
        Input JSON file has no data.
    PolarsError
        Filtered dataframe has no remaining valid study_uid rows,
        caused by only empty strings in CSV rows or JSON key-value pairs resulting in dataframe length equaling 0.
    """
    path = Path(path)
    if path.suffix == ".csv":
        df = pl.read_csv(path, separator=";", has_header=True, raise_if_empty=True)
    elif path.suffix == ".json":
        df = pl.read_json(path)
    else:
        raise OSError(f"Expected `.json` or `.csv` file, got {path.name}")

    df = df.rename(str.lower)
    condition = (pl.col("study_uid").is_not_null()) & (
        pl.col("study_uid").str.len_bytes() > 0
    )
    removed_rows = df.remove(condition)
    df = df.filter(condition)

    if not removed_rows.is_empty():
        log.warning(
            f"Filtered out {len(removed_rows)} row(s) with empty/null study_uid field"
        )
        log.debug(f"Filtered out rows:\n{removed_rows}")

    if df.is_empty():
        log.error(
            "Study cases file has no remaining rows after filtering for non-empty StudyInstanceUID values"
        )
        raise pl.exceptions.PolarsError(
            f"DataFrame has no StudyInstanceUID column, file: `{path}`"
        )
    return df


def read_toml(path: Path | str) -> dict[str, Any]:
    with open(path, mode="rb") as file:
        data = tomllib.load(file)
    return data


def write_to_nifti(
    study_dir: StrPath,
    output_dir: StrPath,
    series_metadatas: dict[str, SeriesMetadata],
):
    """
    Writes DICOM series to NIfTI format.
    """

    study_dir = Path(study_dir)
    output_dir = Path(output_dir)

    reader = ImageSeriesReader()

    for series in series_metadatas.values():
        log.debug(f"Writing {series.series_name} volume as NIfTI")
        output_filepath = output_dir.joinpath(
            series.series_name, "input_ct_volume.nii.gz"
        )
        output_filepath.parent.mkdir(exist_ok=True, parents=True)

        if output_filepath.exists():
            log.debug(f"Overwritting `{output_filepath}`")

        filenames = reader.GetGDCMSeriesFileNames(study_dir, series.series_uid)
        reader.SetFileNames(filenames)
        image = reader.Execute()
        try:
            WriteImage(image, output_filepath)
        except Exception as ex:
            log.error(f"{ex}")
            continue

        log.debug(f"Written {series.series_name} into `{output_filepath}`")


def write_to_json(mapping: Mapping[str, Any], filepath: StrPath):
    """
    Write dictionary-like mapping into `.json`.
    """

    with open(filepath, mode="w") as file:
        json.dump(mapping, file, indent=4)
