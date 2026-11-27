from pathlib import Path

from loguru import logger as log

from sarcosegv2._types import StrPath
from sarcosegv2.data_io import read_from_json


def find_dicoms(dicom_dir: Path) -> tuple[list[Path], Path] | None:
    """
    Discover DICOM files excluding DICOMDIR file.

    Parameters
    ----------
    dicom_dir : Path
        Path to DICOM study directory.

    Returns
    -------
    paths : list(Path)
        List of discovered DICOM file paths, otherwise `None`.
    root : Path
        Root path of study directory.
    """

    for root, dirs, files in dicom_dir.walk():
        if len(files) == 0 or "DICOMDIR" in files:
            continue
        paths = [f for f in root.iterdir() if f.is_file()]

        if not paths:
            return None
        return paths, root


def tube_current_by_instance_number(
    filepath: StrPath, series_name: str, instance_num: int
) -> float | None:
    series_meta = read_from_json(filepath)["series"].get(series_name)

    # depending on preprocessing outcome,
    # key `series_name` may not be present
    if series_meta is None:
        log.error(f"Tube currents for series {series_name} not found")
        return None

    # key `currents_by_instance` always present
    return series_meta["currents_by_instance"].get(str(instance_num))
