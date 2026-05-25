import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

import pydicom
from pydicom.multival import MultiValue
from SimpleITK import ImageSeriesReader, WriteImage

from src.classes import SeriesData, StudyData
from src.utils import SERIES_DESC_PATTERN

log = logging.getLogger("preprocess")

CONTRAST_PHASES_PATTERN = re.compile(
    r"|".join(
        (
            "abdomen",
            "arterial",
            "nephro",
            "venous",
            "thorax",
            "angio",
            "aorta",
            "aortic",
        )
    ),
    re.IGNORECASE,
)


def preprocess_dicom_study(
    input_dir: str | Path, output_dir: str | Path, study_case: StudyData
) -> None:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    dicom_file_dir = find_dicoms(input_dir)

    if not dicom_file_dir:
        log.warning(f"no DICOM files found in `{input_dir}`")
        return None

    dicom_files, dicom_dir = dicom_file_dir
    log.debug(
        f"preprocessing DICOM files for case {study_case.participant}, study {study_case.study_inst_uid}"
    )

    series_files_map, dose_report_paths = filter_dicom_files(dicom_files)

    event_dose_map = None
    if dose_report_paths and len(dose_report_paths) > 0:
        event_dose_map = extract_dose_values(dose_report_paths)

    study_case.series = select_series_to_segment(
        series_files_map, event_dose_map=event_dose_map
    )

    # a single dicom file is enough for finding out the age
    study_case.calculate_ct_relative_patient_age(list(series_files_map.values())[0][0])

    log.debug(f"found {len(study_case.series)} valid series for segmentation")

    output_dir.mkdir(exist_ok=True, parents=True)

    process_tube_currents(
        study_case.series,
        output_dir=output_dir,
    )

    study_case._write_to_json(output_dir)

    write_series_as_nifti(
        dicom_dir,
        output_dir,
        study_case.series,
    )


def write_series_as_nifti(
    dicom_directory, output_study_dir: Path, series_uids: dict[str, SeriesData]
):
    reader = ImageSeriesReader()

    for uid in series_uids:
        log.debug(f"converting {uid} DICOM volume into NifTI")

        output_series_dir = output_study_dir.joinpath(uid)
        output_series_dir.mkdir(exist_ok=True, parents=True)
        output_filepath = output_series_dir.joinpath("input_ct_volume.nii.gz")

        filenames = reader.GetGDCMSeriesFileNames(dicom_directory, uid)
        reader.SetFileNames(filenames)
        image = reader.Execute()
        WriteImage(image, output_filepath)

        log.info(f"written {uid} DICOM as NifTI")

        if output_filepath.exists():
            log.debug(
                f"overwriting existing input_ct_volume.nii.gz at `{str(output_filepath.parent)}`"
            )


def filter_dicom_files(
    dicom_files: list[Path],
) -> tuple[dict[str, list[Path]], list[str | Path] | None]:
    """
    Sorts filepaths by DICOM tag SeriesInstanceUID and removes (filters) out files matching these rules:
    * SeriesDescription contains `protocol`, `topogram`, `scout`, `dose`, `report`, `patient`, `monitor`
        * excluding dose report
    * SliceThickness is None
    * ImageType contains `SECONDARY`

    Also tries to find and return fullpath Dose report series.

    Args:
        dicom_files (list[Path]): List of dicom files, excluding DICOMDIR file.

    Returns:
        tuple: A tuple of mapped series and path to dose report DICOM file.
        - **series_files_map** (dict[str, list[Path]]): Dictionary of filepaths mapped by SeriesInstanceUID keys.
        - **dose_report_file** (Path | None): path to Dose Report file if found, otherwise `None`.
    """

    series_files_map: defaultdict[str, list[Path]] = defaultdict(list[Path])

    dose_report_files = []
    for file in dicom_files:
        ds = pydicom.dcmread(
            file,
            stop_before_pixels=True,
            specific_tags=[
                "ImageType",
                "SeriesInstanceUID",
                "SeriesDescription",
                "SliceThickness",
                "Modality",
                "ConvolutionKernel",
            ],
        )

        # FILTER OUT FILES WITH FOLLOWING RULES:
        # 1. "dose report" in SeriesDescription -> keep path to "dose report" file
        # 2. "SECONDARY" in SeriesDescription -> does not affect files without that string, eg. ["DERIVED", "PRIMARY", "ORIGINAL", "AXIAL", ...]
        #   - also removes plane reconstructed images, 3D volume renderings
        # 3. file has SliceThickness -> removes non image type files - reports, protocols, etc.
        # 4. filter out based on regex pattern match on SeriesDescription -> final clean up for any remaining non-image files

        series_desc = ds.get("SeriesDescription", "").lower()
        if "dose report" in series_desc:
            dose_report_files.append(file)
            continue

        image_type = ds.get("ImageType", [])
        if "SECONDARY" in image_type:
            continue

        if not hasattr(ds, "SliceThickness"):
            continue

        if convolution_kernel := ds.get("ConvolutionKernel", ""):
            convolution_kernel = (
                convolution_kernel[0]
                if isinstance(convolution_kernel, MultiValue)
                else convolution_kernel
            )
            if "bl57" in convolution_kernel.lower():
                continue

        # filter out remaining files with series matching pattern:
        # ("protocol", "topogram", "scout", "patient", "dose", "report"), case insensitive
        if SERIES_DESC_PATTERN.search(series_desc):
            continue

        series_uid = ds.SeriesInstanceUID

        series_files_map[series_uid].append(file)

    log.debug(f"found {len(series_files_map.keys())} image series")

    if len(dose_report_files) == 0:
        log.warning("no dose report DICOM files found")

    return series_files_map, dose_report_files


def select_series_to_segment(
    series_files_map: dict[str, list[Path]],
    event_dose_map: dict[str, dict[str, float | int]] | None,
) -> dict[str, SeriesData]:
    """
    Return one or more CT series with lowest slice thickness and highest file count based on contrast phase type:
        - abdomen (ie. native, no constrast phase), arterial, venous, nephrous.

    Args:
        all_series (dict[str, list[Path]]): Mapping of `SeriesInstanceUID` to filepaths.
        event_dose_map (dict[str, dict[str, float]]): Mapping of `IrradiationEventUID` to dose values.

    Returns:
        series_list (dict[str, SeriesData]): List of series selected for segmentation.
    """

    series_by_contrast: defaultdict[str, list[SeriesData]] = defaultdict(
        list[SeriesData]
    )

    for series_uid, filepaths in series_files_map.items():
        # read only the first file to filter series
        dataset = pydicom.dcmread(filepaths[0], stop_before_pixels=True)

        # filter by words in SeriesDescription
        series_desc: str = dataset.SeriesDescription

        convolution_kernel = dataset.get("ConvolutionKernel", None)

        series_data = SeriesData(
            series_inst_uid=series_uid,
            series_description=series_desc,
            slice_thickness=float(dataset.get("SliceThickness", -1.0)),
            filepaths=filepaths,
            filepaths_num=len(filepaths),
            # has_contrast="yes" if contrast_applied else "no",
            irradiation_event_uid=dataset.get("IrradiationEventUID", "n/a"),
            convolution_kernel=convolution_kernel[0] if convolution_kernel else "n/a",
        )

        contrast_applied = dataset.get("ContrastBolusAgent", None)

        if contrast_match := CONTRAST_PHASES_PATTERN.search(series_desc):
            phase = contrast_match.group().lower()
            series_data.contrast_phase = phase
            series_data.has_contrast = (
                "yes" if contrast_applied and phase != "abdomen" else "no"
            )
        else:
            series_data.contrast_phase = "other"
            series_data.has_contrast = "n/a"

        series_by_contrast[series_data.contrast_phase].append(series_data)

    selected_series = {
        phase: min(
            data_list,
            key=lambda data: (data.slice_thickness, -data.filepaths_num),
        )
        for phase, data_list in series_by_contrast.items()
    }

    for series_data in selected_series.values():
        series_data.kilo_voltage_peak = float(
            pydicom.dcmread(
                series_data.filepaths[0], stop_before_pixels=True, specific_tags=["KVP"]
            ).get("KVP", 0.0)
        )

        if event_dose_map and series_data.irradiation_event_uid != "n/a":
            series_data.mean_ctdi_vol = event_dose_map.get(
                series_data.irradiation_event_uid, {}
            ).get("mean_ctdi_vol", -1.0)
            series_data.dose_length_product = event_dose_map.get(
                series_data.irradiation_event_uid, {}
            ).get("dlp", -1.0)

    return {series.series_inst_uid: series for series in selected_series.values()}


def extract_dose_values(
    dose_filepaths: list[str | Path],
) -> dict[str, dict[str, float]]:
    """
    Maps IrradiationEventUID to a map of dose values:
        - EventUID1
            - DLP = val
            - Mean CTDIvol = val
        - EventUID2
            - DLP = val
            - Mean CTDIvol = val
        - ...

    Args:
        dose_filepath (str | Path): Path to dose report DICOM file.

    Returns:
        event_to_dose (dict[str, dict[str, float]]): map of IrradiationEventUID to map of dose values
    """

    """
    event_to_dose = {
        irradiation_uid1: {"dlp": ..., "mean_ctdi_vol": ...},
        irradiation_uid2: {...},
        ...
    }
    """
    event_to_dose: dict[str, dict[str, float | int]] = {}

    def walk_sequence(sequence, current_event=None):
        for item in sequence:
            vr = item.ValueType

            # get irradiation event uid
            if hasattr(item, "ConceptNameCodeSequence"):
                code_meaning = item.ConceptNameCodeSequence[0].CodeMeaning
                if code_meaning == "Irradiation Event UID" and hasattr(item, "UID"):
                    current_event = item.UID
                    if current_event not in event_to_dose:
                        event_to_dose[current_event] = {
                            "dlp": -1,
                            "mean_ctdi_vol": -1,
                        }

            # find dose values
            if vr == "NUM" and hasattr(item, "MeasuredValueSequence"):
                code_meaning = item.ConceptNameCodeSequence[0].CodeMeaning
                val = float(item.MeasuredValueSequence[0].NumericValue)

                if current_event:
                    if code_meaning in ["DLP"]:
                        event_to_dose[current_event]["dlp"] = val
                    elif code_meaning in ["Mean CTDIvol"]:
                        event_to_dose[current_event]["mean_ctdi_vol"] = val

            # recursively iterate content sequences
            if hasattr(item, "ContentSequence"):
                walk_sequence(item.ContentSequence, current_event)

    for dose_file in dose_filepaths:
        ds = pydicom.dcmread(dose_file)

        if ds.Modality != "SR":
            log.warning(f"file {dose_file} is not dose report")
            continue

        walk_sequence(ds.ContentSequence)
    return event_to_dose


def find_dicoms(dicom_dir: Path) -> tuple[list[Path], Path] | None:
    """
    Returns DICOM files excluding DICOMDIR file.

    Args:
        dicom_dir (Path): Path to DICOM directory.

    Returns:
        paths (list[Path] | None): List of DICOM filepaths, otherwise `None`.
    """

    for root, dirs, files in dicom_dir.walk():
        if len(files) == 0 or "DICOMDIR" in files:
            continue
        paths = [f for f in root.iterdir() if f.is_file()]

        if not paths:
            return None
        return paths, root


def process_tube_currents(series: dict[str, SeriesData], output_dir: Path | str):

    series_currents: defaultdict[str, dict[str, int]] = defaultdict(dict[str, int])

    for uid, series_data in series.items():
        log.debug(f"processing tube currents for {uid}")

        datasets = [
            pydicom.dcmread(
                p,
                stop_before_pixels=True,
                specific_tags=["XRayTubeCurrent", "InstanceNumber"],
            )
            for p in series_data.filepaths
        ]

        instnum_currents = {
            ds.get("InstanceNumber"): int(
                ds.get(
                    "XRayTubeCurrent",
                )
            )
            for ds in datasets
        }

        series_currents[uid] = instnum_currents

        series_data.mean_tube_current = round(
            mean([current for current in instnum_currents.values() if current]),
            ndigits=4,
        )

    with open(Path(output_dir, "inst_num_currents.json"), "w") as file:
        json.dump(series_currents, file, indent=2)
    log.debug(f"saved instance number, tube current for {len(series_currents)} series")
