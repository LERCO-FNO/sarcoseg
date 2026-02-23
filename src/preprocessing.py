import re
import shutil
from pathlib import Path
from statistics import mean

import dcm2niix
import pydicom

from src import slogger
from src.classes import SeriesData, StudyData

logger = slogger.get_logger(__name__)


class DicomStudyPreprocessor:
    def __init__(self):
        pass

    def _preprocess_study(self):
        pass


SERIES_DESC_PATTERN = re.compile(
    r"|".join(
        ("protocol", "topogram", "scout", "patient", "dose", "report", "monitor")
    ),
    re.IGNORECASE,
)
CONTRAST_PHASES_PATTERN = re.compile(
    r"|".join(("abdomen", "arterial", "nephro", "venous")), re.IGNORECASE
)


def preprocess_dicom_study(
    input_dir: str | Path, output_dir: str | Path, study_case: StudyData
) -> None:
    if isinstance(input_dir, str):
        input_dir = Path(input_dir)

    if isinstance(output_dir, str):
        output_dir = Path(output_dir)

    dicom_files = find_dicoms(input_dir)

    if not dicom_files:
        logger.error(f"no DICOM files found in `{input_dir}`")
        return None

    # TODO maybe add verify_dicom_study(), remove if no
    # if not verify_dicom_study(study_case.study_inst_uid, dicom_files[0]):
    #     logger.error("requested study ")
    #     return

    logger.info(
        f"preprocessing DICOM files for participant {study_case.participant}, study {study_case.study_inst_uid}"
    )

    series_files_map, dose_report_path = filter_dicom_files(dicom_files)

    event_dose_map = None
    if dose_report_path:
        event_dose_map = extract_dose_values(dose_report_path)

    study_case.series = select_series_to_segment(
        series_files_map, event_dose_map=event_dose_map
    )

    logger.info(f"found {len(study_case.series)} valid series for segmentation")

    output_dir.mkdir(exist_ok=True, parents=True)

    study_case._write_to_json(output_dir)

    for series_data in study_case.series.values():
        write_series_as_nifti(output_dir, series_data)

    logger.info("-" * 25)


def write_series_as_nifti(output_study_dir: Path, series_data: SeriesData):
    logger.info(f"writing series instance uid {series_data.series_inst_uid}")

    output_series_dir = output_study_dir.joinpath(series_data.series_inst_uid)
    output_series_dir.mkdir(exist_ok=True, parents=True)
    output_filepath = output_series_dir.joinpath("input_ct_volume.nii.gz")

    tmp_dir = Path(output_study_dir, f"tmp_{series_data.description}")
    tmp_dir.mkdir(exist_ok=True, parents=True)
    [shutil.copy2(file, tmp_dir.joinpath(file.name)) for file in series_data.filepaths]

    if output_filepath.exists():
        logger.info(
            f"overwriting existing input_ct_volume.nii.gz at `{str(output_filepath.parent)}`"
        )

    try:
        args = [
            "-o",
            str(output_series_dir),
            "-f",
            "input_ct_volume",
            "-z",
            "y",
            "-b",
            "n",
            "-w",
            "1",
            str(tmp_dir),
        ]
        returncode = dcm2niix.main(args, capture_output=True, text=True)
        shutil.rmtree(tmp_dir)
    except RuntimeError as err:
        logger.error(err)
    logger.info(f"finished NifTI conversion with {returncode=}\n")


def filter_dicom_files(
    dicom_files: list[Path],
) -> tuple[dict[str, list[Path]], Path | None]:
    """
    Sorts filepaths by DICOM tag SeriesInstanceUID and removes (filters) out files matching these rules:
    * SeriesDescription contains `protocol`, `topogram`, `scout`, `dose`, `report`, `patient`, `monitor`
        * excluding dose report
    * SliceThickness is None
    * ImageType contains `DERIVED`

    Also tries to find and return fullpath Dose report series.

    Args:
        dicom_files (list[Path]): List of dicom files, excluding DICOMDIR file.

    Returns:
        tuple: A tuple of mapped series and path to dose report DICOM file.
        - **series_files_map** (dict[str, list[Path]]): Dictionary of filepaths mapped by SeriesInstanceUID keys.
        - **dose_report_file** (Path | None): path to Dose Report file if found, otherwise `None`.
    """

    series_files_map: dict[str, list[Path]] = {}

    dose_report_file = None
    for file in dicom_files:
        ds = pydicom.dcmread(
            file,
            stop_before_pixels=True,
            specific_tags=[
                "ImageType",
                "SeriesInstanceUID",
                "SeriesDescription",
                "SliceThickness",
            ],
        )

        if "dose report" in ds.SeriesDescription.lower():
            dose_report_file = file
            continue

        # filter out files in series matching pattern:
        # ("protocol", "topogram", "scout", "patient", "dose", "report"), case insensitive
        if SERIES_DESC_PATTERN.search(ds.SeriesDescription):
            continue

        if not hasattr(ds, "SliceThickness"):
            continue

        if "DERIVED" in ds.ImageType:
            continue

        series_uid = ds.SeriesInstanceUID
        if series_uid in series_files_map:
            series_files_map[series_uid].append(file)
        else:
            series_files_map[series_uid] = [file]

    logger.info(f"found {len(series_files_map.keys())} image series")

    if not dose_report_file:
        logger.warning("dose report DICOM file not found")

    return series_files_map, dose_report_file


def select_series_to_segment(
    series_files_map: dict[str, list[Path]],
    event_dose_map: dict[str, dict[str, float]] | None,
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

    series_by_contrast: dict[str, list[SeriesData]] = {}

    for series_uid, filepaths in series_files_map.items():
        # read only the first file to filter series
        dataset = pydicom.dcmread(filepaths[0], stop_before_pixels=True)

        # filter by words in SeriesDescription
        series_desc: str = dataset.SeriesDescription

        contrast_applied = dataset.get("ContrastBolusAgent", None)
        convolution_kernel = dataset.get("ConvolutionKernel", None)

        series_data = SeriesData(
            series_inst_uid=series_uid,
            description=series_desc,
            slice_thickness=float(dataset.get("SliceThickness", -1.0)),
            filepaths=filepaths,
            filepaths_num=len(filepaths),
            has_contrast="yes" if contrast_applied else "no",
            irradiation_event_uid=dataset.get("IrradiationEventUID", "n/a"),
            convolution_kernel=convolution_kernel[0] if convolution_kernel else "n/a",
        )

        if contrast_match := CONTRAST_PHASES_PATTERN.search(series_desc):
            series_data.contrast_phase = contrast_match.group().lower()
        else:
            series_data.contrast_phase = "other"

        if series_data.contrast_phase in series_by_contrast:
            series_by_contrast[series_data.contrast_phase].append(series_data)
        else:
            series_by_contrast[series_data.contrast_phase] = [series_data]

    selected_series = {
        phase: min(
            data_list,
            key=lambda data: (data.slice_thickness, -data.filepaths_num),
        )
        for phase, data_list in series_by_contrast.items()
    }

    for series_data in selected_series.values():
        tube_currents = [
            pydicom.dcmread(
                p, stop_before_pixels=True, specific_tags=["XRayTubeCurrent"]
            ).get("XRayTubeCurrent", None)
            for p in series_data.filepaths
        ]
        series_data.mean_tube_current = mean(
            [float(current) for current in tube_currents if current]
        )

        series_data.kilo_voltage_peak = float(
            pydicom.dcmread(
                series_data.filepaths[0], stop_before_pixels=True, specific_tags=["KVP"]
            ).get("KVP", 0.0)
        )

        if event_dose_map and series_data.irradiation_event_uid != "n/a":
            series_data.mean_ctdi_vol = event_dose_map.get(
                series_data.irradiation_event_uid
            ).get("mean_ctdi_vol", -1.0)
            series_data.dose_length_product = event_dose_map.get(
                series_data.irradiation_event_uid
            ).get("dlp", -1.0)

    return {series.series_inst_uid: series for series in selected_series.values()}


def extract_dose_values(dose_filepath: str | Path) -> dict[str, dict[str, float]]:
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
    ds = pydicom.dcmread(dose_filepath)

    if ds.Modality != "SR":
        logger.warning(f"file {dose_filepath} is not dose report")
        return {}

    event_to_dose = {}

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
                            "dlp": None,
                            "mean_ctdi_vol": None,
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

    walk_sequence(ds.ContentSequence)
    return event_to_dose


def find_dicoms(dicom_dir: Path) -> list[Path] | None:
    """
    Returns DICOM files excluding DICOMDIR file.

    Args:
        dicom_dir (Path): Path to DICOM directory.

    Returns:
        paths (list[Path] | None): List of DICOM filepaths, otherwise `None`.
    """

    for root, _, files in dicom_dir.walk():
        if len(files) == 0 or "DICOMDIR" in files:
            continue
        paths = [f for f in root.iterdir() if f.is_file()]

        if not paths:
            return None
        return paths
