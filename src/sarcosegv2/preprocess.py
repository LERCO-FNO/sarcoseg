import re
from collections import defaultdict
from collections.abc import Mapping

# from dataclasses import dataclass, field, fields
from pathlib import Path
from statistics import mean

from loguru import logger as log
from pydicom import Dataset, dcmread
from pydicom.multival import MultiValue

from sarcosegv2.classes import SeriesMetadata, StudyMetadata
from sarcosegv2.series_filter import (
    CONTRAST_PHASE_DESC_PATTERNS,
    filter_cfind_series_responses,
)
from sarcosegv2.utils import find_dicoms


class DicomPreprocessor:
    def __init__(self, preprocess_dose_report: bool = False):
        self.preprocess_dose_report = preprocess_dose_report

    def preprocess_study(self, study_dir: str | Path, output_dir: str | Path):
        """
        Preprocesses DICOM study and writes each selected series as NIfTI.
        Output study directory structure:
            - .../study_uid/series_description1/...
            - .../study_uid/series_description2/...
            - ...

        Parameters
        ----------
        study_dir : str | Path
            Directory path to search for study DICOM files.
        output_dir : str | Path
            Directory path to write selected series.
        """
        study_dir = Path(study_dir)
        output_dir = Path(output_dir)

        dicom_files = find_dicoms(study_dir)
        if not dicom_files:
            log.warning(f"No DICOM files in `{study_dir}`")
            return

        dicom_files, dicom_dir = dicom_files
        log.debug(f"Preprocessing DICOM study, files at `{dicom_dir}`")

        datasets = [
            dcmread(
                fp,
                # specific_tags=[
                #     "PatientID",
                #     "StudyInstanceUID",
                #     "SeriesInstanceUID",
                #     "SeriesDescription",
                #     "ImageType",
                #     "SliceThickness",
                #     "ConvolutionKernel",
                #     "BodyPartExamined",
                # ],
                stop_before_pixels=True,
            )
            for fp in dicom_files
        ]

        datasets_by_uid, dose_datasets = filter_cfind_series_responses(datasets)
        dose_by_irrad_uid = extract_ct_dose(dose_datasets) if dose_datasets else None

        if not dose_by_irrad_uid:
            log.warning(f"No dose data found for study at `{study_dir}`")

        series_metadatas = extract_series_metadata(datasets_by_uid, dose_by_irrad_uid)
        selected_series = select_series_to_segment(series_metadatas)

        return selected_series


@dataclass(repr=False)
class SeriesMetadata:
    uid: str
    name: str
    description: str | None = None
    contrast_phase: str | None = None
    series_contrast_type: str | None = None
    convolution_kernel: str | None = None
    slice_thickness: float | None = None
    slice_count: int | None = None
    irradiation_event_uid: str | None = None
    kilo_voltage_peak: float | None = None
    mean_tube_current: float | None = None
    mean_ctdi_vol: float | None = None
    dose_length_product: float | None = None

    def __repr__(self):
        return f"""
SeriesMetadata(
    name={self.name!r},
    description={self.description!r},
    contrast_applied={self.series_contrast_type!r},
    slice_thickness={self.slice_thickness!r}
)
        """

    def __format__(self, spec: str) -> str:
        if spec == "v":
            parts = "\n\t".join(
                f"{f.name}={getattr(self, f.name)!r}" for f in fields(self)
            )
            return f"{type(self).__name__}({parts})"
        return repr(self)

    def add_dose(self, dose: dict[str, float]):
        """
        Adds dose values meanCTDIvol and Dose Length Product (DLP) to SeriesMetadata.
        Value -1.0 indicates a missing value in Dose Report Structure Report DICOM file.
        """
        self.mean_ctdi_vol = dose.get("mean_ctdi_vol", -1.0)
        self.dose_length_product = dose.get("dose_length_product", -1.0)


# Restructure map for the purpose of guessing contrast phase by pattern matching.
# From contrast-phase: series-desc-pattern:
# {
# "phase": ["patt1", "patt2", ...],
# ...
# }
# into series-descritpion-pattern: contrast-phase:
# {
# "patt1": "phase",
# "patt2": "phase",
# ...
# }
PHASE_BY_CONTRAST_PATTERN = {
    pattrn: phase
    for phase, pattrns in CONTRAST_PHASE_DESC_PATTERNS.items()
    for pattrn in pattrns
}


CONTRAST_PHASES_PATTERN = re.compile(
    r"|".join(re.escape(p) for p in PHASE_BY_CONTRAST_PATTERN),
    re.IGNORECASE,
)


def extract_series_metadata(
    datasets_by_series_uid: Mapping[str, list[Dataset]],
    dose_by_irrad_uid: dict[str, dict[str, float]] | None,
) -> list[SeriesMetadata]:
    """
    Select the highest-resolution series (thinnest slices and highest slice count) per contrast phase.

    Parameters
    ----------
    datasets_by_series_uid : Mapping(str, list(Dataset))
        Datasets of every image series keyed by SeriesInstanceUID.

    Returns
    -------
    list(SeriesMetadata)
        List of candidate `SeriesMetadata` instances.
    """
    series_metadatas = []
    for series_uid, datasets in datasets_by_series_uid.items():
        ds = datasets[0]

        series_desc = ds.get("SeriesDescription")
        convolution_kernel = ds.get("ConvolutionKernel", None)
        if isinstance(convolution_kernel, MultiValue):
            convolution_kernel = convolution_kernel[0]
        if not convolution_kernel:
            convolution_kernel = "nan"

        # TODO: check correct contrast phase resolving
        # - abdomen (native), arterial, nephrous or venous
        # more here: https://github.com/LERCO-FNO/sarcoseg/blob/0561819c0d98d3ac9009b7ae35154b1f8803be12/src/preprocessing.py#L237
        # and here: https://github.com/LERCO-FNO/sarcoseg/blob/0561819c0d98d3ac9009b7ae35154b1f8803be12/src/utils.py#L17
        series_contrast_type, contrast_phase = resolve_contrast_phase(ds)

        slice_thickness = float(ds.get("SliceThickness", -1.0))
        series_name = f"{contrast_phase}_{slice_thickness}_{convolution_kernel}"

        series_meta = SeriesMetadata(
            uid=series_uid,
            name=series_name,
            description=series_desc,
            contrast_phase=contrast_phase,
            series_contrast_type=series_contrast_type,
            convolution_kernel=convolution_kernel,
            slice_thickness=float(ds.get("SliceThickness", -1.0)),
            slice_count=len(datasets),
            irradiation_event_uid=ds.get("IrradiationEventUID"),
            kilo_voltage_peak=float(ds.get("KVP", -1.0)),
            mean_tube_current=extract_mean_tube_current(datasets),
        )

        if dose_by_irrad_uid and series_meta.irradiation_event_uid:
            if dose := dose_by_irrad_uid.get(series_meta.irradiation_event_uid, None):
                series_meta.add_dose(dose)
            else:
                log.warning(
                    f"Series {series_meta.name} (uid: {series_meta.uid}) has no dose data in Dose Report"
                )

        log.debug(f"Constructed {series_meta:v}")
        series_metadatas.append(series_meta)

    return series_metadatas


def select_series_to_segment(
    candidate_series: list[SeriesMetadata],
) -> dict[str, SeriesMetadata]:
    """
    Select series with lowest slice thickness per contrast phase.
    Uses highest slice count to resolve equal slice thickness, otherwise returns result of `min()`.
    Each contrast phase can have only one series for segmentation.

    Parameters
    ----------
    candidate_series : list(SeriesMetadata)
        Candidate series to select for segmentation.

    Returns
    -------
    dict(str, SeriesMetadata)
        Maps selected series by series name for convenience. See `extract_series_metadata()` for definition of series name.
    """

    series_by_contrast: defaultdict[str, list[SeriesMetadata]] = defaultdict(
        list[SeriesMetadata]
    )
    for series in candidate_series:
        series_by_contrast[series.contrast_phase].append(series)
    selected_series = {
        phase: min(
            series_list,
            key=lambda series: (series.slice_thickness, -series.slice_count),
        )
        for phase, series_list in series_by_contrast.items()
    }
    return {series.name: series for series in selected_series.values()}


def resolve_contrast_phase(ds: Dataset) -> tuple[str, str]:
    """
    Pattern match on SeriesDescription to guess contrast phase.
    See CONTRAST_PHASE_DESC_PATTERNS for a map of possible matching patterns.

    Patterns in each list have been identified manually by identifying most
    frequent words in SeriesDescription attribute across thousands of series'.

    Parameters
    ----------
    ds : pydicom.Dataset
        Input dataset.

    Returns
    -------
    contrast_applied : str
        Indicates series has contrast applied, one of ["native", "contrast", "nan"].
        "nan" indicates failure in resolving contrast_phase.
    contrast_phase : str
        Matched contrast phase type, one of ["abdomen", "arterial", "venous", "nephrous"].
    """
    contrast_applied = "contrast" if ds.get("ContrastBolusAgent", None) else "native"
    contrast_phase = "unknown"

    if contrast_match := CONTRAST_PHASES_PATTERN.search(ds.get("SeriesDescription")):
        contrast_phase = PHASE_BY_CONTRAST_PATTERN.get(
            contrast_match.group().lower(), "unknown"
        )

    if contrast_phase == "unknown":
        contrast_applied = "nan"
    return (contrast_applied, contrast_phase)


def extract_ct_dose(
    dose_datasets: list[Dataset] | list[Path],
) -> dict[str, dict[str, float]]:
    """
    Extracts specific CT dose values from dose report datasets.
    Requires Dataset.Modality == "SR" and Dataset.ContentSequence to have dose values for DLP (Dose Length Product)
    and meanCTDIvol (volume computed tomography dose index).

    Parameters
    ----------
    dose_datasets : list(pydicom.Dataset | Path)
        List of pydicom.Dataset or DICOM file paths.


    Returns:
    dose_by_irrad_uid : dict(str, dict(str, float))
        Mapping of dose values by IrradiationEventUID:
        {
            "irradiation_uid1: {"dlp": ..., "mean_ctdi_vol": ...},
            "irradiation_uid2: {"dlp": ..., "mean_ctdi_vol": ...},
            ...
        }
    """

    dose_by_irrad_uid: dict[str, dict[str, float]] = {}

    def walk_sequence(sequence, current_event=None):
        for item in sequence:
            vr = item.ValueType

            # get irradiation event uid
            if hasattr(item, "ConceptNameCodeSequence"):
                code_meaning = item.ConceptNameCodeSequence[0].CodeMeaning
                if code_meaning == "Irradiation Event UID" and hasattr(item, "UID"):
                    current_event = item.UID
                    if current_event not in dose_by_irrad_uid:
                        dose_by_irrad_uid[current_event] = {
                            "dose_length_product": -1.0,
                            "mean_ctdi_vol": -1.0,
                        }

            # find dose values
            if vr == "NUM" and hasattr(item, "MeasuredValueSequence"):
                code_meaning = item.ConceptNameCodeSequence[0].CodeMeaning
                val = float(item.MeasuredValueSequence[0].NumericValue)

                if current_event:
                    if code_meaning in ["DLP"]:
                        dose_by_irrad_uid[current_event]["dose_length_product"] = val
                    elif code_meaning in ["Mean CTDIvol"]:
                        dose_by_irrad_uid[current_event]["mean_ctdi_vol"] = val

            # recursively iterate content sequences
            if hasattr(item, "ContentSequence"):
                walk_sequence(item.ContentSequence, current_event)

    for ds in dose_datasets:
        if isinstance(ds, Path):
            ds = dcmread(ds)

        if ds.Modality != "SR":
            log.warning(f"Dataset {ds.get('SeriesDescription')} is not dose report")
            continue

        walk_sequence(ds.ContentSequence)
    return dose_by_irrad_uid


def extract_mean_tube_current(datasets: list[Dataset]) -> float:
    """
    Collect XRayTubeCurrent values and return their mean.
    """
    raw_tube_currents = [ds.get("XRayTubeCurrent") for ds in datasets]
    tube_currents = [float(c) for c in raw_tube_currents if c]
    if len(tube_currents) < len(raw_tube_currents):
        log.warning(
            f"Some DICOM datasets don't have XRayTubeCurrent: expected {len(raw_tube_currents)} values, got {len(tube_currents)}"
        )

    return mean(tube_currents)
