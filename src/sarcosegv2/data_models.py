from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from enum import StrEnum
from typing import Any, NamedTuple

from loguru import logger as log


class Result(StrEnum):
    Ok = "OK"
    ImageWrite = "IMAGE_WRITE"
    SeriesMetadataNotFound = "SERIES_METADATA_NOT_FOUND"
    VertebraeLabelNotFound = "VERTEBRAE_LABEL_NOT_FOUND"
    SpineSegmentationFail = "SPINE_SEGMENTATION_FAIL"
    TissueSegmentationFail = "TISSUE_SEGMENTATION_FAIL"


@dataclass(repr=False)
class StudyMetadata:
    participant: str
    study_uid: str | None = field(default=None)
    patient_height: float | int | None = field(default=None, compare=False)
    series: dict[str, SeriesMetadata] = field(default_factory=dict, compare=False)

    def __repr__(self):
        return f"""
StudyMetadata(
    participant={self.participant!r},
    study_uid={self.study_uid!r},
    patient_height={self.patient_height!r},
    series={tuple(self.series.keys())}
)
        """

    @classmethod
    def from_mapping(cls, map: Mapping[str, Any]) -> StudyMetadata:
        """
        Construct an instance of `StudyMetadata` from a map, expecting non-empty key-value pairs `study_uid` and `participant`.

        Returns
        -------
        StudyMetadata
        """

        missing = [k for k in ("participant", "study_uid") if not map.get(k)]
        if missing:
            raise ValueError(
                f"Missing or empty required keys {missing} in row {dict(map)}"
            )

        return cls(
            participant=map["participant"],
            study_uid=map.get("study_uid"),
            patient_height=map.get("patient_height"),
            series=map.get("series"),
        )

    def series_by_name(self, series_name: str) -> SeriesMetadata | None:
        """
        Convenience method to access series by `series_name` key.

        Returns
        -------
        SeriesMetadata or None
        """
        series = self.series.get(series_name)
        if not series:
            log.warning(f"StudyMetadata has no series by {series_name}")
        return series

    def to_mapping(self) -> Mapping[str, Any]:
        return asdict(self)


@dataclass(repr=False)
class SeriesMetadata:
    series_uid: str
    series_name: str
    description: str | None = None
    contrast_phase: str | None = None
    contrast_type: str | None = None
    convolution_kernel: str | None = None
    slice_thickness: float | None = None
    slice_count: int | None = None
    irradiation_event_uid: str | None = None
    kilo_voltage_peak: float | None = None
    currents_by_instance: dict[str, float] | None = None
    mean_tube_current: float | None = None
    mean_ctdi_vol: float | None = None
    dose_length_product: float | None = None

    def __repr__(self):
        return f"""
SeriesMetadata(
    name={self.series_name!r},
    description={self.description!r},
    contrast_applied={self.contrast_type!r},
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

    @classmethod
    def from_mapping(cls, map: Mapping[str, Any]):
        """
        Instantiate `StudyMetadata` from dictionary-like map.
        Requires non-empty `series_uid`, `series_name` key-value pairs.
        """
        missing = [k for k in ("series_uid", "series_name") if not map.get(k)]
        if missing:
            raise ValueError(
                f"Missing or empty required keys {missing} in row {dict(map)}"
            )

        return cls(
            series_uid=map["uid"],
            series_name=map["name"],
            description=map.get("description"),
            contrast_phase=map.get("contrast_phase"),
            contrast_type=map.get("contrast_type"),
            convolution_kernel=map.get("convolution_kernel"),
            slice_thickness=map.get("slice_thickness"),
            slice_count=map.get("slice_count"),
            irradiation_event_uid=map.get("irradiation_event_uid"),
            kilo_voltage_peak=map.get("kilo_voltage_peak"),
            currents_by_instance=map.get("currents_by_instance"),
            mean_tube_current=map.get("mean_tube_current"),
            mean_ctdi_vol=map.get("mean_ctdi_vol"),
            dose_length_product=map.get("dose_length_product"),
        )

    def to_mapping(self) -> Mapping[str, Any]:
        """
        Convert `SeriesMetadata` into dictionary map.
        """
        return asdict(self)

    def add_dose(self, dose: Mapping[str, float]):
        """
        Adds dose values meanCTDIvol and Dose Length Product (DLP) to SeriesMetadata.
        Value -1.0 indicates a missing value in Dose Report Structure Report DICOM file.
        """
        self.mean_ctdi_vol = dose.get("mean_ctdi_vol", -1.0)
        self.dose_length_product = dose.get("dose_length_product", -1.0)


class Centroid(NamedTuple):
    """
    Centroid coordinates (voxel space) of vertebrae body.
    Assumes LPS orientation of `sitk.Image`.
    """

    x: int  # Right towards Left
    y: int  # Anterior towards Posterior
    z: int  # Inferior towards Superior


@dataclass
class SegmentationTime:
    """
    Elapsed time (seconds) of each step.
    """

    spine_s: float | None = None
    tissue_s: float | None = None
    postprocess_s: float | None = None


class TissueType(StrEnum):
    # FIX: Improve the docstring.
    """
    For serialization.
    """

    SAT = "sat"
    VAT = "vat"
    IMAT = "imat"
    MUSCLE = "muscle"


@dataclass(frozen=True)
class TissueArea:
    """
    Cross-section area (mm^2) per tissue type:

    sat - subcutaneous adipose tissue
    vat - visceral adipose tissue
    imat - intermuscular adipose tissue
    muscle - muscle tissue
    """

    sat: float
    vat: float
    imat: float
    muscle: float


@dataclass(frozen=True)
class TissueMeanHU:
    """
    Mean Hounsfield Unit per tissue type:

    sat - subcutaneous adipose tissue
    vat - visceral adipose tissue
    imat - intermuscular adipose tissue
    muscle - muscle tissue
    """

    sat: float
    vat: float
    imat: float
    muscle: float


@dataclass
class TissueMetrics:
    area_cm2: TissueArea
    mean_hu: TissueMeanHU


@dataclass(repr=False)
class SeriesResult:
    """
    Segmentation result for each series within a study case.
    Identified by `series_name` and `series_uid`.

    Stores vertebrae centroid, tissue metrics (area cm^2, mean HU)
    and duration (seconds) of each processing step - spine segmentation, tissue segmentation, postprocessing.
    """

    series_name: str
    series_uid: str | None = field(default=None)
    status: Result | None = field(default=None)
    vertebrae_centroid: Centroid | None = field(default=None)
    metrics: TissueMetrics | None = field(default=None)
    durations: SegmentationTime = field(default_factory=SegmentationTime)
    volume_z_size: int | None = field(default=None)
    l3_slice_index: int | None = field(default=None)
    l3_tube_current: float | None = field(default=None)


@dataclass(repr=False)
class StudySegmentationResult:
    participant: str
    study_uid: str | None = field(default=None)
    patient_height: float | None = field(default=None, compare=False)
    series: dict[str, SeriesResult] = field(default_factory=dict, compare=False)

    @classmethod
    def from_study_metadata(cls, study_case: StudyMetadata) -> StudySegmentationResult:
        return cls(
            study_case.participant,
            study_case.study_uid,
            patient_height=study_case.patient_height,
        )
