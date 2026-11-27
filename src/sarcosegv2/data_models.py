from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from enum import StrEnum
from pathlib import Path
from typing import Any

import SimpleITK as sitk

from sarcosegv2._types import StrPath
from sarcosegv2.labels import TISSUE_LABEL_BY_NAME
from sarcosegv2.utils import tube_current_by_instance_number


class Result(StrEnum):
    Ok = "OK"
    ImageWrite = "IMAGE_WRITE"
    SeriesMetadataNotFound = "SERIES_METADATA_NOT_FOUND"
    VertebraeLabelNotFound = "VERTEBRAE_LABEL_NOT_FOUND"
    SpineSegmentationFail = "SPINE_SEGMENTATION_FAIL"
    TissueSegmentationFail = "TISSUE_SEGMENTATION_FAIL"


@dataclass
class SegmentationFailureState:
    paritcipant: str
    study_uid: str | None
    series_name: str
    state: Result | None


@dataclass(repr=False)
class StudyMetadata:
    participant: str
    study_uid: str | None = field(default=None)
    patient_height: float | int | None = field(default=None, compare=False)
    series: list[str] = field(default_factory=list, compare=False)

    def __repr__(self):
        return f"""
StudyMetadata(
    participant={self.participant!r},
    study_uid={self.study_uid!r},
    patient_height={self.patient_height!r},
    series={self.series}
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
            series=map.get("series", []),
        )

    def to_mapping(self) -> Mapping[str, Any]:
        return asdict(self)


@dataclass(repr=False)
class SeriesMetadata:
    series_uid: str
    series_name: str
    series_number: str
    description: str | None = None
    contrast_phase: str | None = None
    contrast_type: str | None = None
    convolution_kernel: str | None = None
    slice_thickness: float | None = None
    slice_count: int | None = None
    irradiation_event_uid: str | None = None
    kilo_voltage_peak: float | None = None
    mean_tube_current: float | None = None
    mean_ctdi_vol: float | None = None
    dose_length_product: float | None = None
    currents_by_instance: dict[str, float] | None = None

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
            parts = "\n    ".join(
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
            series_uid=map["series_uid"],
            series_name=map["series_name"],
            series_number=map["series_number"],
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


@dataclass
class Centroid:
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

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]):
        return cls(
            mapping.get("spine_s"),
            mapping.get("tissue_s"),
            mapping.get("postprocess_s"),
        )


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
    area_cm2: dict[str, float] = field(default_factory=dict)
    mean_hu: dict[str, float] = field(default_factory=dict)
    skeletal_muscle_index: float | None = field(default=None)

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]):
        """
        Construct `TissueMetrics` from map of tissue metrics.
        Expects structure:
        {
            "area_cm2": {"sat": <float>, ...},
            "mean_hu": {"sat": <float>, ...},
            "skeletal_muscle_index": <float>
        }
        """
        return cls(
            area_cm2=mapping["area_cm2"],
            mean_hu=mapping["mean_hu"],
            skeletal_muscle_index=mapping.get("skeletal_muscle_index"),
        )

    @classmethod
    def compute_tissue_metrics(
        cls,
        tissue_vol: sitk.Image,
        tissue_mask: sitk.Image,
        patient_height_cm: float | None = None,
    ) -> TissueMetrics:
        """
        Compute tissue metrics per tissue type - area (cm^2), mean HU - and skeletal muscle index (cm^2 / m^2).
        Expects patient height in centimeters.
        """

        if tissue_vol.GetSize()[-1] == 1 and tissue_mask.GetSize()[-1] == 1:
            tissue_vol = tissue_vol[..., 0]
            tissue_mask == tissue_mask[..., 0]

        stats_filt = sitk.LabelIntensityStatisticsImageFilter()
        stats_filt.Execute(tissue_vol, tissue_mask)

        area: dict[str, float] = {
            tissue_type: round(stats_filt.GetPhysicalSize(label) / 100.0, ndigits=4)
            for tissue_type, label in TISSUE_LABEL_BY_NAME.items()
        }

        mean_hu: dict[str, float] = {
            tissue_type: round(stats_filt.GetMean(label), ndigits=4)
            for tissue_type, label in TISSUE_LABEL_BY_NAME.items()
        }

        skeletal_muscle_index = None
        if patient_height_cm and patient_height_cm > 0:
            skeletal_muscle_index = compute_smi(
                area["muscle"],
                patient_height_cm,
            )

        return TissueMetrics(
            area_cm2=area, mean_hu=mean_hu, skeletal_muscle_index=skeletal_muscle_index
        )


def compute_smi(muscle_area_cm2: float, patient_height_cm: float) -> float:
    """
    Return skeletal muscle index (SMI), defined as `SMI = muscle_area / (patient_height ^ 2)`, rounded to 4 digits.
    Expects muscle area in centimeters^2, patient height in centimeters.

    Patient height is implicitly converted to meters.
    """
    return round(muscle_area_cm2 / ((patient_height_cm / 100.0) ** 2), ndigits=4)


@dataclass(repr=False)
class SeriesResult:
    """
    Segmentation result for each series within a study case.
    Identified by `series_name` and `series_uid`.

    Stores vertebrae centroid, tissue metrics (area cm^2, mean HU)
    and duration (seconds) of each processing step - spine segmentation, tissue segmentation, postprocessing.
    """

    series_name: str
    series_uid: str | None = None
    status: Result | None = None
    vertebrae_centroid: Centroid | None = None
    tissue_metrics: TissueMetrics | None = None
    durations: SegmentationTime | None = None
    volume_z_size: int | None = None
    ct_pixel_spacing: tuple[float, float, float] | None = None
    l3_tube_current: float | None = None

    def __repr__(self):
        return f"""
SeriesResult(
    series_name={self.series_name!r},
    study_uid={self.series_uid!r},
    status={self.status!r},
)
        """

    def __format__(self, spec: str) -> str:
        if spec == "v":
            parts = "\n    ".join(
                f"{f.name}={getattr(self, f.name)!r}" for f in fields(self)
            )
            return f"{type(self).__name__}({parts})"
        return repr(self)

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]):
        if (series_name := mapping.get("series_name")) is None:
            raise KeyError("Key `series_name` not found")

        tiss = mapping.get("tissue_metrics")
        dur = mapping.get("durations")

        return cls(
            series_name=mapping["series_name"],
            series_uid=mapping.get("series_uid"),
            status=mapping.get("status"),
            vertebrae_centroid=mapping.get("vertebrae_centroid"),
            tissue_metrics=TissueMetrics.from_mapping(tiss) if tiss else None,
            durations=SegmentationTime.from_mapping(dur) if dur else None,
            volume_z_size=mapping.get("volume_z_size"),
            ct_pixel_spacing=mapping.get("ct_pixel_spacing", ()),
            l3_tube_current=mapping.get("l3_tube_current"),
        )

    def compute_tissue_metrics(
        self, tissue_vol: sitk.Image, mask: sitk.Image, patient_height
    ) -> None:
        """
        Convenience method for computing tissue metrics. See `TissueMetrics.compute_tissue_metrics()` for more details.
        """
        self.tissue_metrics = TissueMetrics.compute_tissue_metrics(
            tissue_vol, mask, patient_height
        )

    def set_l3_tube_current(self, series_dir: StrPath):
        filepath = Path(series_dir).joinpath("tube_currents_by_instance.json")
        self.l3_tube_current = tube_current_by_instance_number(
            filepath, self.series_name, self.vertebrae_centroid.z
        )


@dataclass(repr=False)
class StudySegmentationResult:
    participant: str
    study_uid: str | None = field(default=None)
    patient_height: float | None = field(default=None, compare=False)
    series: dict[str, SeriesResult] = field(default_factory=dict, compare=False)

    def __repr__(self):
        return f"""
StudySegmentationResult(
    participant={self.participant!r},
    study_uid={self.study_uid!r},
    patient_height={self.patient_height!r},
    series={tuple(self.series.keys())}
)
        """

    @classmethod
    def from_study_metadata(cls, study_case: StudyMetadata) -> StudySegmentationResult:
        return cls(
            study_case.participant,
            study_case.study_uid,
            patient_height=study_case.patient_height,
        )

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]):
        participant = mapping.get("participant")

        if participant is None:
            raise KeyError("Key `participant` not found in `{filepath}`")
        elif participant.lower() in ("null", "nan", "none") or len(participant) == 0:
            raise ValueError(
                f"Key `participant` requires non-null or non-empty value, got {participant}"
            )

        result = StudySegmentationResult(
            participant=participant,
            study_uid=mapping.get("study_uid"),
            patient_height=mapping.get("patient_height"),
            series={
                series_name: SeriesResult.from_mapping(data)
                for series_name, data in mapping.get("series", {}).items()
            },
        )
        return result

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self, dict_factory=lambda d: {key: val for key, val in d})
