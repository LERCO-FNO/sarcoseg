import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Self

from nibabel.nifti1 import Nifti1Image
from numpy.typing import NDArray

from src import slogger

from src.io import read_json

logger = slogger.get_logger(__name__)


@dataclass
class SeriesData:
    series_inst_uid: str
    description: str | None = None
    filepaths: list[Path] = field(default_factory=list, repr=False)
    filepaths_num: int | None = field(default=None, repr=False)
    slice_thickness: float | None = field(default=None, repr=False)
    convolution_kernel: str | None = field(default=None, repr=False)
    has_contrast: str | None = field(default=None, repr=False)
    contrast_phase: str | None = field(default=None, repr=False)
    kilo_voltage_peak: float | None = field(default=None, repr=False)
    mean_tube_current: float | None = field(default=None, repr=False)
    irradiation_event_uid: str | None = field(default=None, repr=False)
    mean_ctdi_vol: float | None = field(default=None, repr=False)
    dose_length_product: float | None = field(default=None, repr=False)

    @classmethod
    def _from_dict(cls, series_data: dict[str, Any]) -> Self:
        return cls(**series_data)


@dataclass
class StudyData:
    participant: str
    study_inst_uid: str
    labkey_row_id: str | None = field(default=None, compare=False, repr=False)
    patient_id: str | None = field(default=None, repr=False, compare=False)
    study_date: str | None = field(default=None, repr=False, compare=False)
    patient_height: float | int | None = field(default=None, repr=False, compare=False)
    series: dict[str, SeriesData] = field(default_factory=dict)

    @classmethod
    def _from_labkey_row(cls, row: dict[str, Any]) -> Self:
        return cls(
            participant=row.get("PARTICIPANT"),
            study_inst_uid=row.get("STUDY_INSTANCE_UID"),
            labkey_row_id=row.get("ID"),
            patient_id=row.get("RODNE_CISLO"),
            patient_height=row.get("VYSKA_PAC."),
        )

    @classmethod
    def _from_json(cls, path: str | Path) -> Self:
        """Deserialize study and series data from JSON to `StudyData`.

        Args:
            filepath (str | Path): Path to .json file.

        Returns:
            study_data (StudyData): Deserialized `StudyData` dataclass object.
        """
        data = read_json(path)
        series_dict: dict[str, Any] = data.pop("series")
        return cls(
            **data,
            series={
                key: SeriesData._from_dict(val) for key, val in series_dict.items()
            },
        )

    def get_series(self, series_uid: str) -> SeriesData | None:
        return self.series.get(series_uid)

    def _write_to_json(
        self,
        output_dir: str | Path,
        exclude_fields: list[str] | None = None,
    ):
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)

        exclude = {"filepaths", "filepaths_num"}
        if exclude_fields:
            exclude.update(exclude_fields)

        serialized = asdict(
            self,
            dict_factory=lambda dic: {
                key: val for key, val in dic if key not in exclude
            },
        )

        filepath = output_dir.joinpath(f"dicom_tags_{self.study_inst_uid}.json")
        if filepath.exists():
            logger.info(f"overwriting existing file at `{str(filepath)}`")

        with open(filepath, mode="w", encoding="utf-8") as file:
            json.dump(serialized, file, indent=2)

        logger.info(
            f"written DICOM tags for participant {self.participant}, study instance uid {self.study_inst_uid}\nfields excluded: {exclude}"
        )

    def _to_list_of_dicts(self):
        _study = {
            "participant": self.participant,
            "study_inst_uid": self.study_inst_uid,
            "study_date": self.study_date,
            "labkey_row_id": self.labkey_row_id,
        }
        return [_study | series.__dict__ for series in self.series.values()]


@dataclass
class ImageData:
    image: Nifti1Image | NDArray
    path: Path
    spacing: NDArray | None = None


@dataclass
class Centroids:
    vertebre_centroid: list = field(default_factory=list)
    body_centroid: list = field(default_factory=list)


@dataclass
class MetricsData:
    area: dict[str, Any]
    mean_hu: dict[str, Any]
    skelet_muscle_index: float | None = None

    series_inst_uid: str | None = None
    contrast_phase: str | None = None

    duration: float | None = None
    centroids: Centroids = field(default_factory=Centroids)

    def _to_dict(self) -> dict[str, Any]:
        tissue_labels = self.area.keys()
        return (
            {
                "series_inst_uid": self.series_inst_uid,
                "contrast_phase": self.contrast_phase,
                "skelet_muscle_index": self.skelet_muscle_index,
                "duration": self.duration,
            }
            | {f"area_{label}": self.area[label] for label in tissue_labels}
            | {f"mean_hu_{label}": self.mean_hu[label] for label in tissue_labels}
        )

    def set_duration(self, *durations):
        self.duration = sum(durations)

    @classmethod
    def _from_dict(cls, d: dict[str, Any]) -> Self:
        return cls(
            area=d.get("area", {}),
            mean_hu=d.get("mean_hu", {}),
            skelet_muscle_index=d.get("skelet_muscle_index"),
            series_inst_uid=d.get("series_inst_uid"),
            contrast_phase=d.get("contrast_phase"),
            duration=d.get("duration"),
            centroids=Centroids(
                d.get("vertebre_centroid", []), d.get("body_centroid", [])
            ),
        )


@dataclass
class SegmentationResult:
    participant: str
    study_inst_uid: str
    labkey_row_id: str | None = None
    patient_height: float | None = None

    metrics_dict: dict[str, MetricsData] = field(default_factory=dict)

    @classmethod
    def _from_study_case(cls, study_data: StudyData) -> Self:
        return cls(
            participant=study_data.participant,
            study_inst_uid=study_data.study_inst_uid,
            labkey_row_id=study_data.labkey_row_id,
            patient_height=study_data.patient_height,
        )

    # TODO: _write_to_json() maybe add exclude_fields: list[str]
    def _write_to_json(self, output_dir: str | Path):
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)

        # TODO: if needed add fields to exclude and its default
        # exclude = {}
        # if exclude_fields:
        #     exclude.update(exclude_fields)

        serialized = asdict(
            self,
            dict_factory=lambda dic: {key: val for key, val in dic},
        )
        # if key not in exclude

        filepath = output_dir.joinpath(f"metrics_{self.study_inst_uid}.json")
        if filepath.exists():
            logger.info(f"overwriting existing file at `{str(filepath)}`")

        with open(filepath, mode="w", encoding="utf-8") as file:
            json.dump(serialized, file, indent=2)

        logger.info(
            f"written DICOM tags for participant {self.participant}, study instance uid {self.study_inst_uid}"
        )

    @classmethod
    def _from_json(cls, path: Path | str) -> Self:
        data = read_json(path)
        return cls(
            participant=data.get("participant"),
            study_inst_uid=data.get("study_inst_uid"),
            patient_height=data.get("patient_height"),
            labkey_row_id=data.get("labkey_row_id"),
            metrics_dict={
                uid: MetricsData._from_dict(metric)
                for uid, metric in data.get("metrics_dict").items()
            },
        )

    def _to_list_of_dicts(self) -> list[dict[str, Any]]:
        _base_dict = {
            "participant": self.participant,
            "labkey_row_id": self.labkey_row_id,
            "patient_height": self.patient_height,
            "study_inst_uid": self.study_inst_uid,
        }
        return [_base_dict | metric._to_dict() for metric in self.metrics_dict.values()]
