import enum
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Self

import pandas as pd

# from pydicom import dcmread
from SimpleITK import Image

from src.io import read_json
from src.labels import DEFAULT_TISSUE_CLASSES

log = logging.getLogger("classes")


@dataclass
class SeriesData:
    series_inst_uid: str
    series_description: str | None = None
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

    def _to_dict(self, exclude_fields: list[str] | None = None) -> dict[str, Any]:
        exclude = {"filepaths", "filepaths_num"}
        if exclude_fields:
            exclude.update(exclude_fields)

        return asdict(
            self,
            dict_factory=lambda dic: {
                key: val for key, val in dic if key not in exclude
            },
        )


@dataclass
class StudyData:
    participant: str
    study_inst_uid: str
    # patient_id: str | None = field(default=None, repr=False, compare=False)
    # study_date: str | None = field(default=None, repr=False, compare=False)
    patient_height: float | int | None = field(default=None, repr=False, compare=False)
    # patient_birthdate: str | None = field(default=None, repr=False, compare=False)
    series: dict[str, SeriesData] = field(default_factory=dict)
    # ct_relative_patient_age: int | None = field(default=None, repr=False, compare=False)

    @classmethod
    def _from_labkey_row(cls, row: dict[str, Any]) -> Self:
        return cls(
            participant=row.get("PARTICIPANT"),
            study_inst_uid=row.get("STUDY_INSTANCE_UID"),
            # patient_id=row.get("RODNE_CISLO", ""),
            patient_height=height
            if (height := row.get("VYSKA_PAC.", 0.0)) and height
            else 0.0,
            # patient_birthdate=row.get("CAS_VYSETRENI"),
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
            log.debug(f"overwriting existing file at `{str(filepath)}`")

        with open(filepath, mode="w", encoding="utf-8") as file:
            json.dump(serialized, file, indent=2)

        log.debug(
            f"written DICOM tags for {self.participant=}, {self.study_inst_uid=}\nfields excluded: {exclude}"
        )

    def _to_list_of_dicts(self) -> list[dict[str, Any]]:
        _study = {
            "participant": self.participant,
            "study_inst_uid": self.study_inst_uid,
            "patient_height": self.patient_height,
            # "study_date": self.study_date,
            # "ct_relative_patient_age": self.ct_relative_patient_age,
        }
        return [_study | series._to_dict() for series in self.series.values()]

"""
    def calculate_ct_relative_patient_age(self, dicom_filepath: Path | str):
        study_date = dcmread(dicom_filepath, specific_tags=["StudyDate"]).get(
            "StudyDate", ""
        )

        # this shouldn't happen, but just in case
        if not study_date:
            log.error(
                f"error parsing StudyDate `{study_date}` for patient {self.participant}, study {self.study_inst_uid}"
            )
            self.ct_relative_patient_age = -1
            return None
        ct_year = int(study_date[:4])  # parse year (YYYYMMDD) as int
        birth_year = int(
            self.patient_birthdate.split("-")[0]
        )  # parse year (YYYY-MM-DD HH:MM:SS.SSS) as int
        self.ct_relative_patient_age = ct_year - birth_year
"""

class ProcessResult(enum.Enum):
    NONE = "NONE"
    MISSING_L3_MASK = "MISSING_L3_MASK"
    SEGMENTATION_FINISHED = "SEGMENTATION_FINISHED"
    MISSING_ON_PACS_OR_LOCAL = "MISSING_ON_PACS_OR_LOCAl"
    NO_SERIES_TO_SEGMENT = "NO_SERIES_TO_SEGMENT"
    SEGMENTATION_SPINE_FAIL = "SEGMENTATION_SPINE_FAIL"
    SEGMENTATION_TISSUE_FAIL = "SEGMENTATION_TISSUE_FAIL"


@dataclass
class ImageData:
    image: Image
    path: Path


@dataclass
class Centroids:
    vertebre_centroid: list = field(default_factory=list)
    body_centroid: list = field(default_factory=list)


@dataclass
class ProcessDurations:
    spine_seg: int | float = 0.0
    tissue_seg: int | float = 0.0
    slice_extraction: int | float = 0.0
    postprocessing: int | float = 0.0


@dataclass
class Metrics:
    area: dict[str, Any] = field(default_factory=dict)
    mean_hu: dict[str, Any] = field(default_factory=dict)
    skelet_muscle_index: float | None = 0.0
    body_mass_index: float | None = 0.0

    process_durations: ProcessDurations = field(default_factory=ProcessDurations)
    total_duration: float | None = 0.0
    centroids: Centroids = field(default_factory=Centroids)
    l3_slice_index: int = -1
    l3_tube_current: int = -1
    image_z_size: int = -1

    def _to_dict(self) -> dict[str, Any]:
        tissue_labels = DEFAULT_TISSUE_CLASSES.keys()
        return (
            {
                "skelet_muscle_index": self.skelet_muscle_index,
                "body_mass_index": self.body_mass_index,
                "total_duration": self.total_duration,
                "L3_slice_index": self.l3_slice_index,
                "L3_tube_current": self.l3_tube_current,
                "image_z_size": self.image_z_size,
            }
            | {f"area_{label}": self.area.get(label) for label in tissue_labels}
            | {f"mean_hu_{label}": self.mean_hu.get(label) for label in tissue_labels}
        )

    def set_durations(self, durations: ProcessDurations):
        self.process_durations = durations
        self.total_duration = sum(durations.__dict__.values())

    def set_l3_tube_current(self, study_dir, series_inst_uid: str):
        with open(Path(study_dir, "inst_num_currents.json"), "r") as file:
            series_currents = json.load(file).get(series_inst_uid, None)

            if not series_currents:
                log.warning(f"series {series_inst_uid} has no tube currents")
                return

            self.l3_tube_current = series_currents.get(str(self.l3_slice_index), -1)
            if self.l3_tube_current == -1:
                log.warning(
                    f"series {series_inst_uid} has no tube current at L3 slice {self.l3_slice_index}"
                )

    @classmethod
    def _from_dict(cls, d: dict[str, Any]) -> Self:
        return cls(
            area=d.get("area", {}),
            mean_hu=d.get("mean_hu", {}),
            skelet_muscle_index=d.get("skelet_muscle_index", 0.0),
            process_durations=ProcessDurations(**d.get("process_durations", {})),
            total_duration=d.get("total_duration", 0.0),
            centroids=Centroids(
                d.get("vertebre_centroid", []), d.get("body_centroid", [])
            ),
        )


@dataclass
class SeriesSegmentationResult:
    series_inst_uid: str
    status: ProcessResult = field(default=ProcessResult.NONE)
    contrast_phase: str | None = None
    metrics: Metrics | None = field(default=None)


@dataclass
class StudySegmentationResult:
    participant: str
    study_inst_uid: str
    patient_height: float | None = None
    series_results: dict[str, SeriesSegmentationResult] = field(default_factory=dict)

    def add_result(self, result: SeriesSegmentationResult):
        if not result.metrics:
            log.warning(f"metrics for {result.series_inst_uid} is None")
        if result.status is ProcessResult.NONE:
            log.warning(
                f"result status for {result.series_inst_uid} is {result.status}"
            )

        log.debug(
            f"added series result {result.series_inst_uid} for study {self.study_inst_uid}"
        )

        self.series_results[result.series_inst_uid] = result

    @classmethod
    def _from_study_case(cls, study_data: StudyData) -> Self:
        return cls(
            participant=study_data.participant,
            study_inst_uid=study_data.study_inst_uid,
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

        # TODO: temporary fix for unserializable ProcessResult
        for result in self.series_results.values():
            result.status = (
                result.status.value
                if isinstance(result.status, ProcessResult)
                else result.status
            )

        serialized = asdict(
            self,
            dict_factory=lambda dic: {key: val for key, val in dic},
        )
        # if key not in exclude

        filepath = output_dir.joinpath(f"metrics_{self.study_inst_uid}.json")
        if filepath.exists():
            log.debug(f"overwriting existing file at `{str(filepath)}`")

        with open(filepath, mode="w", encoding="utf-8") as file:
            json.dump(serialized, file, indent=2)

        log.debug(
            f"written segmentation results for participant {self.participant}, study {self.study_inst_uid}"
        )

    @classmethod
    def _from_json(cls, path: Path | str) -> Self:
        data = read_json(path)
        return cls(
            participant=data.get("participant"),
            study_inst_uid=data.get("study_inst_uid"),
            patient_height=data.get("patient_height"),
            series_results={
                uid: SeriesSegmentationResult(
                    series_inst_uid=uid,
                    status=ProcessResult(result.get("status")),
                    contrast_phase=result.get("contrast_phase"),
                    metrics=Metrics._from_dict(metrics)
                    if (metrics := result.get("metrics"))
                    else None,
                )
                for uid, result in data.get("series_results", {}).items()
            },
        )

    def _to_list_of_dicts(self) -> list[dict[str, Any]]:
        study_base = {
            "participant": self.participant,
            "patient_height": self.patient_height,
            "study_inst_uid": self.study_inst_uid,
        }

        flattened_items = []

        for uid, result in self.series_results.items():
            results = {
                "status": result.status.value
                if isinstance(result.status, ProcessResult)
                else result.status,
                "series_inst_uid": result.series_inst_uid,
                "contrast_phase": result.contrast_phase,
            }
            metrics = result.metrics._to_dict() if result.metrics else {}

            combined = study_base | results | metrics

            flattened_items.append(combined)

        return flattened_items


@dataclass
class Report:
    timestamp: str
    data_rows: list[dict[str, Any]] = field(default_factory=list)

    def add_case(
        self,
        participant: str,
        study_instance_uid: str,
        process_result: ProcessResult | str,
        series_instance_uid: str | None = None,
    ):

        if isinstance(process_result, ProcessResult):
            process_result = process_result.value

        row: dict[str, Any] = {
            "participant": participant,
            "study_inst_uid": study_instance_uid,
            "series_inst_uid": series_instance_uid,
            "process_result": process_result,
        }
        self.data_rows.append(row)

    def write_report(self, directory: str | Path):
        df = pd.DataFrame(self.data_rows)
        df.to_csv(Path(directory, f"./report_{self.timestamp}.csv"), index=False)
