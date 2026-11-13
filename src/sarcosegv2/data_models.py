from dataclasses import dataclass, field, fields
from pathlib import Path

import polars as pl
from loguru import logger as log

from sarcosegv2.types import StrPath


@dataclass(repr=False)
class StudyMetadata:
    participant: str
    study_uid: str
    patient_height: float | int | None = field(default=None, compare=False)
    series: dict[str, SeriesMetadata] = field(default_factory=dict, compare=False)

    @classmethod
    def from_dataframe(cls, df: pl.DataFrame):
        """
        Construct a list of `StudyMetadata` from `polars.DataFrame`.
        Requires `polars.DataFrame` to have non-empty `study_uid` and `participant` columns, case insensitive.

        Returns
        -------
        list(StudyMetadata)
        """
        return NotImplementedError

    def series_by_name(self, name: str):
        """
        Convenience method to access series by `name` key.

        Returns
        -------
        SeriesMetadata or None
        """
        series = self.series.get(name)
        if not series:
            log.warning(f"StudyMetadata has no series by {name}")
        return series

    def write_to_json(self, output_dir: StrPath):
        output_dir = Path(output_dir)


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
