from pathlib import Path
from typing import NamedTuple

type StrPath = str | Path


class SeriesIdentity(NamedTuple):
    """
    Tuple of series name and series uid passed as element of series to convert from DICOM to nifti.
    Constructed from `SeriesMetadata`.
    """

    name: str
    uid: str
