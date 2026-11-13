from copy import deepcopy
from pathlib import Path

import pytest
from pydicom import Dataset, FileDataset, dcmread

DICOM_DIR = Path(__file__).parent / "data" / "dicom"


@pytest.fixture(scope="session")
def test_images_dir():
    """
    Returns path to the local DICOM/Nifti test data.
    Expects directory `./tests/data` at project root.
    """

    path = Path("./tests/data/")
    if not path.exists():
        pytest.fail(
            f"Local test image directory(ies) not found at `{path}`, "
            "expected to find <project_dir/dev/<study>"
        )

    return path


@pytest.fixture(scope="session")
def dicom_datasets(_dicom_datasets_cache) -> list[Dataset]:
    """
    Reads test DICOM files and returns a list of `pydicom.Dataset`.
    Expects test files at `./tests/data/dicom` at project root.
    """

    return [deepcopy(ds) for ds in _dicom_datasets_cache]


@pytest.fixture(scope="session")
def dicom_dir() -> Path:
    if not DICOM_DIR.is_dir():
        pytest.fail(f"Test DICOM directory not found: {DICOM_DIR}")
    return DICOM_DIR


@pytest.fixture(scope="session")
def _dicom_datasets_cache(dicom_dir: Path) -> tuple[FileDataset, ...]:

    files = tuple(dicom_dir.rglob("*.dcm"))
    if not files:
        pytest.fail(f"No dicom test files found in {dicom_dir}")

    return tuple(dcmread(f) for f in files)
