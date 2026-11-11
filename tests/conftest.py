from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def test_images_dir():
    """
    Returns path to the local DICOM/Nifti test data.
    Expects directory `./tests/data` at project root.
    """

    path = Path("./tests/data/")
    if not path.exists():
        pytest.fail(
            f"Local test image directory(ies) not found: {path}, "
            "expected to find <project_dir/dev/<study>"
        )

    return path


# def pytest_addoption(parser):
#     parser.addoption(
#         "--study",
#         action="store",
#         default="03",
#     )
