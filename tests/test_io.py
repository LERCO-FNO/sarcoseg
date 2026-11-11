import pytest

from sarcosegv2.io import find_study_input

DICOM_DIR = "1.2"
NIFTI_FILE = "1.3"
BAD_STUDY_UID = "1.4"  # doesn't exist on purpose


def test_find_study_input_dicom():
    result = find_study_input("tests/data", DICOM_DIR)
    assert result and result.is_dir(), f"Expected a DICOM directory, got `{result}`"


def test_find_study_input_nifti():
    result = find_study_input("tests/data", NIFTI_FILE)
    assert result and result.is_file() and ".nii" in result.name, (
        f"Expected a NIfTI file, got `{result}`"
    )


def test_study_not_found(test_images_dir):
    path = find_study_input(test_images_dir, BAD_STUDY_UID)
    assert path is None
