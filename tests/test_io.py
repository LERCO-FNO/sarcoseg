import pytest

from sarcosegv2.io import find_study_input


def test_find_study_input_dicom():
    result = find_study_input("tests/data", "1.2")
    assert result and result.is_dir(), f"Expected a DICOM directory, got `{result}`"


def test_find_study_input_nifti():
    result = find_study_input("tests/data", "1.3")
    assert result and result.is_file() and ".nii" in result.name, (
        f"Expected a NIfTI file, got `{result}`"
    )
