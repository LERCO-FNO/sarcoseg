import pytest

from sarcosegv2.data_io import find_study_input, read_study_cases_list

DICOM_DIR = "1.2"
NIFTI_FILE = "1.3"
BAD_STUDY_UID = "1.4"  # doesn't exist on purpose


def test_study_metadata_from_csv():
    cases = read_study_cases_list("./tests/data/test_participants.csv")
    assert len(cases) == 3, f"Expected 3 rows, got {len(cases)}"
    assert not cases.get_column("participant").str.contains("PT04").any(), (
        "Expected `PT04` to be missing"
    )


def test_study_metadata_from_json():
    cases = read_study_cases_list("./tests/data/test_participants.json")
    assert len(cases) == 3, f"Expected 3 rows, got {len(cases)}"
    assert not cases.get_column("participant").str.contains("PT04").any(), (
        "Expected `PT04` to be missing"
    )


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
