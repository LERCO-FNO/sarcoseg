from pathlib import Path

import pytest

from sarcosegv2.io import find_study_input

DICOM_DIR = "1.2"
NIFTI_FILE = "1.3"
BAD_STUDY_UID = "1.4"  # doesn't exist on purpose


def test_find_dicom_study(test_images_dir):
    path = find_study_input(test_images_dir, DICOM_DIR)
    assert path and path.is_dir(), f"Expected directory path, got {path}"


def test_find_nifti_study(test_images_dir):
    path = find_study_input(test_images_dir, NIFTI_FILE)
    assert path and path.is_file(), f"Expected file path, got {path}"


def test_study_not_found(test_images_dir):
    path = find_study_input(test_images_dir, BAD_STUDY_UID)
    assert path is None
