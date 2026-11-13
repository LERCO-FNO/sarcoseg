import pytest

from sarcosegv2.preprocess import extract_series_metadata, select_series_to_segment
from sarcosegv2.series_filter import filter_cfind_series_responses


def test_extract_series_metadata(dicom_datasets):
    filtered_datasets, _ = filter_cfind_series_responses(dicom_datasets)
    series_metadatas = extract_series_metadata(filtered_datasets)

    expected_contrast_types = ("native", "contrast")
    assert all([s.contrast_type in expected_contrast_types for s in series_metadatas])


def test_filter_datasets(dicom_datasets):

    filtered_datasets, dose_reports = filter_cfind_series_responses(dicom_datasets)
    assert len(filtered_datasets) == 3, (
        f"Expected 3 filtered datasets, got {len(filtered_datasets)}"
    )
