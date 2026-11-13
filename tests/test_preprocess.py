import math

import pytest
from pydicom import Dataset

from sarcosegv2.preprocess import (
    extract_series_metadata,
    extract_tube_currents,
    select_series_to_segment,
)
from sarcosegv2.series_filter import filter_cfind_series_responses


def test_extract_series_metadata(dicom_datasets):
    filtered_datasets, _ = filter_cfind_series_responses(dicom_datasets)
    series_metadatas, _ = extract_series_metadata(filtered_datasets)

    expected_contrast_types = ("native", "contrast")
    assert all([s.contrast_type in expected_contrast_types for s in series_metadatas])


def test_filter_datasets(dicom_datasets):

    filtered_datasets, dose_reports = filter_cfind_series_responses(dicom_datasets)
    assert len(filtered_datasets) == 3, (
        f"Expected 3 filtered datasets, got {len(filtered_datasets)}"
    )


def test_compute_mean_tube_current():
    datasets = [
        Dataset.from_json(
            {
                "00200013": {
                    "vr": "IS",
                    "Value": [str(i)],
                },
                "00181151": {"vr": "IS", "Value": [str(i)]},
            }
        )
        for i in range(1, 10 + 1)
    ]
    mean_current, currents = extract_tube_currents(datasets)
    assert mean_current == 5.5, f"Expected mean=5.5, got {mean_current}"
    # test correct filtering of None/"" in raw XRayTubeCurrent value
    assert len(currents) == len(datasets), (
        f"Expected {len(datasets)}, got {len(currents)}"
    )

    # modify to test dataset with None/"" in raw XRayTubeCurrent value
    datasets[0].XRayTubeCurrent = None
    datasets[1].XRayTubeCurrent = ""
    bad_mean, bad_currents = extract_tube_currents(datasets)
    assert bad_mean == 6.5, f"Expected mean = 6.5, got {bad_mean}"

    # test correct filtering of None/"" in raw XRayTubeCurrent value
    assert len(bad_currents) != len(datasets), f"Expected 8, got {len(bad_currents)}"

    # also test if all datasets have None/"" in raw XRayTubeCurrent value
    for ds in datasets:
        ds.XRayTubeCurrent = None
    none_mean, none_currents = extract_tube_currents(datasets)
    assert math.isnan(none_mean)
    assert none_currents == {}
