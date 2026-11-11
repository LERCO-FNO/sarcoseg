import json

import pytest
from pydicom.dataset import Dataset

from sarcosegv2.series_filter import filter_cfind_series_responses


def test_filter_cfind_response():
    datasets = []
    with open("tests/test_cfind_response.json", mode="r") as file:
        for ds in json.load(file):
            datasets.append(Dataset.from_json(ds))
    (datasets_by_uid, dose_reports) = filter_cfind_series_responses(datasets)
    assert len(datasets_by_uid) == 1, (
        f"Expected 1 filtered series, got {len(datasets_by_uid)}"
    )
    assert dose_reports and len(dose_reports) >= 1
    series_uids = set(datasets_by_uid.keys())
    test_uids = {"1.2.2"}
    assert series_uids == test_uids, f"Expected {test_uids} UIDs, got {series_uids}"
