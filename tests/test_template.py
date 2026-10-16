import pytest


def test_good_func():
    assert 1 + 1 == 2  # good result


def test_bad_func():
    assert 1 + 1 == 3  # wrong result
