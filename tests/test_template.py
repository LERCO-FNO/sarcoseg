import pytest


def test_func():
    assert 1 + 1 == 2  # success


def test_bad():
    assert 1 + 2 != 2  # failure
