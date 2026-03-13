"""Pytest configuration."""

import numpy
import pytest


@pytest.fixture(autouse=True)
def doctest_config(doctest_namespace: dict[str, object]) -> None:
    """Add numpy to doctest namespace."""
    numpy.set_printoptions(suppress=True, precision=2)
    doctest_namespace['numpy'] = numpy
    doctest_namespace['assert_array_equal'] = numpy.testing.assert_array_equal


# mypy: ignore-errors
