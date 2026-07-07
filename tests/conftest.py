"""Shared pytest fixtures."""
import os

import pytest

from context_engineering.loaders import build_engine

EXAMPLES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "context_engineering",
    "examples",
)


@pytest.fixture
def examples_dir():
    return EXAMPLES_DIR


@pytest.fixture
def engine(examples_dir):
    """A fully-loaded engine with all four example domains registered."""
    return build_engine(examples_dir)
