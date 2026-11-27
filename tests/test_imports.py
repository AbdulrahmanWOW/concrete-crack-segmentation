"""Smoke test: every module in src/ can be imported."""

import importlib

import pytest

MODULES = [
    "data", "metrics", "baselines", "enhancement", "restoration", "frequency",
    "segmentation", "morphology", "pipeline", "evaluate", "run_pipeline",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    """Each src module imports without errors."""
    importlib.import_module(f"src.{name}")
