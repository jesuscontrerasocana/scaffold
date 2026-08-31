"""Your two modules must satisfy the interfaces the runner calls. Run this before you
submit: `pytest tests/test_contract.py`.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from pipeline.forecaster import Forecaster
from pipeline.optimizer import Optimizer
from pipeline.specs import SiteSpecs


SITE = Path(__file__).resolve().parent.parent / "site.yaml"


@pytest.fixture
def specs() -> SiteSpecs:
    return SiteSpecs.from_yaml(SITE)


def test_forecaster_has_the_methods_the_runner_calls(specs: SiteSpecs) -> None:
    forecaster = Forecaster(specs)
    for name in ("fit", "save", "predict"):
        assert callable(getattr(forecaster, name)), f"Forecaster.{name} is missing"
    assert callable(Forecaster.load), "Forecaster.load is missing"

    signature = inspect.signature(Forecaster.predict)
    for parameter in ("at_time", "history", "future_exog", "horizon"):
        assert parameter in signature.parameters, f"Forecaster.predict must accept {parameter}"


def test_optimizer_has_the_methods_the_runner_calls(specs: SiteSpecs) -> None:
    optimizer = Optimizer(specs)
    assert callable(optimizer.solve), "Optimizer.solve is missing"
    signature = inspect.signature(Optimizer.solve)
    for parameter in ("forecast", "prices", "context"):
        assert parameter in signature.parameters, f"Optimizer.solve must accept {parameter}"


def test_specs_are_readable(specs: SiteSpecs) -> None:
    assert specs.battery.capacity_kwh > 0
    assert 0 < specs.battery.round_trip_efficiency <= 1
    assert specs.battery.cycle_cost_eur > 0
    assert specs.offtake_monthly_peak_cost_eur_per_kw > 0
