"""Writing results out and drawing them. Provided by us; extend it if you find it useful."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.harness import write_summary


def _figure(rows: int):  # noqa: ANN202
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt, plt.subplots(rows, 1, figsize=(14, 3 * rows), sharex=True)


def write_results(
    schedule: pd.DataFrame, summary: dict[str, Any], out_path: Path
) -> None:
    """What happened, what it cost, and a picture of it."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    schedule.to_csv(out_path)
    log = schedule.attrs.get("forecast_log")
    if log is not None and len(log):
        log.to_csv(out_path.with_name(out_path.stem + "_forecasts.csv"))
    write_summary(summary, out_path.with_name(out_path.stem + "_summary.json"))
    _plot_simulation(schedule, summary, out_path.with_name(out_path.stem + "_plot.png"))


def _plot_simulation(
    schedule: pd.DataFrame, summary: dict[str, Any], path: Path
) -> None:
    try:
        plt, (fig, axes) = _figure(4)
    except ImportError:
        return

    window = schedule.iloc[: 96 * 7]
    peak = max(v["peak_offtake_kw"] for v in summary["peak_cost_detail"].values())

    axes[0].plot(
        window.index, window["realized_net_no_bess_kw"], label="site, no battery", lw=1
    )
    axes[0].plot(
        window.index,
        window["grid_net_with_bess_kw"],
        label="at the meter, with battery",
        lw=1.2,
    )
    axes[0].axhline(
        peak, color="crimson", ls="--", lw=1, label=f"month peak {peak:.0f} kW"
    )
    axes[0].axhline(0, color="grey", lw=0.5)
    axes[0].set_ylabel("kW")
    axes[0].set_title("First week of the simulation")

    axes[1].plot(
        window.index, window["realized_net_no_bess_kw"], label="realized", lw=1
    )
    axes[1].plot(
        window.index, window["forecast_net_kw"], label="forecast", lw=1, alpha=0.8
    )
    axes[1].set_ylabel("kW")

    axes[2].fill_between(
        window.index, 0, window["applied_charge_kw"], label="charge", alpha=0.7
    )
    axes[2].fill_between(
        window.index, 0, -window["applied_discharge_kw"], label="discharge", alpha=0.7
    )
    ax_soc = axes[2].twinx()
    ax_soc.plot(
        window.index, window["soc"], color="black", lw=1, label="state of charge"
    )
    ax_soc.set_ylim(0, 1)
    ax_soc.set_ylabel("state of charge")
    axes[2].set_ylabel("kW")

    axes[3].plot(
        window.index, window["offtake_price_eur_per_mwh"], label="offtake EUR/MWh", lw=1
    )
    axes[3].plot(
        window.index,
        window["injection_price_eur_per_mwh"],
        label="injection EUR/MWh",
        lw=1,
    )
    axes[3].axhline(0, color="grey", lw=0.5)
    axes[3].set_ylabel("EUR/MWh")

    for ax in axes:
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(
        f"bill {summary['total_cost_eur']:.0f} EUR "
        f"= energy {summary['energy_cost_eur']:.0f} + capacity {summary['peak_cost_eur']:.0f} "
        f"+ cycles {summary['cycle_cost_eur']:.0f}   |   {summary['equivalent_cycles']:.1f} cycles"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
