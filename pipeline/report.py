"""Write simulation artifacts and customer-facing performance charts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.data import HOURS_PER_STEP
from pipeline.harness import write_summary


def _plt():  # noqa: ANN202
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _figure(rows: int):  # noqa: ANN202
    plt = _plt()
    return plt, plt.subplots(rows, 1, figsize=(14, 3 * rows), sharex=True)


def _peak_rate(summary: dict[str, Any]) -> float:
    for item in summary.get("peak_cost_detail", {}).values():
        peak, share = float(item["peak_offtake_kw"]), float(item["month_share"])
        if peak > 0 and share > 0:
            return float(item["peak_cost_eur"]) / (peak * share)
    return 0.0


def _running_peak_cost(
    data: pd.DataFrame,
    import_column: str,
    summary: dict[str, Any],
    rate: float,
) -> pd.Series:
    """Apply the harness's monthly peak proration as each month progresses."""
    result = pd.Series(index=data.index, dtype=float)
    billed_before_month = 0.0
    months = pd.PeriodIndex(data.index.tz_localize(None), freq="M")
    details = summary.get("peak_cost_detail", {})
    for period, group in data.groupby(months):
        detail = details.get(str(period), {})
        final_share = float(detail.get("month_share", 1.0))
        steps_in_month = len(group) / final_share if final_share > 0 else len(group)
        elapsed_share = pd.Series(
            range(1, len(group) + 1), index=group.index, dtype=float
        ) / steps_in_month
        current_month_cost = (
            group[import_column].cummax() * rate * elapsed_share
        )
        result.loc[group.index] = billed_before_month + current_month_cost
        billed_before_month = float(result.loc[group.index].iloc[-1])
    return result


def calculate_dashboard_metrics(
    schedule: pd.DataFrame, summary: dict[str, Any]
) -> tuple[dict[str, float | None], pd.DataFrame]:
    """Calculate bills explicitly; unavailable realized load/PV remain ``None``."""
    data = schedule.copy()
    no_bess = pd.to_numeric(data["realized_net_no_bess_kw"])
    with_bess = pd.to_numeric(data["grid_net_with_bess_kw"])
    offtake_price = pd.to_numeric(data["offtake_price_eur_per_mwh"])
    injection_price = pd.to_numeric(data["injection_price_eur_per_mwh"])
    for name, net in (("without_bess", no_bess), ("with_bess", with_bess)):
        data[f"grid_import_{name}_kw"] = net.clip(lower=0)
        data[f"grid_export_{name}_kw"] = (-net).clip(lower=0)
        data[f"offtake_cost_{name}_eur"] = data[f"grid_import_{name}_kw"] * offtake_price * HOURS_PER_STEP / 1000
        data[f"injection_revenue_{name}_eur"] = data[f"grid_export_{name}_kw"] * injection_price * HOURS_PER_STEP / 1000
        data[f"energy_cost_{name}_eur"] = data[f"offtake_cost_{name}_eur"] - data[f"injection_revenue_{name}_eur"]

    throughput = (pd.to_numeric(data["applied_charge_kw"]) + pd.to_numeric(data["applied_discharge_kw"])) * HOURS_PER_STEP
    cycling_cost = float(summary.get("cycle_cost_eur", 0.0))
    data["cycling_cost_eur"] = 0.0 if throughput.sum() == 0 else throughput * cycling_cost / throughput.sum()
    rate = _peak_rate(summary)
    for name in ("without_bess", "with_bess"):
        data[f"running_peak_{name}_kw"] = data[f"grid_import_{name}_kw"].cummax()
        data[f"running_peak_cost_{name}_eur"] = _running_peak_cost(
            data, f"grid_import_{name}_kw", summary, rate
        )
    data["cumulative_bill_without_bess_eur"] = data["energy_cost_without_bess_eur"].cumsum() + data["running_peak_cost_without_bess_eur"]
    data["cumulative_bill_with_bess_eur"] = data["energy_cost_with_bess_eur"].cumsum() + data["running_peak_cost_with_bess_eur"] + data["cycling_cost_eur"].cumsum()

    energy_without = float(data["energy_cost_without_bess_eur"].sum())
    energy_with = float(data["energy_cost_with_bess_eur"].sum())
    peak_without = float(data["running_peak_cost_without_bess_eur"].iloc[-1])
    peak_with = float(data["running_peak_cost_with_bess_eur"].iloc[-1])
    bill_without = energy_without + peak_without
    bill_with = energy_with + peak_with + cycling_cost
    savings = bill_without - bill_with
    load = float(pd.to_numeric(data["load_kw"]).sum()) * HOURS_PER_STEP / 1000 if "load_kw" in data else None
    pv = float(pd.to_numeric(data["pv_production_kw"]).sum()) * HOURS_PER_STEP / 1000 if "pv_production_kw" in data else None
    metrics = {
        "total_savings_eur": savings,
        "bill_without_bess_eur": bill_without, "bill_with_bess_eur": bill_with,
        "total_load_mwh": load, "pv_production_mwh": pv,
        "grid_offtake_with_bess_mwh": float(data["grid_import_with_bess_kw"].sum()) * HOURS_PER_STEP / 1000,
        "grid_injection_with_bess_mwh": float(data["grid_export_with_bess_kw"].sum()) * HOURS_PER_STEP / 1000,
        "equivalent_full_cycles": float(summary.get("equivalent_cycles", 0.0)),
        "energy_cost_with_bess_eur": energy_with, "peak_cost_with_bess_eur": peak_with,
        "battery_cycling_cost_eur": cycling_cost,
        "monthly_peak_with_bess_kw": float(data["grid_import_with_bess_kw"].max()),
        "energy_cost_value_eur": energy_without - energy_with,
        "peak_cost_value_eur": peak_without - peak_with,
        "cycling_cost_value_eur": -cycling_cost,
    }
    return metrics, data


def write_results(schedule: pd.DataFrame, summary: dict[str, Any], out_path: Path) -> None:
    """Write existing artifacts plus the monthly dashboard."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    schedule.to_csv(out_path)
    log = schedule.attrs.get("forecast_log")
    if log is not None and len(log):
        log.to_csv(out_path.with_name(out_path.stem + "_forecasts.csv"))
    write_summary(summary, out_path.with_name(out_path.stem + "_summary.json"))
    _plot_simulation(schedule, summary, out_path.with_name(out_path.stem + "_plot.png"))
    _plot_dashboard(schedule, summary, out_path.with_name(out_path.stem + "_dashboard.png"))


def _plot_simulation(schedule: pd.DataFrame, summary: dict[str, Any], path: Path) -> None:
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


def _display(value: float | None, unit: str = "") -> str:
    return "N/A" if value is None else f"{value:,.1f} {unit}".strip()


def _plot_dashboard(
    schedule: pd.DataFrame,
    summary: dict[str, Any],
    path: Path,
) -> None:
    try:
        plt = _plt()
    except ImportError:
        return

    metrics, data = calculate_dashboard_metrics(schedule, summary)

    # Average quoted prices over the reporting period.
    avg_offtake_price = data["offtake_price_eur_per_mwh"].mean()
    avg_injection_price = data["injection_price_eur_per_mwh"].mean()

    fig = plt.figure(figsize=(16, 13))
    grid = fig.add_gridspec(
        4,
        2,
        height_ratios=(1.3, 2, 2, 2),
        width_ratios=(0.85, 1.15),
    )

    cards = fig.add_subplot(grid[0, :])
    cumulative = fig.add_subplot(grid[1, :])
    value = fig.add_subplot(grid[2:, 0])

    ops_grid = grid[2:, 1].subgridspec(4, 1, hspace=0.15)
    ops = [fig.add_subplot(ops_grid[i]) for i in range(4)]

    # ------------------------------------------------------------------
    # KPI table
    # ------------------------------------------------------------------
    cards.axis("off")
    cards.set_xlim(0, 5)
    cards.set_ylim(0, 2)

    items = [
        ("Savings", _display(metrics["total_savings_eur"], "EUR")),
        (
            "Bill without BESS",
            _display(metrics["bill_without_bess_eur"], "EUR"),
        ),
        (
            "Bill with BESS",
            _display(metrics["bill_with_bess_eur"], "EUR"),
        ),
        ("Load", _display(metrics["total_load_mwh"], "MWh")),
        ("PV", _display(metrics["pv_production_mwh"], "MWh")),
        (
            "Grid offtake",
            _display(metrics["grid_offtake_with_bess_mwh"], "MWh"),
        ),
        (
            "Grid injection",
            _display(metrics["grid_injection_with_bess_mwh"], "MWh"),
        ),
        (
            "Cycles",
            _display(metrics["equivalent_full_cycles"]),
        ),
        (
            "Avg. offtake price",
            _display(avg_offtake_price, "EUR/MWh"),
        ),
        (
            "Avg. injection price",
            _display(avg_injection_price, "EUR/MWh"),
        ),
    ]

    # Light table separators.
    for x in range(1, 5):
        cards.plot(
            [x, x],
            [0.08, 1.92],
            color="0.82",
            lw=0.8,
        )

    cards.plot(
        [0.05, 4.95],
        [1, 1],
        color="0.82",
        lw=0.8,
    )

    # KPI text.
    for i, (label, text) in enumerate(items):
        col = i % 5
        row = 1 - i // 5

        x = col + 0.5
        y = row + 0.5

        cards.text(
            x,
            y + 0.16,
            label,
            ha="center",
            va="center",
            fontsize=11,
        )

        cards.text(
            x,
            y - 0.05,
            text,
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
        )

    # Small bill composition note inside the "Bill with" cell.
    cards.text(
        2.5,
        1.15,
        (
            f"Energy {_display(metrics['energy_cost_with_bess_eur'], 'EUR')}  ·  "
            f"Peak {_display(metrics['peak_cost_with_bess_eur'], 'EUR')}  ·  "
            f"Cycling {_display(metrics['battery_cycling_cost_eur'], 'EUR')}"
        ),
        ha="center",
        va="center",
        fontsize=7.5,
        color="0.35",
    )

    # ------------------------------------------------------------------
    # Cumulative bill
    # ------------------------------------------------------------------
    cumulative.plot(
        data.index,
        data["cumulative_bill_without_bess_eur"],
        label="without BESS",
    )
    cumulative.plot(
        data.index,
        data["cumulative_bill_with_bess_eur"],
        label="with BESS",
    )
    cumulative.set_title("Cumulative bill")
    cumulative.set_ylabel("EUR")
    cumulative.legend()
    cumulative.grid(alpha=0.2)

    # ------------------------------------------------------------------
    # Value decomposition
    # ------------------------------------------------------------------
    values = [
        metrics["energy_cost_value_eur"],
        metrics["peak_cost_value_eur"],
        metrics["cycling_cost_value_eur"],
    ]

    value.bar(
        ["Energy", "Peak", "Cycling"],
        values,
        color=[
            "#2a9d8f" if x >= 0 else "#e76f51"
            for x in values
        ],
    )
    value.axhline(0, color="black", lw=0.8)
    value.set_title(
        f"Value decomposition: "
        f"{metrics['total_savings_eur']:.1f} EUR"
    )

    # ------------------------------------------------------------------
    # Last complete day
    # ------------------------------------------------------------------
    days = [
        group
        for _, group in data.groupby(data.index.normalize())
        if len(group) == 96
    ]
    last = days[-1] if days else data.iloc[0:0]

    if last.empty:
        for ax in ops:
            ax.text(
                0.5,
                0.5,
                "No complete day available",
                ha="center",
            )
            ax.set_axis_off()

    else:
        # Site vs meter.
        ops[0].plot(
            last.index,
            last["realized_net_no_bess_kw"],
            label="site net",
        )
        ops[0].plot(
            last.index,
            last["grid_net_with_bess_kw"],
            label="meter",
        )
        ops[0].axhline(
            0,
            color="black",
            lw=0.6,
            alpha=0.5,
        )
        ops[0].set_ylabel("kW")
        ops[0].legend(fontsize=8)
        ops[0].set_title("Last complete day")

        # Prices.
        ops[1].plot(
            last.index,
            last["offtake_price_eur_per_mwh"],
            label="offtake price",
        )
        ops[1].plot(
            last.index,
            last["injection_price_eur_per_mwh"],
            label="injection price",
        )
        ops[1].set_ylabel("EUR/MWh")
        ops[1].legend(fontsize=8)
        ops[1].grid(alpha=0.15)

        # Battery power:
        # positive = charging, negative = discharging.
        battery_power_kw = (
            last["applied_charge_kw"]
            - last["applied_discharge_kw"]
        )

        bar_width_days = 12 / (24 * 60)

        ops[2].bar(
            last.index,
            battery_power_kw,
            width=bar_width_days,
            label="battery power",
        )

        charge_limit_kw = 200
        discharge_limit_kw = 200

        ops[2].axhline(
            charge_limit_kw,
            linestyle="--",
            linewidth=0.9,
            label="charge limit",
        )
        ops[2].axhline(
            -discharge_limit_kw,
            linestyle="--",
            linewidth=0.9,
            label="discharge limit",
        )
        ops[2].axhline(
            0,
            color="black",
            lw=0.6,
        )
        ops[2].set_ylabel("kW")
        ops[2].legend(
            fontsize=8,
            ncol=2,
        )
        ops[2].grid(
            axis="y",
            alpha=0.15,
        )

        # State of charge.
        soc_pct = 100 * last["soc"]

        ops[3].plot(
            last.index,
            soc_pct,
        )
        ops[3].axhline(
            5,
            linestyle="--",
            linewidth=0.9,
            label="min SoC",
        )
        ops[3].axhline(
            95,
            linestyle="--",
            linewidth=0.9,
            label="max SoC",
        )
        ops[3].set_ylabel("SoC (%)")
        ops[3].set_ylim(0, 100)
        ops[3].legend(fontsize=8)
        ops[3].grid(alpha=0.15)

        for ax in ops[:-1]:
            ax.tick_params(labelbottom=False)

    fig.suptitle(
        "Monthly BESS performance dashboard",
        fontsize=16,
    )

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--schedule", type=Path, required=True); parser.add_argument("--summary", type=Path); parser.add_argument("--log", type=Path); parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv); summary_path = args.summary or args.schedule.with_name(args.schedule.stem + "_summary.json"); log_path = args.log or args.schedule.with_name(args.schedule.stem + "_forecasts.csv")
    schedule = pd.read_csv(args.schedule, index_col=0, parse_dates=True)
    if log_path.exists(): schedule.attrs["forecast_log"] = pd.read_csv(log_path, index_col=0, parse_dates=True)
    write_results(schedule, json.loads(summary_path.read_text()), args.out or args.schedule); return 0


if __name__ == "__main__":
    raise SystemExit(main())
