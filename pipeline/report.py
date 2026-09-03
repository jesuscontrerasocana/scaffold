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


def _peak_rate(summary: dict[str, Any]) -> float:
    for item in summary.get("peak_cost_detail", {}).values():
        peak, share = float(item["peak_offtake_kw"]), float(item["month_share"])
        if peak > 0 and share > 0:
            return float(item["peak_cost_eur"]) / (peak * share)
    return 0.0


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
        data[f"running_peak_cost_{name}_eur"] = data[f"running_peak_{name}_kw"] * rate
    data["cumulative_bill_without_bess_eur"] = data["energy_cost_without_bess_eur"].cumsum() + data["running_peak_cost_without_bess_eur"]
    data["cumulative_bill_with_bess_eur"] = data["energy_cost_with_bess_eur"].cumsum() + data["running_peak_cost_with_bess_eur"] + data["cycling_cost_eur"].cumsum()

    energy_without = float(data["energy_cost_without_bess_eur"].sum())
    energy_with = float(data["energy_cost_with_bess_eur"].sum())
    peak_without = float(data["grid_import_without_bess_kw"].max()) * rate
    peak_with = float(data["grid_import_with_bess_kw"].max()) * rate
    bill_without = energy_without + peak_without
    bill_with = energy_with + peak_with + cycling_cost
    savings = bill_without - bill_with
    load = float(pd.to_numeric(data["load_kw"]).sum()) * HOURS_PER_STEP / 1000 if "load_kw" in data else None
    pv = float(pd.to_numeric(data["pv_production_kw"]).sum()) * HOURS_PER_STEP / 1000 if "pv_production_kw" in data else None
    metrics = {
        "total_savings_eur": savings,
        "savings_rate_pct": None if bill_without == 0 else 100 * savings / bill_without,
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
        plt = _plt()
    except ImportError:
        return
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    window = schedule.iloc[: 96 * 7]
    axes[0].plot(window.index, window["realized_net_no_bess_kw"], label="site, no battery")
    axes[0].plot(window.index, window["grid_net_with_bess_kw"], label="meter, with battery")
    axes[1].plot(window.index, window["realized_net_no_bess_kw"], label="realized")
    axes[1].plot(window.index, window["forecast_net_kw"], label="forecast")
    axes[2].plot(window.index, window["applied_charge_kw"], label="charge")
    axes[2].plot(window.index, -window["applied_discharge_kw"], label="discharge")
    axes[3].plot(window.index, window["offtake_price_eur_per_mwh"], label="offtake EUR/MWh")
    axes[3].plot(window.index, window["injection_price_eur_per_mwh"], label="injection EUR/MWh")
    for ax in axes:
        ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def _display(value: float | None, unit: str = "") -> str:
    return "N/A" if value is None else f"{value:,.1f} {unit}".strip()


def _plot_dashboard(schedule: pd.DataFrame, summary: dict[str, Any], path: Path) -> None:
    try:
        plt = _plt()
    except ImportError:
        return
    metrics, data = calculate_dashboard_metrics(schedule, summary)
    fig = plt.figure(figsize=(16, 13)); grid = fig.add_gridspec(4, 2, height_ratios=(1.2, 2, 2, 2))
    cards = fig.add_subplot(grid[0, :]); cumulative = fig.add_subplot(grid[1, :]); value = fig.add_subplot(grid[2:, 0])
    ops_grid = grid[2:, 1].subgridspec(3, 1); ops = [fig.add_subplot(ops_grid[i]) for i in range(3)]
    cards.axis("off")
    items = [("Savings", _display(metrics["total_savings_eur"], "EUR")), ("Savings rate", _display(metrics["savings_rate_pct"], "%")), ("Bill without", _display(metrics["bill_without_bess_eur"], "EUR")), ("Bill with", _display(metrics["bill_with_bess_eur"], "EUR")), ("Load", _display(metrics["total_load_mwh"], "MWh")), ("PV", _display(metrics["pv_production_mwh"], "MWh")), ("Grid offtake", _display(metrics["grid_offtake_with_bess_mwh"], "MWh")), ("Grid injection", _display(metrics["grid_injection_with_bess_mwh"], "MWh")), ("Cycles", _display(metrics["equivalent_full_cycles"]))]
    for i, (label, text) in enumerate(items):
        cards.text((i % 5) / 5 + .01, .85 - (i // 5) * .45, f"{label}\n{text}", va="top")
    cards.text(.61, .12, f"Bill: energy {_display(metrics['energy_cost_with_bess_eur'], 'EUR')} | peak {_display(metrics['peak_cost_with_bess_eur'], 'EUR')} | cycling {_display(metrics['battery_cycling_cost_eur'], 'EUR')}")
    cumulative.plot(data.index, data["cumulative_bill_without_bess_eur"], label="without BESS"); cumulative.plot(data.index, data["cumulative_bill_with_bess_eur"], label="with BESS"); cumulative.set_title("Cumulative bill"); cumulative.set_ylabel("EUR"); cumulative.legend(); cumulative.grid(alpha=.2)
    values = [metrics["energy_cost_value_eur"], metrics["peak_cost_value_eur"], metrics["cycling_cost_value_eur"]]
    value.bar(["Energy", "Peak", "Cycling"], values, color=["#2a9d8f" if x >= 0 else "#e76f51" for x in values]); value.axhline(0, color="black", lw=.8); value.set_title(f"Value decomposition: {metrics['total_savings_eur']:.1f} EUR")
    days = [group for _, group in data.groupby(data.index.normalize()) if len(group) == 96]
    last = days[-1] if days else data.iloc[0:0]
    if last.empty:
        for ax in ops: ax.text(.5, .5, "No complete day available", ha="center"); ax.set_axis_off()
    else:
        ops[0].plot(last.index, last["realized_net_no_bess_kw"], label="site net"); ops[0].plot(last.index, last["grid_net_with_bess_kw"], label="meter"); ops[0].legend(fontsize=8); ops[0].set_title("Last complete day")
        ops[1].plot(last.index, last["applied_charge_kw"], label="charge"); ops[1].plot(last.index, -last["applied_discharge_kw"], label="discharge"); ops[1].legend(fontsize=8)
        ops[2].plot(last.index, 100 * last["soc"]); ops[2].set_ylabel("SoC (%)"); ops[2].set_ylim(0, 100)
    fig.suptitle("Monthly BESS performance dashboard", fontsize=16); fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--schedule", type=Path, required=True); parser.add_argument("--summary", type=Path); parser.add_argument("--log", type=Path); parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv); summary_path = args.summary or args.schedule.with_name(args.schedule.stem + "_summary.json"); log_path = args.log or args.schedule.with_name(args.schedule.stem + "_forecasts.csv")
    schedule = pd.read_csv(args.schedule, index_col=0, parse_dates=True)
    if log_path.exists(): schedule.attrs["forecast_log"] = pd.read_csv(log_path, index_col=0, parse_dates=True)
    write_results(schedule, json.loads(summary_path.read_text()), args.out or args.schedule); return 0


if __name__ == "__main__":
    raise SystemExit(main())
