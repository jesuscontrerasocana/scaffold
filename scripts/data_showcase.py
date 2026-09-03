from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DATA_PATH = Path("data/history.csv")
OUT_DIR = Path("out/data_figures")

OFFTAKE_PEAK_COST_EUR_PER_KW = 4.25
HOURS_PER_STEP = 0.25

def plot_monthly_bill(monthly_bills: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.bar(
        monthly_bills[1:-1].index.astype(str),
        monthly_bills[1:-1]["total_bill_eur"],
    )

    ax.set_ylabel("Total bill [€]")
    ax.set_title("Monthly electricity bill without battery")
    ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "05_monthly_bill.png", dpi=180)
    plt.close(fig)

def calculate_monthly_bills(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    data["grid_import_kw"] = data["grid_net_kw"].clip(lower=0)
    data["grid_export_kw"] = (-data["grid_net_kw"]).clip(lower=0)

    data["offtake_cost_eur"] = (
        data["grid_import_kw"]
        * data["offtake_price_eur_per_mwh"]
        * HOURS_PER_STEP
        / 1000
    )

    data["injection_revenue_eur"] = (
        data["grid_export_kw"]
        * data["injection_price_eur_per_mwh"]
        * HOURS_PER_STEP
        / 1000
    )

    monthly = data.groupby(data.index.to_period("M")).agg(
        energy_offtake_cost_eur=("offtake_cost_eur", "sum"),
        injection_revenue_eur=("injection_revenue_eur", "sum"),
        peak_offtake_kw=("grid_import_kw", "max"),
    )

    monthly["peak_cost_eur"] = (
        monthly["peak_offtake_kw"] * OFFTAKE_PEAK_COST_EUR_PER_KW
    )

    monthly["total_bill_eur"] = (
        monthly["energy_offtake_cost_eur"]
        - monthly["injection_revenue_eur"]
        + monthly["peak_cost_eur"]
    )

    return monthly

def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep=";",
        decimal=",",
        parse_dates=["datetime_utc"],
    )
    df = df.set_index("datetime_utc").sort_index()

    # Useful derived quantity: site consumption before PV
    df["load_kw"] = df["grid_net_kw"] + df["pv_production_kw"]

    return df


def print_summary(df: pd.DataFrame) -> None:
    print(f"Period: {df.index.min()} -> {df.index.max()}")
    print(f"Observations: {len(df):,}")
    print()
    print(f"Mean load:       {df['load_kw'].mean():.1f} kW")
    print(f"Peak load:       {df['load_kw'].max():.1f} kW")
    print(f"Mean PV:         {df['pv_production_kw'].mean():.1f} kW")
    print(f"Peak PV:         {df['pv_production_kw'].max():.1f} kW")
    print(f"Peak net import: {df['grid_net_kw'].max():.1f} kW")
    print(f"Peak injection:  {-df['grid_net_kw'].min():.1f} kW")
    print(f"Total offtake: {df['grid_net_kw'].clip(lower=0).sum()* HOURS_PER_STEP:.1f} kWh")
    print(f"Total injection:  {(-df['grid_net_kw']).clip(lower=0).sum()* HOURS_PER_STEP:.1f} kWh")
    print()
    print(
        f"Offtake price range: "
        f"{df['offtake_price_eur_per_mwh'].min():.1f} to "
        f"{df['offtake_price_eur_per_mwh'].max():.1f} €/MWh"
    )


def plot_example_week(df: pd.DataFrame) -> None:
    start = df.index.min() + pd.Timedelta(days=30)
    week = df.loc[start : start + pd.Timedelta(days=7)]

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(week.index, week["load_kw"], label="Site load", linewidth=1.2)
    ax.plot(
        week.index,
        week["pv_production_kw"],
        label="PV production",
        linewidth=1.2,
    )
    ax.plot(
        week.index,
        week["grid_net_kw"],
        label="Grid net",
        linewidth=1.0,
    )

    ax.set_ylabel("kW")
    ax.set_title("Example week: load, PV and grid exchange")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_example_week.png", dpi=180)
    plt.close(fig)


def plot_daily_profile(df: pd.DataFrame) -> None:
    profile = (
        df.assign(time=df.index.strftime("%H:%M"))
        .groupby("time")[["load_kw", "pv_production_kw", "grid_net_kw"]]
        .mean()
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(profile.index, profile["load_kw"], label="Average load")
    ax.plot(profile.index, profile["pv_production_kw"], label="Average PV")
    ax.plot(profile.index, profile["grid_net_kw"], label="Average grid net")

    ax.set_ylabel("kW")
    ax.set_title("Average daily profile")
    ax.legend()

    ticks = range(0, len(profile), 8)
    ax.set_xticks(ticks)
    ax.set_xticklabels(profile.index[ticks])

    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_daily_profile.png", dpi=180)
    plt.close(fig)


def plot_prices(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.hist(
        df["offtake_price_eur_per_mwh"].dropna(),
        bins=60,
        alpha=0.7,
        label="Offtake",
    )
    ax.hist(
        df["injection_price_eur_per_mwh"].dropna(),
        bins=60,
        alpha=0.7,
        label="Injection",
    )

    ax.set_xlabel("€/MWh")
    ax.set_ylabel("Quarter hours")
    ax.set_title("Electricity price distribution")
    ax.legend()

    fig.tight_layout()
    fig.savefig(OUT_DIR / "03_price_distribution.png", dpi=180)
    plt.close(fig)


def plot_pv_forecast_relation(df: pd.DataFrame) -> None:
    sample = df[
        ["most_recent_load_factor_forecast", "pv_production_kw"]
    ].dropna()

    # Keep the figure light if the dataset is large
    if len(sample) > 5000:
        sample = sample.sample(5000, random_state=0)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(
        sample["most_recent_load_factor_forecast"],
        sample["pv_production_kw"],
        s=8,
        alpha=0.25,
    )

    ax.set_xlabel("Most recent PV load-factor forecast")
    ax.set_ylabel("Actual PV production [kW]")
    ax.set_title("PV forecast signal vs actual production")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "04_pv_forecast_signal.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data(DATA_PATH)

    print_summary(df)

    monthly_bills = calculate_monthly_bills(df)

    print("\nMonthly bill without battery:")
    print(
        monthly_bills[
            [
                "energy_offtake_cost_eur",
                "injection_revenue_eur",
                "peak_offtake_kw",
                "peak_cost_eur",
                "total_bill_eur",
            ]
        ].round(2)
    )

    plot_monthly_bill(monthly_bills)

    plot_example_week(df)
    plot_daily_profile(df)
    plot_prices(df)
    plot_pv_forecast_relation(df)

    print(f"\nFigures written to {OUT_DIR}")


if __name__ == "__main__":
    main()