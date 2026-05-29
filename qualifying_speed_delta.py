#!/usr/bin/env python3
"""Plot F1 qualifying speed delta by race using FastF1 session results."""

from __future__ import annotations

import argparse
import datetime as dt
import logging
from pathlib import Path

import fastf1
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patheffects


SESSION_CODE = "Q"
TIME_COLUMNS = ["Q1", "Q2", "Q3"]
TEAM_COLOR_FALLBACKS = {
    "Alpine": "#00A1E8",
    "Alpine F1 Team": "#00A1E8",
    "Aston Martin": "#006F62",
    "Ferrari": "#E80020",
    "Haas F1 Team": "#B6BABD",
    "Kick Sauber": "#52E252",
    "McLaren": "#FF8000",
    "Mercedes": "#27F4D2",
    "RB": "#6692FF",
    "RB F1 Team": "#6692FF",
    "Red Bull": "#3671C6",
    "Racing Bulls": "#6692FF",
    "Williams": "#64C4FF",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a qualifying development graph from FastF1. The default "
            "metric is each team's speed deficit versus the fastest team in "
            "that qualifying session."
        )
    )
    parser.add_argument("--season", type=int, default=dt.date.today().year)
    parser.add_argument("--start-round", type=int, help="First round to include.")
    parser.add_argument("--end-round", type=int, help="Last round to include.")
    parser.add_argument(
        "--teams",
        help="Comma-separated team names to plot. Defaults to every team found.",
    )
    parser.add_argument(
        "--benchmark-team",
        help=(
            "Compare against one team instead of the fastest team each race. "
            "Use FastF1 team names, for example 'Red Bull' or 'McLaren'."
        ),
    )
    parser.add_argument(
        "--metric",
        choices=["speed-deficit", "speed-delta", "time-delta"],
        default="speed-deficit",
        help=(
            "speed-deficit is positive percent slower than the benchmark; "
            "speed-delta is percent speed advantage/deficit, where slower cars "
            "are negative; time-delta is seconds slower than the benchmark."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="PNG output path. Defaults to outputs/quali_speed_delta_<season>.png.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="CSV output path. Defaults to outputs/quali_speed_delta_<season>.csv.",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("fastf1_cache"),
        help="FastF1 cache directory.",
    )
    parser.add_argument(
        "--include-future",
        action="store_true",
        help="Try every scheduled race, including future events.",
    )
    return parser.parse_args()


def best_driver_time_seconds(results: pd.DataFrame) -> pd.Series:
    times = results[TIME_COLUMNS].apply(pd.to_timedelta, errors="coerce")
    return times.min(axis=1).dt.total_seconds()


def short_race_name(event_name: str) -> str:
    return (
        event_name.replace(" Grand Prix", "")
        .replace("Grand Prix", "")
        .replace("Emilia Romagna", "Imola")
        .strip()
    )


def collect_qualifying_data(args: argparse.Namespace) -> pd.DataFrame:
    args.cache.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(args.cache))
    fastf1.set_log_level(logging.WARNING)

    schedule = fastf1.get_event_schedule(args.season, include_testing=False)
    schedule = schedule[schedule["RoundNumber"] > 0].copy()
    if args.start_round is not None:
        schedule = schedule[schedule["RoundNumber"] >= args.start_round]
    if args.end_round is not None:
        schedule = schedule[schedule["RoundNumber"] <= args.end_round]
    if not args.include_future:
        today = pd.Timestamp(dt.date.today())
        schedule = schedule[pd.to_datetime(schedule["EventDate"]) <= today]

    rows: list[dict[str, object]] = []
    for _, event in schedule.sort_values("RoundNumber").iterrows():
        round_number = int(event["RoundNumber"])
        event_name = str(event["EventName"])

        try:
            session = fastf1.get_session(args.season, round_number, SESSION_CODE)
            session.load(laps=False, telemetry=False, weather=False, messages=False)
        except Exception as exc:
            print(f"Skipping R{round_number} {event_name}: {exc}")
            continue

        results = session.results.copy()
        if results.empty or not set(TIME_COLUMNS).issubset(results.columns):
            print(f"Skipping R{round_number} {event_name}: no qualifying times")
            continue

        results["BestSeconds"] = best_driver_time_seconds(results)
        results = results.dropna(subset=["BestSeconds", "TeamName"])
        if results.empty:
            print(f"Skipping R{round_number} {event_name}: no usable qualifying laps")
            continue

        colors = (
            results.dropna(subset=["TeamName"])
            .drop_duplicates("TeamName")
            .set_index("TeamName")["TeamColor"]
            .to_dict()
        )

        team_best = (
            results.groupby("TeamName", as_index=False)["BestSeconds"]
            .min()
            .sort_values("BestSeconds")
        )
        fastest_seconds = float(team_best["BestSeconds"].min())
        if args.benchmark_team:
            benchmark = team_best.loc[
                team_best["TeamName"].str.casefold() == args.benchmark_team.casefold(),
                "BestSeconds",
            ]
            if benchmark.empty:
                print(
                    f"Skipping R{round_number} {event_name}: "
                    f"benchmark team '{args.benchmark_team}' not found"
                )
                continue
            benchmark_seconds = float(benchmark.iloc[0])
            benchmark_label = args.benchmark_team
        else:
            benchmark_seconds = fastest_seconds
            benchmark_label = str(team_best.iloc[0]["TeamName"])

        for _, team_row in team_best.iterrows():
            team_seconds = float(team_row["BestSeconds"])
            time_delta = team_seconds - benchmark_seconds
            speed_delta_pct = (benchmark_seconds / team_seconds - 1.0) * 100.0
            speed_deficit_pct = -speed_delta_pct
            if abs(time_delta) < 1e-12:
                time_delta = 0.0
            if abs(speed_delta_pct) < 1e-12:
                speed_delta_pct = 0.0
            if abs(speed_deficit_pct) < 1e-12:
                speed_deficit_pct = 0.0
            rows.append(
                {
                    "Season": args.season,
                    "Round": round_number,
                    "Race": short_race_name(event_name),
                    "EventName": event_name,
                    "Team": team_row["TeamName"],
                    "TeamBestSeconds": team_seconds,
                    "BenchmarkTeam": benchmark_label,
                    "BenchmarkSeconds": benchmark_seconds,
                    "TimeDeltaSeconds": time_delta,
                    "SpeedDeltaPct": speed_delta_pct,
                    "SpeedDeficitPct": speed_deficit_pct,
                    "TeamColor": normalize_color(colors.get(team_row["TeamName"])),
                }
            )

    if not rows:
        raise RuntimeError("No qualifying data could be loaded for the requested season.")

    data = pd.DataFrame(rows)
    if args.teams:
        wanted = {team.strip().casefold() for team in args.teams.split(",") if team.strip()}
        data = data[data["Team"].str.casefold().isin(wanted)]
        if data.empty:
            raise RuntimeError("None of the requested teams were found in the data.")

    return data.sort_values(["Round", "Team"]).reset_index(drop=True)


def metric_column_and_label(metric: str) -> tuple[str, str, str]:
    if metric == "time-delta":
        return "TimeDeltaSeconds", "Qualifying lap-time delta to benchmark (s)", "s"
    if metric == "speed-delta":
        return "SpeedDeltaPct", "Qualifying speed delta to benchmark (%)", "%"
    return "SpeedDeficitPct", "Qualifying speed deficit to benchmark (%)", "%"


def normalize_color(color: object) -> str | None:
    if pd.isna(color):
        return None
    text = str(color).strip()
    if not text:
        return None
    return text if text.startswith("#") else f"#{text}"


def team_color(team: str, data: pd.DataFrame, used: set[str]) -> str:
    color_values = data.loc[data["Team"] == team, "TeamColor"].dropna()
    color = str(color_values.iloc[0]) if not color_values.empty else None
    color = color or TEAM_COLOR_FALLBACKS.get(team)
    if color and color.casefold() not in used:
        used.add(color.casefold())
        return color

    cmap = plt.get_cmap("tab20")
    for idx in range(cmap.N):
        fallback = "#{:02x}{:02x}{:02x}".format(
            *(int(channel * 255) for channel in cmap(idx)[:3])
        )
        if fallback.casefold() not in used:
            used.add(fallback.casefold())
            return fallback
    return "#FFFFFF"


def nice_metric_name(metric: str) -> str:
    if metric == "time-delta":
        return "Lap-Time Deficit"
    if metric == "speed-delta":
        return "Speed Delta"
    return "Speed Deficit"


def plot_delta(data: pd.DataFrame, args: argparse.Namespace) -> None:
    output = args.output or Path("outputs") / f"quali_speed_delta_{args.season}.png"
    csv_path = args.csv or Path("outputs") / f"quali_speed_delta_{args.season}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    metric_col, ylabel, unit = metric_column_and_label(args.metric)
    pivot = data.pivot(index=["Round", "Race"], columns="Team", values=metric_col)

    plt.style.use("dark_background")
    fig_width = max(19.2, len(pivot) * 0.72)
    fig, ax = plt.subplots(figsize=(fig_width, 10.8), constrained_layout=False)
    fig.patch.set_facecolor("#07090D")
    ax.set_facecolor("#0D1118")
    fig.subplots_adjust(left=0.095, right=0.88, top=0.855, bottom=0.205)

    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(
        gradient,
        extent=[0, 1, 0, 1],
        transform=ax.transAxes,
        cmap=plt.get_cmap("bone_r"),
        alpha=0.08,
        aspect="auto",
        zorder=0,
    )

    teams = list(pivot.columns)
    used_colors: set[str] = set()
    line_colors = {team: team_color(team, data, used_colors) for team in teams}
    last_values = pivot.ffill().iloc[-1].sort_values()
    for idx, team in enumerate(teams):
        values = pivot[team]
        is_front = team in set(last_values.head(4).index)
        ax.plot(
            np.arange(len(pivot)),
            values,
            marker="o",
            linewidth=3.4 if is_front else 2.25,
            markersize=5.5 if is_front else 4.4,
            label=team,
            color=line_colors[team],
            alpha=0.98 if is_front else 0.78,
            solid_capstyle="round",
            path_effects=[
                patheffects.Stroke(linewidth=5.8 if is_front else 4.1, foreground="#05070B"),
                patheffects.Normal(),
            ],
            zorder=4 if is_front else 3,
        )

    if args.metric == "speed-delta":
        ax.axhline(0, color="#F4F6FA", linewidth=1.35, alpha=0.82, zorder=2)
    else:
        ax.axhline(0, color="#F4F6FA", linewidth=1.35, alpha=0.82, zorder=2)
        ax.text(
            0.985,
            0.03,
            "BENCHMARK",
            transform=ax.transAxes,
            color="#F4F6FA",
            fontsize=9,
            fontweight="bold",
            ha="right",
            va="bottom",
            alpha=0.82,
        )

    race_labels = [race for _, race in pivot.index]
    ax.set_xticks(np.arange(len(pivot)))
    ax.set_xticklabels(race_labels, rotation=35, ha="right", fontsize=10, color="#D7DEE8")
    ax.tick_params(axis="y", colors="#D7DEE8", labelsize=11)
    ax.tick_params(axis="x", colors="#D7DEE8", length=0)
    ax.set_ylabel(ylabel.upper(), fontsize=11, fontweight="bold", color="#D7DEE8", labelpad=12)
    ax.set_xlabel("")

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(True, axis="y", color="#FFFFFF", alpha=0.13, linewidth=1)
    ax.grid(True, axis="x", color="#FFFFFF", alpha=0.045, linewidth=1)
    ax.margins(x=0.04, y=0.14)
    ax.set_xlim(-0.55, len(pivot) + 0.9)

    metric_name = nice_metric_name(args.metric).upper()
    benchmark = (
        args.benchmark_team.upper()
        if args.benchmark_team
        else "FASTEST TEAM EACH QUALIFYING"
    )
    fig.text(
        0.035,
        0.94,
        f"{args.season} F1 QUALIFYING DEVELOPMENT",
        color="#FFFFFF",
        fontsize=28,
        fontweight="heavy",
        ha="left",
    )
    fig.text(
        0.036,
        0.902,
        f"{metric_name} BY RACE  |  BENCHMARK: {benchmark}",
        color="#FFEB3B",
        fontsize=12,
        fontweight="bold",
        ha="left",
    )
    fig.text(
        0.965,
        0.94,
        "FASTF1",
        color="#FFFFFF",
        fontsize=22,
        fontweight="heavy",
        ha="right",
        alpha=0.9,
    )

    label_x = len(pivot) - 1
    y_min = float(pivot.min(numeric_only=True).min())
    y_max = float(pivot.max(numeric_only=True).max())
    y_span = y_max - y_min
    label_pad = y_span * 0.035 if y_span else 0.08
    sorted_last = pivot.ffill().iloc[-1].sort_values()
    occupied: list[float] = []
    for team, value in sorted_last.items():
        label_y = float(value)
        for existing in occupied:
            if abs(label_y - existing) < label_pad:
                label_y = existing + label_pad
        occupied.append(label_y)
        ax.text(
            label_x + 0.18,
            label_y,
            team.upper(),
            color=line_colors[team],
            fontsize=9.5,
            fontweight="bold",
            ha="left",
            va="center",
            path_effects=[patheffects.Stroke(linewidth=3.5, foreground="#05070B"), patheffects.Normal()],
            clip_on=False,
        )
    if occupied:
        ax.set_ylim(min(y_min - y_span * 0.08, min(occupied) - label_pad), max(y_max + y_span * 0.08, max(occupied) + label_pad))

    if unit == "%":
        ax.yaxis.set_major_formatter(lambda value, _: f"{value:.2f}%")
    else:
        ax.yaxis.set_major_formatter(lambda value, _: f"{value:.2f}s")

    fig.text(
        0.035,
        0.035,
        "Best team qualifying result from Q1/Q2/Q3. Lower deficit indicates stronger one-lap pace.",
        color="#8D98A8",
        fontsize=9.5,
        ha="left",
    )

    data.to_csv(csv_path, index=False)
    fig.savefig(output, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print(f"Wrote {output}")
    print(f"Wrote {csv_path}")


def main() -> None:
    args = parse_args()
    data = collect_qualifying_data(args)
    plot_delta(data, args)


if __name__ == "__main__":
    main()
