from __future__ import annotations

import base64
import datetime as dt
import json
from io import BytesIO
import logging
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

import fastf1
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
from fastf1.core import Laps
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from timple.timedelta import strftimedelta
from PIL import Image


CACHE_DIR = Path("fastf1_cache")
KMH_TO_MPH = 0.621371
TIME_COLUMNS = ["Q1", "Q2", "Q3"]
TEAM_LOGO_DIR = Path("assets/team_logos")
TEAM_WIKI_TITLES = {
    "alpine": "Alpine F1 Team",
    "aston martin": "Aston Martin F1 Team",
    "cadillac": "Cadillac Formula 1 Team",
    "ferrari": "Scuderia Ferrari",
    "haas": "Haas F1 Team",
    "kick sauber": "Sauber Motorsport",
    "mclaren": "McLaren",
    "mercedes": "Mercedes-AMG Petronas Formula One Team",
    "racing bulls": "Racing Bulls",
    "rb": "Racing Bulls",
    "red bull": "Red Bull Racing",
    "sauber": "Sauber Motorsport",
    "williams": "Williams Racing",
}
SESSION_OPTIONS = {
    "Practice 1": "FP1",
    "Practice 2": "FP2",
    "Practice 3": "FP3",
    "Sprint Shootout / Sprint Qualifying": "SQ",
    "Sprint": "S",
    "Qualifying": "Q",
    "Race": "R",
}
NASCAR_SERIES = {
    "Cup Series": 1,
    "Xfinity Series": 2,
    "Truck Series": 3,
}
NASCAR_SERIES_SLUGS = {
    1: "nascar-cup-series",
    2: "nascar-xfinity-series",
    3: "nascar-craftsman-truck-series",
}
INDYCAR_SERIES_ID = "b856a4f1-e85c-4fac-8c36-fd58d962227a"
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
    "Red Bull Racing": "#3671C6",
    "Racing Bulls": "#6692FF",
    "Williams": "#64C4FF",
    "Williams Racing": "#64C4FF",
}


@dataclass(frozen=True)
class TelemetryLap:
    driver: str
    team: str
    lap_time: float
    frame: pd.DataFrame


st.set_page_config(page_title="F1 Session Intelligence", page_icon="F1", layout="wide")


def configure_fastf1() -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    fastf1.set_log_level(logging.WARNING)


@st.cache_data(show_spinner=False, ttl=60 * 60 * 12)
def load_schedule(year: int) -> pd.DataFrame:
    configure_fastf1()
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    schedule = schedule[schedule["RoundNumber"] > 0].copy()
    schedule["Label"] = schedule["RoundNumber"].astype(int).astype(str) + " - " + schedule["EventName"]
    return schedule


@st.cache_resource(show_spinner=False, ttl=60 * 60 * 12)
def load_session(year: int, round_number: int, session_code: str):
    configure_fastf1()
    session = fastf1.get_session(year, round_number, session_code)
    session.load(laps=True, telemetry=True, weather=False, messages=False)
    return session


def seconds(value) -> float:
    if pd.isna(value):
        return np.nan
    return pd.to_timedelta(value).total_seconds()


def format_time(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    minutes, remainder = divmod(float(value), 60)
    return f"{int(minutes)}:{remainder:06.3f}"


def format_speed_mph(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value) * KMH_TO_MPH:.1f} mph"


def normalize_color(value: object, team: str | None = None) -> str:
    if value is not None and not pd.isna(value):
        text = str(value).strip()
        if text:
            return text if text.startswith("#") else f"#{text}"
    return TEAM_COLOR_FALLBACKS.get(str(team), "#8A94A6")


def display_frame(data: pd.DataFrame, height: int = 340) -> None:
    st.dataframe(data, hide_index=True, width="stretch", height=height)


def valid_laps(session) -> pd.DataFrame:
    laps = session.laps.copy()
    laps["LapTimeSeconds"] = laps["LapTime"].map(seconds)
    laps = laps.dropna(subset=["LapTimeSeconds", "Driver", "Team"])
    if "Deleted" in laps.columns:
        deleted = laps["Deleted"].astype("boolean").fillna(False)
        laps = laps[~deleted]
    return laps


@st.cache_resource(show_spinner=False, ttl=60 * 60 * 12)
def load_laps_only_session(year: int, round_number: int, session_code: str):
    configure_fastf1()
    session = fastf1.get_session(year, round_number, session_code)
    session.load(laps=True, telemetry=False, weather=False, messages=False)
    return session


@st.cache_resource(show_spinner=False, ttl=60 * 60 * 12)
def load_basic_session(year: int, round_number: int, session_code: str):
    configure_fastf1()
    session = fastf1.get_session(year, round_number, session_code)
    session.load(laps=True, telemetry=False, weather=False, messages=False)
    return session


@st.cache_resource(show_spinner=False, ttl=60 * 60 * 12)
def load_results_only_session(year: int, round_number: int, session_code: str):
    configure_fastf1()
    session = fastf1.get_session(year, round_number, session_code)
    session.load(laps=False, telemetry=False, weather=False, messages=False)
    return session


def load_json_url(url: str):
    request = Request(url, headers={"User-Agent": "F1SessionIntelligence/1.0"})
    with urlopen(request, timeout=15) as response:
        return json.loads(response.read().decode("utf-8"))


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def load_openf1_json(path: str, query: str) -> list[dict]:
    return load_json_url(f"https://api.openf1.org/v1/{path}?{query}")


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def load_nascar_schedule(year: int, series_id: int) -> pd.DataFrame:
    payload = load_json_url(f"https://www.nascar.com/json/schedule/?season={year}&series={series_id}")
    rows = [row for row in payload.get("response", []) if str(row.get("Series_Id")) == str(series_id)]
    data = pd.DataFrame(rows)
    if data.empty:
        return data
    data["Race_Date_Value"] = pd.to_datetime(data["Race_Date"], errors="coerce", utc=True)
    data = data.sort_values("Race_Date_Value").reset_index(drop=True)
    data["Label"] = (
        data["Race_Date_Plain"].astype(str)
        + " - "
        + data["Race_Name"].astype(str)
        + " ("
        + data["Track_Name"].astype(str)
        + ")"
    )
    return data


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def load_nascar_cached_file(year: int, series_id: int, race_id: int, filename: str):
    url = f"https://cf.nascar.com/cacher/{year}/{series_id}/{race_id}/{filename}"
    return load_json_url(url)


def nascar_driver_badge_source(series_id: int, number: object) -> str | None:
    if pd.isna(number):
        return None
    return f"https://cf.nascar.com/data/images/carbadges/{series_id}/{str(number).strip()}.png"


def nascar_results_dataframe(results: list[dict], series_id: int) -> pd.DataFrame:
    rows = []
    for row in results or []:
        driver = row.get("DriverNameTag") or " ".join(
            part for part in [row.get("DriverFirstName"), row.get("DriverLastName")] if part
        )
        rows.append(
            {
                "Driver": driver,
                "Number": row.get("Number"),
                "Manufacturer": row.get("Manufacturer"),
                "Team": row.get("TeamOwner") or row.get("Manufacturer") or "",
                "Finish": numeric_value(row.get("RunningPos"), np.nan),
                "Start": numeric_value(row.get("StartPos"), np.nan),
                "Position Gain": numeric_value(row.get("StartPos"), np.nan) - numeric_value(row.get("RunningPos"), np.nan),
                "Best Lap": numeric_value(row.get("BestLapTime"), np.nan),
                "Best Lap Speed": numeric_value(row.get("BestLapSpeed"), np.nan),
                "Average Speed": numeric_value(row.get("AvgSpeed"), np.nan),
                "Average Running Position": numeric_value(row.get("AvgRunningPos"), np.nan),
                "Laps Top 10": numeric_value(row.get("LapsInTop10"), 0),
                "Laps Led": numeric_value(row.get("LapsLed"), 0),
                "Total Pit Stops": numeric_value(row.get("TotalPitStops"), 0),
                "Best 4 Tire Stop": numeric_value(row.get("Best4TireStopTime"), np.nan),
                "Last Stop Time": numeric_value(row.get("LastStopTime"), np.nan),
                "Pit Stops": row.get("PitStops") or [],
                "Badge": nascar_driver_badge_source(series_id, row.get("Number")),
            }
        )
    return pd.DataFrame(rows)


def nascar_qualifying_dataframe(weekend: dict, series_id: int) -> pd.DataFrame:
    rows = []
    for run in weekend.get("weekend_runs") or []:
        if int(run.get("run_type") or 0) != 2:
            continue
        for row in run.get("results") or []:
            rows.append(
                {
                    "Driver": row.get("driver_name"),
                    "Number": row.get("vehicle_number") or row.get("car_number"),
                    "Manufacturer": row.get("manufacturer"),
                    "Position": numeric_value(row.get("finishing_position"), np.nan),
                    "Best Lap": numeric_value(row.get("best_lap_time"), np.nan),
                    "Best Lap Speed": numeric_value(row.get("best_lap_speed"), np.nan),
                    "Delta": abs(numeric_value(row.get("delta_leader"), np.nan)),
                    "Badge": nascar_driver_badge_source(series_id, row.get("vehicle_number") or row.get("car_number")),
                }
            )
    return pd.DataFrame(rows)


def nascar_pitstop_dataframe(pit_data: list[dict], series_id: int) -> pd.DataFrame:
    rows = []
    for stop in pit_data or []:
        box = numeric_value(stop.get("pit_stop_duration"), np.nan)
        box_start = numeric_value(stop.get("box_stop_race_time"), np.nan)
        box_end = numeric_value(stop.get("box_leave_race_time"), np.nan)
        stop_type = str(stop.get("pit_stop_type") or "").strip()
        if pd.notna(box) and 0 < box < 60 and box_start >= 0 and box_end >= 0 and stop_type != "OTHER":
            number = stop.get("vehicle_number")
            rows.append(
                {
                    "Driver": stop.get("driver_name"),
                    "Number": number,
                    "Manufacturer": stop.get("vehicle_manufacturer"),
                    "Team": stop.get("vehicle_manufacturer"),
                    "Lap": stop.get("lap_count") or stop.get("leader_lap"),
                    "Leader Lap": stop.get("leader_lap"),
                    "Stop Type": stop_type.replace("_", " ").title(),
                    "Box Time": box,
                    "Total Pit Road Time": numeric_value(stop.get("total_duration"), np.nan),
                    "In Travel Time": numeric_value(stop.get("in_travel_duration"), np.nan),
                    "Out Travel Time": numeric_value(stop.get("out_travel_duration"), np.nan),
                    "Pit In Rank": numeric_value(stop.get("pit_in_rank"), np.nan),
                    "Pit Out Rank": numeric_value(stop.get("pit_out_rank"), np.nan),
                    "Positions Gained/Lost": numeric_value(stop.get("positions_gained_lost"), np.nan),
                    "Badge": nascar_driver_badge_source(series_id, number),
                }
            )
    return pd.DataFrame(rows)


def indycar_time_seconds(value: object) -> float:
    if value is None or pd.isna(value):
        return np.nan
    text = str(value).strip()
    if not text or text in {"--.----", "--"}:
        return np.nan
    try:
        parts = text.split(":")
        if len(parts) == 1:
            return float(parts[0])
        seconds_part = float(parts[-1])
        minutes = int(parts[-2]) if len(parts) >= 2 else 0
        hours = int(parts[-3]) if len(parts) >= 3 else 0
        return hours * 3600 + minutes * 60 + seconds_part
    except Exception:
        return np.nan


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def load_indycar_seasons() -> list[int]:
    return load_json_url(f"https://www.indycar.com/api/results/YearsBySeries?series={INDYCAR_SERIES_ID}")


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def load_indycar_season_dropdown() -> list[dict]:
    return load_json_url(f"https://www.indycar.com/api/results/SeasonDropDown?id={INDYCAR_SERIES_ID}")


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def load_indycar_session(session_id: str | int) -> dict:
    return load_json_url(f"https://www.indycar.com/api/results/EventsSessionDetails?id={session_id}")


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def load_indycar_standings(year: int) -> dict:
    return load_json_url(f"https://www.indycar.com/api/results/YearPointSummary?year={year}&id={INDYCAR_SERIES_ID}")


def indycar_records_dataframe(session: dict) -> pd.DataFrame:
    rows = []
    for row in session.get("records") or []:
        rows.append(
            {
                "Driver": row.get("DriverName"),
                "Number": row.get("CarNumber"),
                "Team": row.get("TeamName"),
                "Finish": numeric_value(row.get("PositionFinish"), np.nan),
                "Start": numeric_value(row.get("PositionStart"), np.nan),
                "Position Gain": numeric_value(row.get("PositionStart"), np.nan) - numeric_value(row.get("PositionFinish"), np.nan),
                "Laps": numeric_value(row.get("LapsComplete"), np.nan),
                "Laps Led": numeric_value(row.get("LapsLed"), 0),
                "Pit Stops": numeric_value(row.get("PitStops"), np.nan),
                "Best Lap": indycar_time_seconds(row.get("BestLapTime")),
                "Best Lap Text": row.get("BestLapTime"),
                "Best Speed": numeric_value(row.get("BestSpeed"), np.nan),
                "Average Speed": numeric_value(row.get("SpeedAvg"), np.nan),
                "Status": row.get("Status"),
                "Points": numeric_value(row.get("PointsEarned"), np.nan),
                "Badge": None,
            }
        )
    return pd.DataFrame(rows)


def indycar_standings_dataframe(payload: dict) -> pd.DataFrame:
    rows = []
    for row in payload.get("DriverList") or []:
        rows.append(
            {
                "Pos": numeric_value(row.get("OverallPosition"), np.nan),
                "Driver": row.get("DriverName"),
                "Points": numeric_value(row.get("TotalPoints"), np.nan),
                "Wins": numeric_value(row.get("TotalWins"), 0),
                "Poles": numeric_value(row.get("TotalPoles"), 0),
                "Top 5s": numeric_value(row.get("TotalTop5s"), 0),
                "Best Finish": numeric_value(row.get("BestFinish"), np.nan),
                "Road Points": numeric_value(row.get("RoadPoints"), np.nan),
                "Oval Points": numeric_value(row.get("OvalPoints"), np.nan),
                "Team": "",
                "Number": "",
                "Badge": None,
            }
        )
    return pd.DataFrame(rows)


def fastest_lap_per_driver(laps: pd.DataFrame) -> pd.DataFrame:
    idx = laps.groupby("Driver")["LapTimeSeconds"].idxmin()
    return laps.loc[idx].sort_values("LapTimeSeconds").reset_index(drop=True)


def get_lap_telemetry(lap) -> pd.DataFrame:
    tel = lap.get_telemetry()
    if tel.empty:
        return tel
    if "Distance" not in tel.columns:
        tel = tel.add_distance()
    keep = [column for column in ["Distance", "Speed", "Throttle", "X", "Y", "Time"] if column in tel.columns]
    tel = tel[keep].dropna(subset=["Distance", "Speed"]).sort_values("Distance")
    return tel


@st.cache_data(show_spinner=False, ttl=60 * 60 * 12)
def collect_fastest_lap_telemetry(year: int, round_number: int, session_code: str) -> list[TelemetryLap]:
    session = load_session(year, round_number, session_code)
    laps = fastest_lap_per_driver(valid_laps(session))
    telemetry: list[TelemetryLap] = []
    for _, lap in laps.iterlaps():
        tel = get_lap_telemetry(lap)
        if tel.empty or len(tel) < 20:
            continue
        telemetry.append(
            TelemetryLap(
                driver=str(lap["Driver"]),
                team=str(lap["Team"]),
                lap_time=float(lap["LapTimeSeconds"]),
                frame=tel,
            )
        )
    return telemetry


def circuit_corners(session, track_length: float) -> pd.DataFrame:
    try:
        corners = session.get_circuit_info().corners.copy()
    except Exception:
        corners = pd.DataFrame()

    if not corners.empty and "Distance" in corners.columns:
        corners = corners.dropna(subset=["Distance"]).copy()
        corners["Label"] = corners["Number"].astype(str) + corners.get("Letter", "").fillna("").astype(str)
        return corners[["Label", "Distance"]].sort_values("Distance").reset_index(drop=True)

    # Fallback: evenly spaced named zones keeps the interface useful on sessions
    # where FastF1 has no circuit metadata yet.
    distances = np.linspace(track_length * 0.08, track_length * 0.92, 12)
    return pd.DataFrame({"Label": [f"Zone {idx}" for idx in range(1, 13)], "Distance": distances})


def corner_mask(distances: np.ndarray, corners: pd.DataFrame, track_length: float, radius: float) -> np.ndarray:
    mask = np.zeros(len(distances), dtype=bool)
    for center in corners["Distance"].to_numpy(dtype=float):
        direct = np.abs(distances - center)
        wrapped = track_length - direct
        mask |= np.minimum(direct, wrapped) <= radius
    return mask


def mean_speed_in_mask(tel: pd.DataFrame, mask: np.ndarray) -> float:
    if not mask.any():
        return np.nan
    return float(tel.loc[mask, "Speed"].mean())


def classify_corner_speed(speed: float) -> str:
    if speed < 140:
        return "Slow Speed"
    if speed < 210:
        return "Medium Speed"
    return "High Speed"


def corner_speed_samples(
    telemetry: list[TelemetryLap], corners: pd.DataFrame, track_length: float, radius: float
) -> pd.DataFrame:
    rows = []
    for item in telemetry:
        distances = item.frame["Distance"].to_numpy(dtype=float)
        for _, corner in corners.iterrows():
            single_corner = pd.DataFrame([corner])
            mask = corner_mask(distances, single_corner, track_length, radius)
            if mask.any():
                rows.append(
                    {
                        "Corner": corner["Label"],
                        "Driver": item.driver,
                        "Team": item.team,
                        "Average Speed": float(item.frame.loc[mask, "Speed"].mean()),
                        "Minimum Speed": float(item.frame.loc[mask, "Speed"].min()),
                    }
                )
    data = pd.DataFrame(rows)
    if data.empty:
        return data

    corner_classes = (
        data.groupby("Corner", as_index=False)["Minimum Speed"]
        .median()
        .rename(columns={"Minimum Speed": "Median Minimum Speed"})
    )
    corner_classes["Corner Type"] = corner_classes["Median Minimum Speed"].map(classify_corner_speed)
    return data.merge(corner_classes, on="Corner", how="left")


def corner_type_rankings(corner_samples: pd.DataFrame, corner_type: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = corner_samples[corner_samples["Corner Type"] == corner_type].copy()
    if data.empty:
        return pd.DataFrame(), pd.DataFrame()

    driver = (
        data.groupby(["Driver", "Team"], as_index=False)
        .agg(
            **{
                "Average Speed": ("Average Speed", "mean"),
                "Minimum Speed": ("Minimum Speed", "mean"),
            }
        )
        .sort_values("Average Speed", ascending=False)
        .reset_index(drop=True)
    )
    team = (
        driver.groupby("Team", as_index=False)
        .agg(
            **{
                "Average Speed": ("Average Speed", "max"),
                "Minimum Speed": ("Minimum Speed", "max"),
            }
        )
        .sort_values("Average Speed", ascending=False)
        .reset_index(drop=True)
    )
    return driver, team


def format_speed_columns(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    for column in ["Average Speed", "Minimum Speed", "Median Minimum Speed"]:
        if column in result.columns:
            result[column] = result[column].map(format_speed_mph)
    return result


def corner_ranking_graphic(
    data: pd.DataFrame,
    corner_type: str,
    mode: str,
    event_name: str,
    session_name: str,
    headshots: dict[str, str],
    corner_list: pd.DataFrame,
    top_n: int = 8,
):
    if data.empty:
        return None

    name_col = "Driver" if mode == "Drivers" else "Team"
    plot = data.sort_values("Average Speed", ascending=False).head(top_n).copy().reset_index(drop=True)
    fastest = float(plot["Average Speed"].max())
    slowest = float(plot["Average Speed"].min())
    spread = max(fastest - slowest, 1.0)
    baseline = max(slowest - spread * 0.5, 0.0)
    plot["Visual Speed"] = plot["Average Speed"] - baseline

    fig_height = max(5.6, 1.0 + len(plot) * 0.72)
    fig, ax = plt.subplots(figsize=(13.5, fig_height), facecolor="#07090D")
    ax.set_facecolor("#0D1118")

    y_values = np.arange(len(plot))
    colors = [normalize_color(None, team) for team in plot["Team"]]
    ax.barh(
        y_values,
        plot["Visual Speed"],
        left=baseline,
        color=colors,
        edgecolor="#F8FAFC",
        linewidth=1.1,
        height=0.54,
        alpha=0.96,
    )
    ax.invert_yaxis()

    corner_names = ", ".join(corner_list["Corner"].astype(str).tolist())
    if len(corner_names) > 80:
        corner_names = corner_names[:77].rstrip() + "..."
    ax.set_title(
        f"{corner_type.upper()} CORNER PACE",
        color="#FFFFFF",
        fontsize=24,
        fontweight="bold",
        loc="left",
        pad=28,
    )
    ax.text(
        1,
        1.075,
        f"{event_name} - {session_name}",
        transform=ax.transAxes,
        ha="right",
        va="center",
        color="#F8FAFC",
        fontsize=15,
        fontweight="bold",
    )
    ax.text(
        0,
        1.015,
        f"{mode} ranking | Corners: {corner_names}",
        transform=ax.transAxes,
        ha="left",
        va="center",
        color="#A7B0C0",
        fontsize=11,
    )

    x_min = baseline - spread * 0.42
    x_max = fastest + spread * 0.55
    image_x = baseline - spread * 0.22
    label_x = baseline - spread * 0.06
    value_x = fastest + spread * 0.12
    zoom = 0.22 if mode == "Drivers" else 0.18

    for idx, (_, row) in enumerate(plot.iterrows()):
        name = str(row[name_col])
        team = str(row["Team"])
        speed_mph = float(row["Average Speed"]) * KMH_TO_MPH
        min_mph = float(row["Minimum Speed"]) * KMH_TO_MPH
        delta_mph = (float(row["Average Speed"]) - fastest) * KMH_TO_MPH
        image = load_plot_image(headshots.get(name)) if mode == "Drivers" else load_plot_image(team_logo_source(team))

        if image is not None:
            ax.add_artist(AnnotationBbox(OffsetImage(image, zoom=zoom), (image_x, idx), frameon=False, zorder=10))
        else:
            draw_fallback_badge(ax, image_x, idx, name if mode == "Drivers" else team, normalize_color(None, team), zoom)

        ax.text(label_x, idx - 0.13, name, ha="left", va="center", color="#FFFFFF", fontsize=13, fontweight="bold")
        if mode == "Drivers":
            ax.text(label_x, idx + 0.17, team, ha="left", va="center", color="#A7B0C0", fontsize=10)

        delta_text = "Leader" if idx == 0 else f"{delta_mph:.1f} mph"
        ax.text(
            value_x,
            idx - 0.12,
            f"{speed_mph:.1f} mph",
            ha="left",
            va="center",
            color="#F8FAFC",
            fontsize=13,
            fontweight="bold",
        )
        ax.text(
            value_x,
            idx + 0.17,
            f"min {min_mph:.1f} mph | {delta_text}",
            ha="left",
            va="center",
            color="#A7B0C0",
            fontsize=9.5,
        )

        ax.text(
            x_min,
            idx,
            f"{idx + 1:02d}",
            ha="left",
            va="center",
            color="#F8FAFC" if idx < 3 else "#7E8796",
            fontsize=13,
            fontweight="bold",
        )

    ax.set_xlim(x_min, x_max)
    ax.set_yticks([])
    ax.set_xlabel("Average corner speed", color="#A7B0C0", fontsize=10, fontweight="bold")
    ax.tick_params(axis="x", colors="#7E8796", labelsize=9)
    ax.xaxis.grid(True, color="#273244", alpha=0.5, linestyle="--")
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return fig


def corner_leaders_graphic(
    data: pd.DataFrame,
    mode: str,
    event_name: str,
    session_name: str,
    headshots: dict[str, str],
    top_n: int = 10,
):
    if data.empty:
        return None

    name_col = "Driver" if mode == "Drivers" else "Team"
    plot = data.sort_values("Minimum Speed", ascending=False).head(top_n).copy().reset_index(drop=True)
    fastest = float(plot["Minimum Speed"].max())
    slowest = float(plot["Minimum Speed"].min())
    spread = max(fastest - slowest, 1.0)
    baseline = max(slowest - spread * 0.5, 0.0)
    plot["Visual Speed"] = plot["Minimum Speed"] - baseline

    fig_height = max(5.2, 1.0 + len(plot) * 0.62)
    fig, ax = plt.subplots(figsize=(13.5, fig_height), facecolor="#07090D")
    ax.set_facecolor("#0D1118")

    y_values = np.arange(len(plot))
    colors = [normalize_color(None, team) for team in plot["Team"]]
    ax.barh(
        y_values,
        plot["Visual Speed"],
        left=baseline,
        color=colors,
        edgecolor="#F8FAFC",
        linewidth=1.0,
        height=0.48,
        alpha=0.96,
    )
    ax.invert_yaxis()

    ax.set_title(
        "APEX SPEED LEADERS",
        color="#FFFFFF",
        fontsize=24,
        fontweight="bold",
        loc="left",
        pad=28,
    )
    ax.text(
        1,
        1.075,
        f"{event_name} - {session_name}",
        transform=ax.transAxes,
        ha="right",
        va="center",
        color="#F8FAFC",
        fontsize=15,
        fontweight="bold",
    )
    ax.text(
        0,
        1.015,
        f"Fastest minimum speed by corner | {mode}",
        transform=ax.transAxes,
        ha="left",
        va="center",
        color="#A7B0C0",
        fontsize=11,
    )

    x_min = baseline - spread * 0.44
    x_max = fastest + spread * 0.58
    image_x = baseline - spread * 0.24
    label_x = baseline - spread * 0.08
    value_x = fastest + spread * 0.12
    zoom = 0.2 if mode == "Drivers" else 0.16

    for idx, (_, row) in enumerate(plot.iterrows()):
        team = str(row["Team"])
        name = str(row[name_col])
        corner = str(row["Corner"])
        speed_mph = float(row["Minimum Speed"]) * KMH_TO_MPH
        image = load_plot_image(headshots.get(name)) if mode == "Drivers" else load_plot_image(team_logo_source(team))

        if image is not None:
            ax.add_artist(AnnotationBbox(OffsetImage(image, zoom=zoom), (image_x, idx), frameon=False, zorder=10))
        else:
            draw_fallback_badge(ax, image_x, idx, name, normalize_color(None, team), zoom)

        ax.text(label_x, idx - 0.12, f"Turn {corner}", ha="left", va="center", color="#FFFFFF", fontsize=12.5, fontweight="bold")
        detail = f"{name} | {team}" if mode == "Drivers" else name
        ax.text(label_x, idx + 0.16, detail, ha="left", va="center", color="#A7B0C0", fontsize=9.5)
        ax.text(value_x, idx, f"{speed_mph:.1f} mph", ha="left", va="center", color="#F8FAFC", fontsize=12.5, fontweight="bold")

    ax.set_xlim(x_min, x_max)
    ax.set_yticks([])
    ax.set_xlabel("Minimum speed", color="#A7B0C0", fontsize=10, fontweight="bold")
    ax.tick_params(axis="x", colors="#7E8796", labelsize=9)
    ax.xaxis.grid(True, color="#273244", alpha=0.5, linestyle="--")
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return fig


def straight_ranking_graphic(
    data: pd.DataFrame,
    mode: str,
    event_name: str,
    session_name: str,
    headshots: dict[str, str],
    top_n: int = 10,
):
    if data.empty:
        return None

    name_col = "Driver" if mode == "Drivers" else "Team"
    plot = data.sort_values("Straight Avg Speed", ascending=False).head(top_n).copy().reset_index(drop=True)
    fastest = float(plot["Straight Avg Speed"].max())
    slowest = float(plot["Straight Avg Speed"].min())
    spread = max(fastest - slowest, 1.0)
    baseline = max(slowest - spread * 0.5, 0.0)
    plot["Visual Speed"] = plot["Straight Avg Speed"] - baseline

    fig_height = max(5.6, 1.0 + len(plot) * 0.7)
    fig, ax = plt.subplots(figsize=(13.5, fig_height), facecolor="#07090D")
    ax.set_facecolor("#0D1118")

    y_values = np.arange(len(plot))
    colors = [normalize_color(None, team) for team in plot["Team"]]
    ax.barh(
        y_values,
        plot["Visual Speed"],
        left=baseline,
        color=colors,
        edgecolor="#F8FAFC",
        linewidth=1.1,
        height=0.54,
        alpha=0.96,
    )
    ax.invert_yaxis()

    ax.set_title(
        "STRAIGHT-LINE PACE",
        color="#FFFFFF",
        fontsize=24,
        fontweight="bold",
        loc="left",
        pad=28,
    )
    ax.text(
        1,
        1.075,
        f"{event_name} - {session_name}",
        transform=ax.transAxes,
        ha="right",
        va="center",
        color="#F8FAFC",
        fontsize=15,
        fontweight="bold",
    )
    ax.text(
        0,
        1.015,
        f"{mode} ranking | Average speed outside mapped corner windows",
        transform=ax.transAxes,
        ha="left",
        va="center",
        color="#A7B0C0",
        fontsize=11,
    )

    x_min = baseline - spread * 0.42
    x_max = fastest + spread * 0.62
    image_x = baseline - spread * 0.22
    label_x = baseline - spread * 0.06
    value_x = fastest + spread * 0.12
    zoom = 0.22 if mode == "Drivers" else 0.18

    for idx, (_, row) in enumerate(plot.iterrows()):
        name = str(row[name_col])
        team = str(row["Team"])
        avg_mph = float(row["Straight Avg Speed"]) * KMH_TO_MPH
        trap_mph = float(row["Speed Trap"]) * KMH_TO_MPH if "Speed Trap" in row and pd.notna(row["Speed Trap"]) else np.nan
        delta_mph = (float(row["Straight Avg Speed"]) - fastest) * KMH_TO_MPH
        image = load_plot_image(headshots.get(name)) if mode == "Drivers" else load_plot_image(team_logo_source(team))

        if image is not None:
            ax.add_artist(AnnotationBbox(OffsetImage(image, zoom=zoom), (image_x, idx), frameon=False, zorder=10))
        else:
            draw_fallback_badge(ax, image_x, idx, name if mode == "Drivers" else team, normalize_color(None, team), zoom)

        ax.text(label_x, idx - 0.13, name, ha="left", va="center", color="#FFFFFF", fontsize=13, fontweight="bold")
        if mode == "Drivers":
            best_lap = str(row.get("Best Lap", "-"))
            ax.text(label_x, idx + 0.17, f"{team} | best lap {best_lap}", ha="left", va="center", color="#A7B0C0", fontsize=10)
        else:
            ax.text(label_x, idx + 0.17, "best team straight package", ha="left", va="center", color="#A7B0C0", fontsize=10)

        delta_text = "Leader" if idx == 0 else f"{delta_mph:.1f} mph"
        trap_text = f"trap {trap_mph:.1f} mph" if pd.notna(trap_mph) else "trap -"
        ax.text(
            value_x,
            idx - 0.12,
            f"{avg_mph:.1f} mph",
            ha="left",
            va="center",
            color="#F8FAFC",
            fontsize=13,
            fontweight="bold",
        )
        ax.text(
            value_x,
            idx + 0.17,
            f"{trap_text} | {delta_text}",
            ha="left",
            va="center",
            color="#A7B0C0",
            fontsize=9.5,
        )
        ax.text(
            x_min,
            idx,
            f"{idx + 1:02d}",
            ha="left",
            va="center",
            color="#F8FAFC" if idx < 3 else "#7E8796",
            fontsize=13,
            fontweight="bold",
        )

    ax.set_xlim(x_min, x_max)
    ax.set_yticks([])
    ax.set_xlabel("Average straight speed", color="#A7B0C0", fontsize=10, fontweight="bold")
    ax.tick_params(axis="x", colors="#7E8796", labelsize=9)
    ax.xaxis.grid(True, color="#273244", alpha=0.5, linestyle="--")
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return fig


def team_efficiency_data(telemetry: list[TelemetryLap]) -> pd.DataFrame:
    rows = []
    for item in telemetry:
        rows.append(
            {
                "Driver": item.driver,
                "Team": item.team,
                "Lap Time": item.lap_time,
                "Mean Speed": float(item.frame["Speed"].mean()),
                "Top Speed": float(item.frame["Speed"].max()),
            }
        )
    data = pd.DataFrame(rows)
    if data.empty:
        return data
    idx = data.groupby("Team")["Lap Time"].idxmin()
    return data.loc[idx].sort_values("Mean Speed", ascending=False).reset_index(drop=True)


def team_efficiency_graphic(data: pd.DataFrame, event_name: str, session_name: str, year: int):
    if data.empty:
        return None

    plot = data.copy()
    x = plot["Mean Speed"].to_numpy(dtype=float)
    y = plot["Top Speed"].to_numpy(dtype=float)
    x_mid = float(np.median(x))
    y_mid = float(np.median(y))
    x_pad = max((float(x.max()) - float(x.min())) * 0.24, 1.2)
    y_pad = max((float(y.max()) - float(y.min())) * 0.24, 1.2)
    x_min, x_max = float(x.min()) - x_pad, float(x.max()) + x_pad
    y_min, y_max = float(y.min()) - y_pad, float(y.max()) + y_pad

    fig, ax = plt.subplots(figsize=(13.5, 9.2), facecolor="#F7F8FA")
    ax.set_facecolor("#FFFFFF")

    ax.axvline(x_mid, color="#1F2937", linewidth=1.4, alpha=0.72)
    ax.axhline(y_mid, color="#1F2937", linewidth=1.4, alpha=0.72)

    arrow_color = "#111827"
    arrow_kw = {
        "arrowstyle": "-|>,head_width=0.55,head_length=0.8",
        "color": arrow_color,
        "linewidth": 1.45,
        "alpha": 0.9,
        "shrinkA": 0,
        "shrinkB": 0,
    }
    ax.annotate("", xy=(x_max - x_pad * 0.35, y_mid), xytext=(x_mid, y_mid), arrowprops=arrow_kw)
    ax.annotate("", xy=(x_min + x_pad * 0.35, y_mid), xytext=(x_mid, y_mid), arrowprops=arrow_kw)
    ax.annotate("", xy=(x_mid, y_max - y_pad * 0.35), xytext=(x_mid, y_mid), arrowprops=arrow_kw)
    ax.annotate("", xy=(x_mid, y_min + y_pad * 0.35), xytext=(x_mid, y_mid), arrowprops=arrow_kw)
    ax.annotate("", xy=(x_max - x_pad * 0.45, y_max - y_pad * 0.45), xytext=(x_mid, y_mid), arrowprops=arrow_kw)
    ax.annotate("", xy=(x_min + x_pad * 0.45, y_max - y_pad * 0.45), xytext=(x_mid, y_mid), arrowprops=arrow_kw)
    ax.annotate("", xy=(x_max - x_pad * 0.45, y_min + y_pad * 0.45), xytext=(x_mid, y_mid), arrowprops=arrow_kw)
    ax.annotate("", xy=(x_min + x_pad * 0.45, y_min + y_pad * 0.45), xytext=(x_mid, y_mid), arrowprops=arrow_kw)

    ax.text(x_min + x_pad * 0.2, y_max - y_pad * 0.2, "Correlated with\nLow Downforce", ha="left", va="top", color="#111827", fontsize=13)
    ax.text(x_max - x_pad * 0.2, y_max - y_pad * 0.2, "Correlated with\nHigh Efficiency", ha="right", va="top", color="#111827", fontsize=13)
    ax.text(x_min + x_pad * 0.2, y_min + y_pad * 0.2, "Correlated with\nLow Efficiency", ha="left", va="bottom", color="#111827", fontsize=13)
    ax.text(x_max - x_pad * 0.2, y_min + y_pad * 0.2, "Correlated with\nHigh Downforce", ha="right", va="bottom", color="#111827", fontsize=13)
    ax.text(x_mid, y_max - y_pad * 0.28, "Low Drag", ha="center", va="top", color="#111827", fontsize=13)
    ax.text(x_mid, y_min + y_pad * 0.28, "High Drag", ha="center", va="bottom", color="#111827", fontsize=13)
    ax.text(x_max - x_pad * 0.48, y_mid + y_pad * 0.08, "Quick", ha="right", va="bottom", color="#111827", fontsize=13)
    ax.text(x_min + x_pad * 0.48, y_mid + y_pad * 0.08, "Slow", ha="left", va="bottom", color="#111827", fontsize=13)

    for _, row in plot.iterrows():
        team = str(row["Team"])
        color = normalize_color(None, team)
        ax.scatter(row["Mean Speed"], row["Top Speed"], s=190, color=color, edgecolor="#FFFFFF", linewidth=1.6, zorder=5)
        ax.text(
            row["Mean Speed"] + x_pad * 0.055,
            row["Top Speed"] + y_pad * 0.045,
            team.replace(" Racing", "").replace(" F1 Team", ""),
            color=color,
            fontsize=10.5,
            fontweight="bold",
            zorder=6,
        )

    ax.set_title(
        f"{year} {event_name} - {session_name} (Best Lap of Each Team)",
        color="#111827",
        fontsize=17,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel("Mean Speed (km/h)", fontsize=13, color="#111827")
    ax.set_ylabel("Top Speed (km/h)", fontsize=13, color="#111827")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, color="#E5E7EB", linewidth=0.9)
    ax.tick_params(colors="#111827", labelsize=10)
    for spine in ax.spines.values():
        spine.set_color("#111827")
        spine.set_linewidth(1.1)
    fig.tight_layout()
    return fig


def speed_trap_graphic(data: pd.DataFrame, event_name: str, session_name: str, headshots: dict[str, str], top_n: int = 10):
    if data.empty:
        return None

    plot = data.sort_values("Speed Trap", ascending=False).head(top_n).copy().reset_index(drop=True)
    fastest = float(plot["Speed Trap"].max())
    slowest = float(plot["Speed Trap"].min())
    spread = max(fastest - slowest, 1.0)
    baseline = max(slowest - spread * 0.5, 0.0)
    plot["Visual Speed"] = plot["Speed Trap"] - baseline

    fig_height = max(5.5, 1.0 + len(plot) * 0.7)
    fig, ax = plt.subplots(figsize=(13.5, fig_height), facecolor="#07090D")
    ax.set_facecolor("#0D1118")
    colors = [normalize_color(None, team) for team in plot["Team"]]
    y_values = np.arange(len(plot))
    ax.barh(y_values, plot["Visual Speed"], left=baseline, color=colors, edgecolor="#F8FAFC", linewidth=1.1, height=0.54)
    ax.invert_yaxis()

    ax.set_title("SPEED TRAP LEADERS", color="#FFFFFF", fontsize=24, fontweight="bold", loc="left", pad=28)
    ax.text(1, 1.075, f"{event_name} - {session_name}", transform=ax.transAxes, ha="right", va="center", color="#F8FAFC", fontsize=15, fontweight="bold")
    ax.text(0, 1.015, "Maximum speed recorded on each driver's fastest telemetry lap", transform=ax.transAxes, ha="left", va="center", color="#A7B0C0", fontsize=11)

    x_min = baseline - spread * 0.42
    x_max = fastest + spread * 0.62
    image_x = baseline - spread * 0.22
    label_x = baseline - spread * 0.06
    value_x = fastest + spread * 0.12

    for idx, (_, row) in enumerate(plot.iterrows()):
        driver = str(row["Driver"])
        team = str(row["Team"])
        speed_mph = float(row["Speed Trap"]) * KMH_TO_MPH
        delta_mph = (float(row["Speed Trap"]) - fastest) * KMH_TO_MPH
        image = load_plot_image(headshots.get(driver))
        if image is not None:
            ax.add_artist(AnnotationBbox(OffsetImage(image, zoom=0.22), (image_x, idx), frameon=False, zorder=10))
        else:
            draw_fallback_badge(ax, image_x, idx, driver, normalize_color(None, team), 0.22)
        ax.text(x_min, idx, f"{idx + 1:02d}", ha="left", va="center", color="#F8FAFC" if idx < 3 else "#7E8796", fontsize=13, fontweight="bold")
        ax.text(label_x, idx - 0.13, driver, ha="left", va="center", color="#FFFFFF", fontsize=13, fontweight="bold")
        ax.text(label_x, idx + 0.17, team, ha="left", va="center", color="#A7B0C0", fontsize=10)
        ax.text(value_x, idx - 0.12, f"{speed_mph:.1f} mph", ha="left", va="center", color="#F8FAFC", fontsize=13, fontweight="bold")
        ax.text(value_x, idx + 0.17, "Leader" if idx == 0 else f"{delta_mph:.1f} mph", ha="left", va="center", color="#A7B0C0", fontsize=9.5)

    ax.set_xlim(x_min, x_max)
    ax.set_yticks([])
    ax.set_xlabel("Speed trap", color="#A7B0C0", fontsize=10, fontweight="bold")
    ax.tick_params(axis="x", colors="#7E8796", labelsize=9)
    ax.xaxis.grid(True, color="#273244", alpha=0.5, linestyle="--")
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return fig


def theoretical_lap_graphic(data: pd.DataFrame, event_name: str, session_name: str, headshots: dict[str, str], top_n: int = 10):
    if data.empty:
        return None

    plot = data.sort_values("Best Theoretical Seconds").head(top_n).copy().reset_index(drop=True)
    leader = float(plot["Best Theoretical Seconds"].min())
    fig_height = max(5.8, 1.1 + len(plot) * 0.78)
    fig, ax = plt.subplots(figsize=(13.5, fig_height), facecolor="#07090D")
    ax.set_facecolor("#0D1118")

    y_values = np.arange(len(plot))
    left = np.zeros(len(plot))
    sector_cols = ["S1 Seconds", "S2 Seconds", "S3 Seconds"]
    sector_colors = ["#38BDF8", "#A3E635", "#F97316"]
    for col, color, label in zip(sector_cols, sector_colors, ["S1", "S2", "S3"]):
        ax.barh(y_values, plot[col], left=left, color=color, edgecolor="#0D1118", linewidth=1.2, height=0.54, label=label)
        left += plot[col].to_numpy(dtype=float)
    ax.invert_yaxis()

    ax.set_title("BEST THEORETICAL LAP", color="#FFFFFF", fontsize=24, fontweight="bold", loc="left", pad=28)
    ax.text(1, 1.075, f"{event_name} - {session_name}", transform=ax.transAxes, ha="right", va="center", color="#F8FAFC", fontsize=15, fontweight="bold")
    ax.text(0, 1.015, "Best sector combination by driver", transform=ax.transAxes, ha="left", va="center", color="#A7B0C0", fontsize=11)

    total_max = float(plot["Best Theoretical Seconds"].max())
    image_x = -total_max * 0.13
    label_x = -total_max * 0.08
    value_x = total_max * 1.02
    for idx, (_, row) in enumerate(plot.iterrows()):
        driver = str(row["Driver"])
        team = str(row["Team"])
        total = float(row["Best Theoretical Seconds"])
        delta = total - leader
        image = load_plot_image(headshots.get(driver))
        if image is not None:
            ax.add_artist(AnnotationBbox(OffsetImage(image, zoom=0.2), (image_x, idx), frameon=False, zorder=10))
        else:
            draw_fallback_badge(ax, image_x, idx, driver, normalize_color(None, team), 0.2)
        ax.text(label_x, idx - 0.13, driver, ha="left", va="center", color="#FFFFFF", fontsize=13, fontweight="bold")
        ax.text(label_x, idx + 0.17, team, ha="left", va="center", color="#A7B0C0", fontsize=9.5)
        ax.text(value_x, idx - 0.12, format_time(total), ha="left", va="center", color="#F8FAFC", fontsize=13, fontweight="bold")
        ax.text(value_x, idx + 0.17, "Ideal leader" if idx == 0 else f"+{delta:.3f}s", ha="left", va="center", color="#A7B0C0", fontsize=9.5)

    ax.set_xlim(-total_max * 0.16, total_max * 1.2)
    ax.set_yticks([])
    ax.set_xlabel("Theoretical sector stack", color="#A7B0C0", fontsize=10, fontweight="bold")
    ax.tick_params(axis="x", colors="#7E8796", labelsize=9)
    ax.legend(loc="lower right", frameon=False, labelcolor="#E5E7EB")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return fig


def f1_strategy_data(laps: pd.DataFrame) -> pd.DataFrame:
    required = {"Driver", "Team", "LapNumber", "Stint", "Compound"}
    if laps.empty or not required.issubset(laps.columns):
        return pd.DataFrame()

    data = laps.dropna(subset=["Driver", "Team", "LapNumber", "Stint", "Compound"]).copy()
    if data.empty:
        return pd.DataFrame()

    data["LapNumber"] = pd.to_numeric(data["LapNumber"], errors="coerce")
    data["Stint"] = pd.to_numeric(data["Stint"], errors="coerce")
    rows = (
        data.groupby(["Driver", "Team", "Stint", "Compound"], as_index=False)
        .agg(
            **{
                "Start Lap": ("LapNumber", "min"),
                "End Lap": ("LapNumber", "max"),
                "Laps": ("LapNumber", "count"),
            }
        )
        .sort_values(["Driver", "Start Lap", "Stint"])
    )
    rows["Stint"] = rows["Stint"].astype(int)
    rows["Start Lap"] = rows["Start Lap"].astype(int)
    rows["End Lap"] = rows["End Lap"].astype(int)
    rows["Laps"] = rows["Laps"].astype(int)
    return rows.reset_index(drop=True)


def f1_strategy_graphic(data: pd.DataFrame, event_name: str, session_name: str, top_n: int = 20):
    if data.empty:
        return None

    totals = (
        data.groupby(["Driver", "Team"], as_index=False)["Laps"]
        .sum()
        .sort_values("Laps", ascending=False)
        .head(top_n)
    )
    drivers = totals["Driver"].tolist()
    plot = data[data["Driver"].isin(drivers)].copy()
    order = {driver: idx for idx, driver in enumerate(drivers)}
    plot["DriverOrder"] = plot["Driver"].map(order)
    plot = plot.sort_values(["DriverOrder", "Start Lap", "Stint"])

    compound_colors = {
        "SOFT": "#EF4444",
        "MEDIUM": "#FACC15",
        "HARD": "#F8FAFC",
        "INTERMEDIATE": "#22C55E",
        "WET": "#3B82F6",
        "UNKNOWN": "#64748B",
        "TEST_UNKNOWN": "#64748B",
    }

    fig_height = max(5.8, 1.0 + len(drivers) * 0.45)
    fig, ax = plt.subplots(figsize=(13.5, fig_height), facecolor="#07090D")
    ax.set_facecolor("#0D1118")

    for y, driver in enumerate(drivers):
        driver_stints = plot[plot["Driver"] == driver]
        left = 0
        team = str(driver_stints["Team"].iloc[0])
        for _, stint in driver_stints.iterrows():
            compound = str(stint["Compound"]).upper()
            laps_count = int(stint["Laps"])
            color = compound_colors.get(compound, "#94A3B8")
            edge = "#111827" if compound == "HARD" else "#F8FAFC"
            ax.barh(y, laps_count, left=left, color=color, edgecolor=edge, linewidth=1.0, height=0.58)
            label = f"{compound[:3]} {laps_count}"
            if laps_count >= 4:
                ax.text(
                    left + laps_count / 2,
                    y,
                    label,
                    ha="center",
                    va="center",
                    color="#111827" if compound in {"MEDIUM", "HARD"} else "#FFFFFF",
                    fontsize=8.5,
                    fontweight="bold",
                )
            left += laps_count
        ax.text(-max(totals["Laps"]) * 0.035, y, driver, ha="right", va="center", color="#FFFFFF", fontsize=10.5, fontweight="bold")
        ax.text(left + max(totals["Laps"]) * 0.012, y, team, ha="left", va="center", color="#A7B0C0", fontsize=8.5)

    ax.set_title("RACE STRATEGY", color="#FFFFFF", fontsize=24, fontweight="bold", loc="left", pad=28)
    ax.text(1, 1.075, f"{event_name} - {session_name}", transform=ax.transAxes, ha="right", va="center", color="#F8FAFC", fontsize=15, fontweight="bold")
    ax.text(0, 1.015, "Tyre compound and stint length by driver", transform=ax.transAxes, ha="left", va="center", color="#A7B0C0", fontsize=11)
    ax.set_yticks([])
    ax.set_xlabel("Race laps", color="#A7B0C0", fontsize=10, fontweight="bold")
    ax.tick_params(axis="x", colors="#7E8796", labelsize=9)
    ax.xaxis.grid(True, color="#273244", alpha=0.45, linestyle="--")
    ax.set_axisbelow(True)
    ax.invert_yaxis()
    ax.set_xlim(-max(totals["Laps"]) * 0.18, max(totals["Laps"]) * 1.2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return fig


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def f1_season_pitstop_data(year: int, through_round: int | None = None) -> pd.DataFrame:
    if year < 2023:
        return pd.DataFrame()

    sessions = pd.DataFrame(load_openf1_json("sessions", f"year={year}&session_name=Race"))
    if sessions.empty:
        return pd.DataFrame()

    sessions["date_start"] = pd.to_datetime(sessions["date_start"], errors="coerce", utc=True)
    sessions = sessions[(sessions["date_start"].notna()) & (sessions["date_start"] <= pd.Timestamp.now(tz="UTC"))]
    sessions = sessions.sort_values("date_start")
    if through_round is not None:
        sessions = sessions.head(int(through_round))

    rows = []
    for round_index, (_, session_row) in enumerate(sessions.iterrows(), start=1):
        session_key = int(session_row["session_key"])
        meeting_key = int(session_row["meeting_key"])
        try:
            pits = load_openf1_json("pit", f"session_key={session_key}")
            drivers = load_openf1_json("drivers", f"session_key={session_key}")
        except Exception:
            continue

        driver_map = {
            int(driver["driver_number"]): {
                "Driver": driver.get("name_acronym") or driver.get("broadcast_name") or driver.get("full_name"),
                "Team": driver.get("team_name"),
                "Team Color": normalize_color(driver.get("team_colour"), driver.get("team_name")),
            }
            for driver in drivers
            if driver.get("driver_number") is not None
        }

        for stop in pits:
            stop_duration = numeric_value(stop.get("stop_duration"), np.nan)
            driver_number = stop.get("driver_number")
            driver_info = driver_map.get(int(driver_number)) if driver_number is not None else None
            if driver_info and pd.notna(stop_duration):
                rows.append(
                    {
                        "Year": year,
                        "Round": round_index,
                        "Meeting Key": meeting_key,
                        "Session Key": session_key,
                        "Race": session_row.get("meeting_name") or session_row.get("location"),
                        "Driver": driver_info["Driver"],
                        "Driver Number": driver_number,
                        "Team": driver_info["Team"],
                        "Team Color": driver_info["Team Color"],
                        "Lap": stop.get("lap_number"),
                        "Stop Duration": stop_duration,
                        "Pit Lane Duration": numeric_value(stop.get("pit_duration") or stop.get("lane_duration"), np.nan),
                    }
                )

    return pd.DataFrame(rows)


def filter_normal_f1_pitstops(data: pd.DataFrame, max_duration: float) -> pd.DataFrame:
    if data.empty:
        return data
    filtered = data[(data["Stop Duration"] >= 1.5) & (data["Stop Duration"] <= max_duration)].copy()
    if filtered.empty:
        return filtered
    q1 = float(filtered["Stop Duration"].quantile(0.25))
    q3 = float(filtered["Stop Duration"].quantile(0.75))
    iqr = max(q3 - q1, 0.1)
    upper = min(float(max_duration), q3 + 1.5 * iqr)
    return filtered[filtered["Stop Duration"] <= upper].copy()


def f1_pit_crews_summary(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame()
    summary = (
        data.groupby("Team", as_index=False)
        .agg(
            **{
                "Average Stop": ("Stop Duration", "mean"),
                "Median Stop": ("Stop Duration", "median"),
                "Best Stop": ("Stop Duration", "min"),
                "Stops Counted": ("Stop Duration", "count"),
            }
        )
        .sort_values("Average Stop")
    )
    return summary.reset_index(drop=True)


def f1_pit_crews_graphic(data: pd.DataFrame, year: int, top_n: int = 10):
    if data.empty:
        return None
    plot = data.sort_values("Average Stop").head(top_n).copy().reset_index(drop=True)
    fastest = float(plot["Average Stop"].min())
    slowest = float(plot["Average Stop"].max())
    spread = max(slowest - fastest, 0.25)
    baseline = max(fastest - spread * 0.55, 0.0)
    plot["Visual Stop"] = plot["Average Stop"] - baseline

    fig_height = max(5.5, 1.0 + len(plot) * 0.72)
    fig, ax = plt.subplots(figsize=(13.5, fig_height), facecolor="#07090D")
    ax.set_facecolor("#0D1118")
    y_values = np.arange(len(plot))
    colors = [normalize_color(None, team) for team in plot["Team"]]
    ax.barh(y_values, plot["Visual Stop"], left=baseline, color=colors, edgecolor="#F8FAFC", linewidth=1.1, height=0.54)
    ax.invert_yaxis()

    ax.set_title("FASTEST PIT CREWS", color="#FFFFFF", fontsize=24, fontweight="bold", loc="left", pad=28)
    ax.text(1, 1.075, f"{year} season", transform=ax.transAxes, ha="right", va="center", color="#F8FAFC", fontsize=15, fontweight="bold")
    ax.text(0, 1.015, "Average OpenF1 stop_duration, filtered for normal service stops", transform=ax.transAxes, ha="left", va="center", color="#A7B0C0", fontsize=11)

    x_min = baseline - spread * 0.42
    x_max = slowest + spread * 0.62
    image_x = baseline - spread * 0.22
    label_x = baseline - spread * 0.06
    value_x = slowest + spread * 0.12

    for idx, (_, row) in enumerate(plot.iterrows()):
        team = str(row["Team"])
        image = load_plot_image(team_logo_source(team))
        if image is not None:
            ax.add_artist(AnnotationBbox(OffsetImage(image, zoom=0.18), (image_x, idx), frameon=False, zorder=10))
        else:
            draw_fallback_badge(ax, image_x, idx, team, normalize_color(None, team), 0.18)
        ax.text(x_min, idx, f"{idx + 1:02d}", ha="left", va="center", color="#F8FAFC" if idx < 3 else "#7E8796", fontsize=13, fontweight="bold")
        ax.text(label_x, idx - 0.13, team, ha="left", va="center", color="#FFFFFF", fontsize=13, fontweight="bold")
        ax.text(label_x, idx + 0.17, f"{int(row['Stops Counted'])} stops | best {row['Best Stop']:.2f}s", ha="left", va="center", color="#A7B0C0", fontsize=9.5)
        ax.text(value_x, idx - 0.12, f"{row['Average Stop']:.2f}s", ha="left", va="center", color="#F8FAFC", fontsize=13, fontweight="bold")
        ax.text(value_x, idx + 0.17, "Leader" if idx == 0 else f"+{row['Average Stop'] - fastest:.2f}s", ha="left", va="center", color="#A7B0C0", fontsize=9.5)

    ax.set_xlim(x_min, x_max)
    ax.set_yticks([])
    ax.set_xlabel("Average stop duration", color="#A7B0C0", fontsize=10, fontweight="bold")
    ax.tick_params(axis="x", colors="#7E8796", labelsize=9)
    ax.xaxis.grid(True, color="#273244", alpha=0.5, linestyle="--")
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return fig


def standings_graphic(data: pd.DataFrame, mode: str, year: int, through_round: int, top_n: int = 10):
    if data.empty:
        return None

    plot = data.sort_values("Points", ascending=False).head(top_n).copy().reset_index(drop=True)
    leader = float(plot["Points"].max())
    fig_height = max(5.7, 1.0 + len(plot) * 0.72)
    fig, ax = plt.subplots(figsize=(13.5, fig_height), facecolor="#07090D")
    ax.set_facecolor("#0D1118")

    y_values = np.arange(len(plot))
    colors = [normalize_color(None, team) for team in plot["Team"]]
    ax.barh(y_values, plot["Points"], color=colors, edgecolor="#F8FAFC", linewidth=1.1, height=0.54)
    ax.invert_yaxis()
    title = "DRIVERS' STANDINGS" if mode == "Drivers" else "CONSTRUCTORS' STANDINGS"
    ax.set_title(title, color="#FFFFFF", fontsize=24, fontweight="bold", loc="left", pad=28)
    ax.text(1, 1.075, f"{year} season | Through round {through_round}", transform=ax.transAxes, ha="right", va="center", color="#F8FAFC", fontsize=15, fontweight="bold")
    ax.text(0, 1.015, "Championship points leaderboard", transform=ax.transAxes, ha="left", va="center", color="#A7B0C0", fontsize=11)

    x_max = max(leader * 1.22, 1.0)
    image_x = -x_max * 0.08
    label_x = -x_max * 0.035
    value_x = leader * 1.04
    for idx, (_, row) in enumerate(plot.iterrows()):
        name = str(row["Name"])
        team = str(row["Team"])
        points = float(row["Points"])
        gap = leader - points
        image = load_plot_image(row.get("Logo"))
        if image is not None:
            ax.add_artist(AnnotationBbox(OffsetImage(image, zoom=0.18), (image_x, idx), frameon=False, zorder=10))
        else:
            draw_fallback_badge(ax, image_x, idx, name, normalize_color(None, team), 0.18)
        ax.text(-x_max * 0.14, idx, f"{int(row['Pos']):02d}", ha="left", va="center", color="#F8FAFC" if idx < 3 else "#7E8796", fontsize=13, fontweight="bold")
        ax.text(label_x, idx - 0.13, name, ha="left", va="center", color="#FFFFFF", fontsize=13, fontweight="bold")
        detail = team if mode == "Drivers" else f"Podiums {row.get('Podiums', 0)} | Poles {row.get('Poles', 0)}"
        ax.text(label_x, idx + 0.17, detail, ha="left", va="center", color="#A7B0C0", fontsize=9.5)
        ax.text(value_x, idx - 0.12, f"{points:.1f} pts", ha="left", va="center", color="#F8FAFC", fontsize=13, fontweight="bold")
        ax.text(value_x, idx + 0.17, "Leader" if idx == 0 else f"-{gap:.1f}", ha="left", va="center", color="#A7B0C0", fontsize=9.5)

    ax.set_xlim(-x_max * 0.16, x_max)
    ax.set_yticks([])
    ax.set_xlabel("Points", color="#A7B0C0", fontsize=10, fontweight="bold")
    ax.tick_params(axis="x", colors="#7E8796", labelsize=9)
    ax.xaxis.grid(True, color="#273244", alpha=0.5, linestyle="--")
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return fig


def nascar_ranking_graphic(
    data: pd.DataFrame,
    title: str,
    subtitle: str,
    metric_col: str,
    lower_is_better: bool,
    top_n: int = 10,
    value_suffix: str = "",
    value_format: str = "{:.2f}",
):
    if data.empty or metric_col not in data:
        return None

    plot = data.dropna(subset=[metric_col]).copy()
    if plot.empty:
        return None
    plot = plot.sort_values(metric_col, ascending=lower_is_better).head(top_n).reset_index(drop=True)
    values = plot[metric_col].to_numpy(dtype=float)
    best = float(values.min() if lower_is_better else values.max())
    worst = float(values.max() if lower_is_better else values.min())
    spread = max(abs(worst - best), 1.0)
    baseline = max(float(values.min()) - spread * 0.5, 0.0)
    plot["Visual Value"] = plot[metric_col] - baseline

    fig_height = max(5.6, 1.0 + len(plot) * 0.7)
    fig, ax = plt.subplots(figsize=(13.5, fig_height), facecolor="#07090D")
    ax.set_facecolor("#0D1118")
    y_values = np.arange(len(plot))
    colors = ["#D71920" if str(row.get("Manufacturer", "")).lower().startswith("toy") else "#2563EB" if str(row.get("Manufacturer", "")).lower().startswith("ford") else "#F97316" for _, row in plot.iterrows()]
    ax.barh(y_values, plot["Visual Value"], left=baseline, color=colors, edgecolor="#F8FAFC", linewidth=1.1, height=0.54)
    ax.invert_yaxis()

    ax.set_title(title.upper(), color="#FFFFFF", fontsize=24, fontweight="bold", loc="left", pad=28)
    ax.text(0, 1.015, subtitle, transform=ax.transAxes, ha="left", va="center", color="#A7B0C0", fontsize=11)

    x_min = baseline - spread * 0.42
    x_max = float(values.max()) + spread * 0.62
    image_x = baseline - spread * 0.22
    label_x = baseline - spread * 0.06
    value_x = float(values.max()) + spread * 0.12

    for idx, (_, row) in enumerate(plot.iterrows()):
        driver = str(row.get("Driver", row.get("Name", "")))
        team = str(row.get("Team", row.get("Manufacturer", "")))
        value = float(row[metric_col])
        delta = value - best if lower_is_better else best - value
        badge = load_plot_image(row.get("Badge"))
        color = colors[idx]
        if badge is not None:
            ax.add_artist(AnnotationBbox(OffsetImage(badge, zoom=0.18), (image_x, idx), frameon=False, zorder=10))
        else:
            draw_fallback_badge(ax, image_x, idx, str(row.get("Number", driver)), color, 0.18)
        ax.text(x_min, idx, f"{idx + 1:02d}", ha="left", va="center", color="#F8FAFC" if idx < 3 else "#7E8796", fontsize=13, fontweight="bold")
        ax.text(label_x, idx - 0.13, driver, ha="left", va="center", color="#FFFFFF", fontsize=13, fontweight="bold")
        ax.text(label_x, idx + 0.17, f"#{row.get('Number', '-')} | {team}", ha="left", va="center", color="#A7B0C0", fontsize=9.5)
        ax.text(value_x, idx - 0.12, value_format.format(value) + value_suffix, ha="left", va="center", color="#F8FAFC", fontsize=13, fontweight="bold")
        ax.text(value_x, idx + 0.17, "Leader" if idx == 0 else ("+" if lower_is_better else "-") + value_format.format(abs(delta)) + value_suffix, ha="left", va="center", color="#A7B0C0", fontsize=9.5)

    ax.set_xlim(x_min, x_max)
    ax.set_yticks([])
    ax.set_xlabel(metric_col, color="#A7B0C0", fontsize=10, fontweight="bold")
    ax.tick_params(axis="x", colors="#7E8796", labelsize=9)
    ax.xaxis.grid(True, color="#273244", alpha=0.5, linestyle="--")
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return fig


def build_section_rankings(telemetry: list[TelemetryLap], corners: pd.DataFrame, track_length: float, radius: float):
    rows = []
    for item in telemetry:
        distances = item.frame["Distance"].to_numpy(dtype=float)
        c_mask = corner_mask(distances, corners, track_length, radius)
        s_mask = ~c_mask
        rows.append(
            {
                "Driver": item.driver,
                "Team": item.team,
                "Best Lap": format_time(item.lap_time),
                "Corner Avg Speed": mean_speed_in_mask(item.frame, c_mask),
                "Straight Avg Speed": mean_speed_in_mask(item.frame, s_mask),
                "Speed Trap": float(item.frame["Speed"].max()),
            }
        )
    driver = pd.DataFrame(rows)
    team = (
        driver.groupby("Team", as_index=False)
        .agg(
            **{
                "Corner Avg Speed": ("Corner Avg Speed", "max"),
                "Straight Avg Speed": ("Straight Avg Speed", "max"),
                "Speed Trap": ("Speed Trap", "max"),
            }
        )
    )
    return driver, team


def minimum_corner_speed_leaderboard(
    telemetry: list[TelemetryLap], corners: pd.DataFrame, track_length: float, radius: float
) -> pd.DataFrame:
    rows = []
    for item in telemetry:
        for _, corner in corners.iterrows():
            distances = item.frame["Distance"].to_numpy(dtype=float)
            mask = corner_mask(distances, pd.DataFrame([corner]), track_length, radius)
            if mask.any():
                rows.append(
                    {
                        "Corner": corner["Label"],
                        "Driver": item.driver,
                        "Team": item.team,
                        "Minimum Speed": float(item.frame.loc[mask, "Speed"].min()),
                    }
                )
    data = pd.DataFrame(rows)
    if data.empty:
        return data
    idx = data.groupby("Corner")["Minimum Speed"].idxmax()
    return data.loc[idx].sort_values("Minimum Speed", ascending=False).reset_index(drop=True)


def theoretical_lap_leaderboard(laps: pd.DataFrame) -> pd.DataFrame:
    sector_cols = [col for col in ["Sector1Time", "Sector2Time", "Sector3Time"] if col in laps.columns]
    if len(sector_cols) != 3:
        return pd.DataFrame()
    data = laps.copy()
    for col in sector_cols:
        data[f"{col}Seconds"] = data[col].map(seconds)
    rows = []
    for (driver, team), group in data.groupby(["Driver", "Team"]):
        sectors = [float(group[f"{col}Seconds"].min()) for col in sector_cols]
        if all(np.isfinite(sectors)):
            rows.append(
                {
                    "Driver": driver,
                    "Team": team,
                    "S1 Seconds": sectors[0],
                    "S2 Seconds": sectors[1],
                    "S3 Seconds": sectors[2],
                    "Best Theoretical Seconds": sum(sectors),
                    "S1": format_time(sectors[0]),
                    "S2": format_time(sectors[1]),
                    "S3": format_time(sectors[2]),
                    "Best Theoretical": sum(sectors),
                }
            )
    result = pd.DataFrame(rows).sort_values("Best Theoretical Seconds").reset_index(drop=True)
    if not result.empty:
        result["Best Theoretical"] = result["Best Theoretical"].map(format_time)
    return result


def short_race_name(event_name: str) -> str:
    return (
        event_name.replace(" Grand Prix", "")
        .replace("Grand Prix", "")
        .replace("Emilia Romagna", "Imola")
        .strip()
    )


def best_driver_time_seconds(results: pd.DataFrame) -> pd.Series:
    times = results[TIME_COLUMNS].apply(pd.to_timedelta, errors="coerce")
    return times.min(axis=1).dt.total_seconds()


def is_clean_race_pace_lap(laps: pd.DataFrame) -> pd.Series:
    mask = laps["LapTimeSeconds"].between(55, 180)
    if "IsAccurate" in laps.columns:
        mask &= laps["IsAccurate"].fillna(True)
    for column in ["PitInTime", "PitOutTime"]:
        if column in laps.columns:
            mask &= laps[column].isna()
    if "TrackStatus" in laps.columns:
        mask &= laps["TrackStatus"].astype(str).isin(["1", "nan", "<NA>"])
    if "FreshTyre" in laps.columns:
        mask &= laps["FreshTyre"].fillna(False).astype(bool) | laps["LapNumber"].notna()
    return mask


def is_clean_qualifying_lap(laps: pd.DataFrame) -> pd.Series:
    mask = laps["LapTimeSeconds"].between(55, 180)
    if "IsAccurate" in laps.columns:
        mask &= laps["IsAccurate"].fillna(True)
    for column in ["PitInTime", "PitOutTime"]:
        if column in laps.columns:
            mask &= laps[column].isna()
    return mask


def slugify_asset_name(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in str(value)).strip("_")


def team_logo_path(team: str) -> Path | None:
    slug = slugify_asset_name(team)
    candidates = [
        TEAM_LOGO_DIR / f"{slug}.png",
        TEAM_LOGO_DIR / f"{slug}.jpg",
        TEAM_LOGO_DIR / f"{slug}.jpeg",
    ]
    return next((path for path in candidates if path.exists()), None)


def team_wiki_title(team: str) -> str | None:
    normalized = str(team).casefold()
    for key, title in TEAM_WIKI_TITLES.items():
        if key in normalized:
            return title
    return None


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24 * 7)
def wiki_team_logo_url(team: str) -> str | None:
    title = team_wiki_title(team)
    if not title:
        return None
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(title)}"
    try:
        request = Request(url, headers={"User-Agent": "F1SessionIntelligence/1.0"})
        with urlopen(request, timeout=8) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return None

    thumbnail = payload.get("thumbnail") or {}
    original = payload.get("originalimage") or {}
    return thumbnail.get("source") or original.get("source")


def team_logo_source(team: str) -> str | Path | None:
    return team_logo_path(team) or wiki_team_logo_url(team)


def load_plot_image(source: str | Path | None) -> np.ndarray | None:
    if source is None:
        return None
    try:
        if isinstance(source, Path):
            image = Image.open(source)
        elif str(source).startswith("data:image/"):
            encoded = str(source).split(",", 1)[1]
            image = Image.open(BytesIO(base64.b64decode(encoded)))
        else:
            with urlopen(source, timeout=5) as response:
                image = Image.open(BytesIO(response.read()))
        return np.asarray(image.convert("RGBA"))
    except Exception:
        return None


def image_file_data_uri(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    suffix = path.suffix.lower().lstrip(".")
    mime = "jpeg" if suffix in {"jpg", "jpeg"} else "png"
    return f"data:image/{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def team_badge_data_uri(team: str) -> str:
    color = normalize_color(None, team)
    initials = "".join(part[0] for part in str(team).replace("F1 Team", "").split()[:2]).upper() or "F1"
    svg = f"""
    <svg xmlns="http://www.w3.org/2000/svg" width="96" height="96" viewBox="0 0 96 96">
      <rect width="96" height="96" rx="18" fill="#0b0f14"/>
      <rect x="6" y="6" width="84" height="84" rx="16" fill="{color}"/>
      <text x="48" y="58" text-anchor="middle" font-family="Arial, sans-serif" font-size="28" font-weight="800" fill="#ffffff">{initials}</text>
    </svg>
    """
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")


def team_identity_image(team: str) -> str:
    local_logo = image_file_data_uri(team_logo_path(team))
    return local_logo or wiki_team_logo_url(team) or team_badge_data_uri(team)


def draw_fallback_badge(ax, x: float, y: float, label: str, color: str, zoom: float) -> None:
    initials = "".join(part[0] for part in str(label).replace("F1 Team", "").split()[:2]).upper() or "F1"
    ax.text(
        x,
        y,
        initials,
        ha="center",
        va="center",
        fontsize=max(7, int(18 * zoom)),
        fontweight="bold",
        color="#FFFFFF",
        bbox={"boxstyle": "round,pad=0.38", "facecolor": color, "edgecolor": "#FFFFFF", "linewidth": 1.1},
        zorder=10,
    )


def figure_to_png(fig) -> BytesIO:
    output = BytesIO()
    fig.savefig(output, format="png", dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    output.seek(0)
    return output


def top_order_from_summary(data: pd.DataFrame, top_n: int) -> list[str]:
    summary = delta_box_summary(data)
    if top_n > 0:
        summary = summary.head(top_n)
    return summary["Name"].tolist()


@st.cache_data(show_spinner=False, ttl=60 * 60 * 12)
def driver_headshot_urls(year: int, round_number: int, session_code: str) -> dict[str, str]:
    try:
        session = load_results_only_session(year, round_number, session_code)
        results = session.results.copy()
    except Exception:
        return {}
    if results.empty or "HeadshotUrl" not in results.columns:
        return {}

    urls: dict[str, str] = {}
    for _, row in results.iterrows():
        url = row.get("HeadshotUrl")
        if pd.isna(url) or not str(url).strip():
            continue
        for column in ["Abbreviation", "DriverNumber", "BroadcastName", "FullName"]:
            value = row.get(column)
            if not pd.isna(value):
                urls[str(value)] = str(url)
    return urls


@st.cache_data(show_spinner=False, ttl=60 * 60 * 12)
def delta_box_data(year: int, round_number: int, pace_type: str, grouping: str) -> pd.DataFrame:
    session_code = "Q" if pace_type == "Qualifying" else "R"
    session = load_laps_only_session(year, round_number, session_code)
    laps = valid_laps(session)
    if laps.empty:
        return pd.DataFrame()

    clean = laps[is_clean_qualifying_lap(laps) if session_code == "Q" else is_clean_race_pace_lap(laps)].copy()
    if clean.empty:
        return pd.DataFrame()

    clean["Name"] = clean["Team"] if grouping == "Teams" else clean["Driver"]
    clean["GroupTeam"] = clean["Team"]
    clean["LapTime (s)"] = clean["LapTimeSeconds"]
    clean["Lap Time"] = clean["LapTimeSeconds"].map(format_time)
    return clean[["Name", "GroupTeam", "Driver", "Team", "LapNumber", "Lap Time", "LapTime (s)"]].reset_index(drop=True)


def delta_box_summary(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return data
    summary = (
        data.groupby(["Name", "GroupTeam"], as_index=False)
        .agg(
            Laps=("LapTime (s)", "count"),
            MedianSeconds=("LapTime (s)", "median"),
            BestSeconds=("LapTime (s)", "min"),
            WorstSeconds=("LapTime (s)", "max"),
        )
        .sort_values("MedianSeconds")
    )
    fastest_median = float(summary["MedianSeconds"].min())
    summary["Median Pace"] = summary["MedianSeconds"].map(format_time)
    summary["Median Delta"] = (summary["MedianSeconds"] - fastest_median).map(
        lambda value: f"+{value:.3f}s" if value > 0 else "0.000s"
    )
    summary["Best Lap"] = summary["BestSeconds"].map(format_time)
    summary["Worst Lap"] = summary["WorstSeconds"].map(format_time)
    return summary[["Name", "GroupTeam", "Laps", "Median Pace", "Median Delta", "Best Lap", "Worst Lap"]]


def qualifying_delta_bar_data(year: int, round_number: int, grouping: str, top_n: int) -> tuple[pd.DataFrame, object]:
    session = load_basic_session(year, round_number, "Q")
    laps = session.laps
    fastest_laps = []

    if grouping == "Drivers":
        for driver in pd.unique(laps["Driver"].dropna()):
            fastest = laps.pick_drivers(driver).pick_fastest()
            if fastest is not None and not pd.isna(fastest.get("LapTime")):
                fastest_laps.append(fastest)
        fastest = Laps(fastest_laps).sort_values(by="LapTime").reset_index(drop=True)
        fastest["Name"] = fastest["Driver"]
        fastest["GroupTeam"] = fastest["Team"]
    else:
        for team in pd.unique(laps["Team"].dropna()):
            fastest = laps.pick_teams(team).pick_fastest()
            if fastest is not None and not pd.isna(fastest.get("LapTime")):
                fastest_laps.append(fastest)
        fastest = Laps(fastest_laps).sort_values(by="LapTime").reset_index(drop=True)
        fastest["Name"] = fastest["Team"]
        fastest["GroupTeam"] = fastest["Team"]

    if fastest.empty:
        return pd.DataFrame(), session

    pole_lap = fastest.pick_fastest()
    fastest["DeltaSeconds"] = (fastest["LapTime"] - pole_lap["LapTime"]).dt.total_seconds()
    fastest["Lap Time"] = fastest["LapTime"].map(lambda value: strftimedelta(value, "%m:%s.%ms"))
    fastest["Delta"] = fastest["DeltaSeconds"].map(lambda value: f"+{value:.3f}s" if value > 0 else "0.000s")
    if top_n > 0:
        fastest = fastest.head(top_n)
    return pd.DataFrame(fastest), session


def qualifying_delta_bar_chart(
    data: pd.DataFrame,
    session,
    title: str,
    grouping: str,
    headshots: dict[str, str],
    presentation: bool,
):
    fig_height = max(7.5 if presentation else 5.8, len(data) * (0.62 if presentation else 0.46))
    fig, ax = plt.subplots(figsize=(17 if presentation else 13.5, fig_height), facecolor="#07090D")
    ax.set_facecolor("#0D1118")

    colors = [normalize_color(None, team) for team in data["GroupTeam"]]
    y_values = np.arange(len(data))
    bars = ax.barh(y_values, data["DeltaSeconds"], color=colors, edgecolor="#D7DEE8", linewidth=1.1)
    ax.set_yticks(y_values)
    ax.set_yticklabels(data["Name"], color="#D7DEE8", fontsize=13 if presentation else 10)
    ax.invert_yaxis()
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, which="major", linestyle="--", color="#344054", alpha=0.7)

    max_delta = max(float(data["DeltaSeconds"].max()), 0.12)
    zoom = 0.31 if presentation else 0.22
    for idx, (_, row) in enumerate(data.iterrows()):
        team = str(row["GroupTeam"])
        name = str(row["Name"])
        x_position = max(float(row["DeltaSeconds"]) * 0.52, max_delta * 0.035)
        image = load_plot_image(team_logo_source(team)) if grouping == "Teams" else load_plot_image(headshots.get(name))
        if image is not None:
            ax.add_artist(AnnotationBbox(OffsetImage(image, zoom=zoom), (x_position, idx), frameon=False, zorder=10))
        else:
            draw_fallback_badge(ax, x_position, idx, name if grouping == "Drivers" else team, normalize_color(None, team), zoom)
        ax.text(
            float(row["DeltaSeconds"]) + max_delta * 0.018,
            idx,
            row["Delta"],
            va="center",
            ha="left",
            color="#F8FAFC",
            fontsize=12 if presentation else 9.5,
            fontweight="bold",
        )

    pole_lap = session.laps.pick_fastest()
    lap_time_string = strftimedelta(pole_lap["LapTime"], "%m:%s.%ms")
    ax.set_title(
        f"{title}\nFastest Lap: {lap_time_string} ({pole_lap['Driver']})",
        color="#FFFFFF",
        fontsize=20 if presentation else 15,
        fontweight="bold",
        pad=16,
    )
    ax.set_xlabel("Delta to fastest lap (s)", color="#D7DEE8", fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", colors="#D7DEE8")
    ax.set_xlim(0, max_delta * 1.22)
    for spine in ax.spines.values():
        spine.set_color("#273244")
    fig.tight_layout()
    return fig


def delta_box_plot(data: pd.DataFrame, title: str, grouping: str, headshots: dict[str, str], top_n: int, presentation: bool):
    summary = delta_box_summary(data)
    if top_n > 0:
        summary = summary.head(top_n)
    order = summary["Name"].tolist()
    data = data[data["Name"].isin(order)].copy()
    palette = {
        name: normalize_color(None, data.loc[data["Name"] == name, "GroupTeam"].iloc[0])
        for name in order
    }

    fig_width = max(17 if presentation else 13.5, len(order) * (0.92 if presentation else 0.72))
    fig_height = 10 if presentation else 8.5
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor="#07090D")
    ax.set_facecolor("#0D1118")
    sns.boxplot(
        data=data,
        x="Name",
        y="LapTime (s)",
        hue="Name",
        order=order,
        palette=palette,
        whiskerprops={"color": "white", "linewidth": 1.2},
        boxprops={"edgecolor": "white", "linewidth": 1.2},
        medianprops={"color": "#E5E7EB", "linewidth": 2.1},
        capprops={"color": "white", "linewidth": 1.2},
        flierprops={"marker": "o", "markerfacecolor": "#F8FAFC", "markeredgecolor": "#111827", "markersize": 3.5},
        legend=False,
        ax=ax,
    )

    y_span = max(data["LapTime (s)"].max() - data["LapTime (s)"].min(), 1.0)
    zoom = max(0.22 if presentation else 0.18, min(0.42 if presentation else 0.34, 5.5 / max(len(order), 1)))
    for idx, name in enumerate(order):
        group = data[data["Name"] == name]
        team = str(group["GroupTeam"].iloc[0])
        median = float(group["LapTime (s)"].median())
        image = None
        if grouping == "Teams":
            image = load_plot_image(team_logo_source(team))
        else:
            image = load_plot_image(headshots.get(str(name)))

        if image is not None:
            artist = AnnotationBbox(OffsetImage(image, zoom=zoom), (idx, median), frameon=False, zorder=10)
            ax.add_artist(artist)
        else:
            draw_fallback_badge(ax, idx, median, str(name if grouping == "Drivers" else team), normalize_color(None, team), zoom)

    ax.set_title(title, color="#FFFFFF", fontsize=22 if presentation else 18, fontweight="bold", pad=18)
    ax.set_xlabel("")
    ax.set_ylabel("LapTime (s)", color="#D7DEE8", fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", colors="#D7DEE8", rotation=35, labelsize=12 if presentation else 10)
    ax.tick_params(axis="y", colors="#D7DEE8")
    ax.grid(visible=False)
    ax.set_ylim(data["LapTime (s)"].min() - y_span * 0.08, data["LapTime (s)"].max() + y_span * 0.14)
    for spine in ax.spines.values():
        spine.set_color("#273244")
    fig.tight_layout()
    return fig


def safe_position(value) -> float:
    number = pd.to_numeric(value, errors="coerce")
    return float(number) if pd.notna(number) else np.nan


def is_retirement_status(status: object) -> bool:
    if pd.isna(status):
        return False
    text = str(status).strip().casefold()
    if not text:
        return False
    if text == "finished" or text.startswith("+"):
        return False
    return True


def current_top_streak(positions: list[float], threshold: int) -> int:
    streak = 0
    for position in reversed(positions):
        if pd.notna(position) and position <= threshold:
            streak += 1
        else:
            break
    return streak


def fmt_gap(value: float) -> str:
    if pd.isna(value):
        return "-"
    return f"+{value:.3f}s" if value > 0 else "0.000s"


def numeric_value(value, default: float = 0.0) -> float:
    number = pd.to_numeric(value, errors="coerce")
    return float(number) if pd.notna(number) else default


def first_present(row: pd.Series, columns: list[str], default: str = "-") -> str:
    for column in columns:
        value = row.get(column)
        if pd.notna(value) and str(value).strip():
            return str(value)
    return default


def average_or_nan(values: list[float]) -> float:
    clean = [float(value) for value in values if pd.notna(value)]
    return float(np.mean(clean)) if clean else np.nan


def add_stat_entry(stats: dict, key: str, name: str, team: str | None = None, image: str | None = None) -> dict:
    if key not in stats:
        stats[key] = {
            "Name": name,
            "Team": team or name,
            "Image": image,
            "Points": 0.0,
            "RacePositions": [],
            "TopStreakPositions": [],
            "QualifyingPositions": [],
            "RacePaceGaps": [],
            "QualifyingGaps": [],
            "DNFs": 0,
            "SprintWins": 0,
            "SprintPodiums": 0,
            "Poles": 0,
            "Podiums": 0,
        }
    if team:
        stats[key]["Team"] = team
    if image and not stats[key].get("Image"):
        stats[key]["Image"] = image
    return stats[key]


def qualifying_gap_rows(session, mode: str) -> tuple[dict[str, float], dict[str, float]]:
    laps = valid_laps(session)
    if laps.empty:
        return {}, {}
    clean = laps[is_clean_qualifying_lap(laps)].copy()
    if clean.empty:
        return {}, {}
    driver_best = clean.groupby("Driver")["LapTimeSeconds"].min()
    team_best = clean.groupby("Team")["LapTimeSeconds"].min()
    driver_gap = (driver_best - driver_best.min()).to_dict()
    team_gap = (team_best - team_best.min()).to_dict()
    return driver_gap, team_gap


def race_pace_gap_rows(session) -> tuple[dict[str, float], dict[str, float]]:
    laps = valid_laps(session)
    if laps.empty:
        return {}, {}
    clean = laps[is_clean_race_pace_lap(laps)].copy()
    if clean.empty:
        return {}, {}
    driver_pace = clean.groupby("Driver")["LapTimeSeconds"].median()
    team_pace = clean.groupby("Team")["LapTimeSeconds"].median()
    return (driver_pace - driver_pace.min()).to_dict(), (team_pace - team_pace.min()).to_dict()


@st.cache_data(show_spinner=False, ttl=60 * 60 * 12)
def championship_standings(year: int, mode: str, through_round: int) -> pd.DataFrame:
    schedule = load_schedule(year)
    today = pd.Timestamp(dt.date.today())
    schedule = schedule[(schedule["RoundNumber"] <= through_round) & (pd.to_datetime(schedule["EventDate"]) <= today)]
    if schedule.empty:
        return pd.DataFrame()

    driver_stats: dict[str, dict] = {}
    team_stats: dict[str, dict] = {}

    for _, event in schedule.sort_values("RoundNumber").iterrows():
        round_number = int(event["RoundNumber"])

        try:
            race = load_laps_only_session(year, round_number, "R")
            race_results = race.results.copy()
        except Exception:
            continue

        race_driver_gaps, race_team_gaps = race_pace_gap_rows(race)
        if not race_results.empty:
            team_round_positions: dict[str, list[float]] = {}
            for _, row in race_results.iterrows():
                driver = first_present(row, ["Abbreviation", "BroadcastName", "DriverNumber"])
                team = first_present(row, ["TeamName", "Team"])
                headshot = row.get("HeadshotUrl")
                driver_image = str(headshot) if not pd.isna(headshot) and str(headshot).strip() else None
                driver_entry = add_stat_entry(driver_stats, driver, driver, team, driver_image)
                team_entry = add_stat_entry(team_stats, team, team, team, team_identity_image(team))

                points = numeric_value(row.get("Points"))
                position = safe_position(row.get("Position"))
                status = row.get("Status")
                driver_entry["Points"] += points
                driver_entry["RacePositions"].append(position)
                driver_entry["TopStreakPositions"].append(position)
                driver_entry["RacePaceGaps"].append(race_driver_gaps.get(driver, np.nan))
                team_entry["Points"] += points
                team_entry["RacePositions"].append(position)
                team_entry["RacePaceGaps"].append(race_team_gaps.get(team, np.nan))
                team_round_positions.setdefault(team, []).append(position)

                if pd.notna(position) and position <= 3:
                    driver_entry["Podiums"] += 1
                    team_entry["Podiums"] += 1
                if is_retirement_status(status):
                    driver_entry["DNFs"] += 1
                    team_entry["DNFs"] += 1

            for team, positions in team_round_positions.items():
                clean_positions = [position for position in positions if pd.notna(position)]
                if clean_positions and team in team_stats:
                    team_stats[team]["TopStreakPositions"].append(min(clean_positions))

        try:
            quali = load_laps_only_session(year, round_number, "Q")
            quali_results = quali.results.copy()
            quali_driver_gaps, quali_team_gaps = qualifying_gap_rows(quali, mode)
        except Exception:
            quali_results = pd.DataFrame()
            quali_driver_gaps, quali_team_gaps = {}, {}

        if not quali_results.empty:
            for _, row in quali_results.iterrows():
                driver = first_present(row, ["Abbreviation", "BroadcastName", "DriverNumber"])
                team = first_present(row, ["TeamName", "Team"])
                headshot = row.get("HeadshotUrl")
                driver_image = str(headshot) if not pd.isna(headshot) and str(headshot).strip() else None
                driver_entry = add_stat_entry(driver_stats, driver, driver, team, driver_image)
                team_entry = add_stat_entry(team_stats, team, team, team, team_identity_image(team))

                q_position = safe_position(row.get("Position"))
                driver_entry["QualifyingPositions"].append(q_position)
                driver_entry["QualifyingGaps"].append(quali_driver_gaps.get(driver, np.nan))
                team_entry["QualifyingPositions"].append(q_position)
                team_entry["QualifyingGaps"].append(quali_team_gaps.get(team, np.nan))
                if pd.notna(q_position) and q_position == 1:
                    driver_entry["Poles"] += 1
                    team_entry["Poles"] += 1

        try:
            sprint = load_results_only_session(year, round_number, "S")
            sprint_results = sprint.results.copy()
        except Exception:
            sprint_results = pd.DataFrame()

        if not sprint_results.empty:
            for _, row in sprint_results.iterrows():
                driver = first_present(row, ["Abbreviation", "BroadcastName", "DriverNumber"])
                team = first_present(row, ["TeamName", "Team"])
                headshot = row.get("HeadshotUrl")
                driver_image = str(headshot) if not pd.isna(headshot) and str(headshot).strip() else None
                driver_entry = add_stat_entry(driver_stats, driver, driver, team, driver_image)
                team_entry = add_stat_entry(team_stats, team, team, team, team_identity_image(team))
                points = numeric_value(row.get("Points"))
                position = safe_position(row.get("Position"))
                driver_entry["Points"] += points
                team_entry["Points"] += points
                if pd.notna(position) and position == 1:
                    driver_entry["SprintWins"] += 1
                    team_entry["SprintWins"] += 1
                if pd.notna(position) and position <= 3:
                    driver_entry["SprintPodiums"] += 1
                    team_entry["SprintPodiums"] += 1

    source = driver_stats if mode == "Drivers" else team_stats
    if not source:
        return pd.DataFrame()

    leader_points = max(entry["Points"] for entry in source.values())
    rows = []
    for entry in source.values():
        race_positions = [pos for pos in entry["RacePositions"] if pd.notna(pos)]
        quali_positions = [pos for pos in entry["QualifyingPositions"] if pd.notna(pos)]
        row = {
            "Logo": entry.get("Image") or team_identity_image(entry["Team"]),
            "Name": entry["Name"],
            "Team": entry["Team"],
            "Points": round(entry["Points"], 1),
            "Gap to Leader": round(leader_points - entry["Points"], 1),
            "Best Finish": int(min(race_positions)) if race_positions else "-",
            "Worst Finish": int(max(race_positions)) if race_positions else "-",
            "Best Quali": int(min(quali_positions)) if quali_positions else "-",
            "Worst Quali": int(max(quali_positions)) if quali_positions else "-",
            "Top 10 Streak": current_top_streak(entry["TopStreakPositions"], 10),
            "Top 5 Streak": current_top_streak(entry["TopStreakPositions"], 5),
            "Top 3 Streak": current_top_streak(entry["TopStreakPositions"], 3),
            "Poles": entry["Poles"],
            "Podiums": entry["Podiums"],
            "Avg Quali Gap": fmt_gap(average_or_nan(entry["QualifyingGaps"])),
            "Avg Race Pace Gap": fmt_gap(average_or_nan(entry["RacePaceGaps"])),
            "DNFs": entry["DNFs"],
            "Sprint Wins": entry["SprintWins"],
            "Sprint Podiums": entry["SprintPodiums"],
        }
        rows.append(row)

    standings = pd.DataFrame(rows).sort_values(["Points", "Podiums", "Poles"], ascending=[False, False, False])
    standings.insert(0, "Pos", range(1, len(standings) + 1))
    return standings.reset_index(drop=True)


def track_heatmap(telemetry: list[TelemetryLap], track_length: float, bins: int) -> go.Figure:
    if not telemetry:
        return go.Figure()

    edges = np.linspace(0, track_length, bins + 1)
    winners = []
    sample = max(telemetry, key=lambda item: item.frame["Distance"].max()).frame
    center_distances = []
    center_x = []
    center_y = []

    for start, end in zip(edges[:-1], edges[1:]):
        best_driver = None
        best_team = None
        best_speed = -np.inf
        for item in telemetry:
            frame = item.frame
            mask = frame["Distance"].between(start, end)
            if mask.any():
                speed = float(frame.loc[mask, "Speed"].mean())
                if speed > best_speed:
                    best_speed = speed
                    best_driver = item.driver
                    best_team = item.team
        center = (start + end) / 2
        nearest = sample.iloc[(sample["Distance"] - center).abs().argmin()]
        center_distances.append(center)
        center_x.append(float(nearest.get("X", center)))
        center_y.append(float(nearest.get("Y", 0)))
        winners.append({"Driver": best_driver, "Team": best_team, "Speed": best_speed})

    plot = pd.DataFrame(winners)
    plot["Distance"] = center_distances
    plot["X"] = center_x
    plot["Y"] = center_y
    plot["Winner"] = plot["Driver"].fillna("Unknown") + " - " + plot["Team"].fillna("Unknown")
    plot["Speed mph"] = plot["Speed"] * KMH_TO_MPH

    sample_x = sample["X"].to_numpy(dtype=float) if "X" in sample else sample["Distance"].to_numpy(dtype=float)
    sample_y = sample["Y"].to_numpy(dtype=float) if "Y" in sample else np.zeros(len(sample), dtype=float)
    sample_dist = sample["Distance"].to_numpy(dtype=float)
    edge_x = np.interp(edges, sample_dist, sample_x)
    edge_y = np.interp(edges, sample_dist, sample_y)

    team_counts = plot["Team"].dropna().value_counts()
    total_counted = int(team_counts.sum()) or 1
    team_labels = {
        team: f"{team} ({count / total_counted:.1%})"
        for team, count in team_counts.items()
    }
    shown_teams: set[str] = set()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sample_x,
            y=sample_y,
            mode="lines",
            line={"color": "rgba(255,255,255,0.72)", "width": 13},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sample_x,
            y=sample_y,
            mode="lines",
            line={"color": "rgba(11,15,20,0.88)", "width": 8},
            hoverinfo="skip",
            showlegend=False,
        )
    )

    run_start = 0
    for idx in range(1, len(plot) + 1):
        previous_team = plot.iloc[idx - 1]["Team"] if idx > 0 else None
        current_team = plot.iloc[idx]["Team"] if idx < len(plot) else None
        if idx < len(plot) and current_team == previous_team:
            continue

        team = previous_team
        if pd.notna(team):
            run = plot.iloc[run_start:idx]
            team = str(team)
            color = normalize_color(None, team)
            showlegend = team not in shown_teams
            shown_teams.add(team)
            hover_text = [
                (
                    f"{row['Team']}<br>"
                    f"Fastest: {row['Driver']}<br>"
                    f"Distance: {row['Distance']:.0f} m<br>"
                    f"Speed: {row['Speed mph']:.1f} mph"
                )
                for _, row in run.iterrows()
            ]
            fig.add_trace(
                go.Scatter(
                    x=edge_x[run_start : idx + 1],
                    y=edge_y[run_start : idx + 1],
                    mode="lines",
                    line={"color": color, "width": 8},
                    name=team_labels.get(team, team),
                    legendgroup=team,
                    showlegend=showlegend,
                    hoverinfo="text",
                    text=hover_text + [hover_text[-1]],
                )
            )

        run_start = idx

    fig.update_layout(
        height=610,
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
        paper_bgcolor="#0b0f14",
        plot_bgcolor="#0b0f14",
        legend_title_text="Team dominance",
        legend={
            "bgcolor": "rgba(0,0,0,0.35)",
            "font": {"color": "#E5E7EB", "size": 13},
            "itemsizing": "constant",
        },
        xaxis={"visible": False, "scaleanchor": "y"},
        yaxis={"visible": False},
    )
    return fig


def nascar_dashboard() -> None:
    current_year = dt.date.today().year
    with st.container(border=True):
        st.subheader("NASCAR Session Picker")
        picker_a, picker_b, picker_c, picker_d = st.columns([1.2, 1, 2.4, 1.2])
        with picker_a:
            series_name = st.selectbox("Series", list(NASCAR_SERIES.keys()), index=0)
        series_id = NASCAR_SERIES[series_name]
        with picker_b:
            year = st.selectbox("Year", list(range(current_year, 2019, -1)), index=0, key="nascar_year")
        schedule = load_nascar_schedule(year, series_id)
        if schedule.empty:
            st.warning("No NASCAR schedule data was available for this selection.")
            return

        today = pd.Timestamp(dt.date.today(), tz="UTC")
        completed = schedule[schedule["Race_Date_Value"] <= today]
        default_idx = int(completed.index.max()) if not completed.empty else 0
        with picker_c:
            race_label = st.selectbox("Race", schedule["Label"].tolist(), index=default_idx, key="nascar_race")
        race = schedule.loc[schedule["Label"] == race_label].iloc[0]
        race_id = int(race["Race_Id"])
        with picker_d:
            session_choice = st.selectbox("Session", ["Race", "Qualifying"], key="nascar_session")

    st.caption(f"{series_name} | {race['Race_Name']} at {race['Track_Name']} | Race ID {race_id}")

    try:
        weekend = load_nascar_cached_file(year, series_id, race_id, "weekend-feed.json")
    except Exception as exc:
        st.error(f"Could not load NASCAR weekend data: {exc}")
        return

    qualifying = nascar_qualifying_dataframe(weekend, series_id)
    race_results = pd.DataFrame()
    run_data = {}
    try:
        current_results = load_nascar_cached_file(year, series_id, race_id, "current-results.json")
        race_results = nascar_results_dataframe(current_results.get("Results", []), series_id)
        run_data = (current_results.get("RunData") or [{}])[0]
    except Exception:
        current_results = {}

    tab_qpace, tab_rpace, tab_long_run, tab_pits = st.tabs(
        ["Quali Pace", "Race Pace", "Long Run", "Pit Stops"]
    )

    with tab_qpace:
        if qualifying.empty:
            st.info("No qualifying result data was available for this race.")
        else:
            fig = nascar_ranking_graphic(
                qualifying,
                "Qualifying Pace",
                f"{race['Race_Name']} | best qualifying lap",
                "Best Lap",
                lower_is_better=True,
                top_n=12,
                value_suffix="s",
                value_format="{:.3f}",
            )
            if fig is not None:
                st.pyplot(fig, width="stretch")
                plt.close(fig)
            with st.expander("Qualifying raw data"):
                table = qualifying.sort_values("Position").copy()
                display_frame(table[["Position", "Driver", "Number", "Manufacturer", "Best Lap", "Best Lap Speed", "Delta"]])

    with tab_rpace:
        if race_results.empty:
            st.info("No race result data was available for this race.")
        else:
            fig = nascar_ranking_graphic(
                race_results,
                "Race Pace",
                f"{race['Race_Name']} | fastest race lap by driver",
                "Best Lap",
                lower_is_better=True,
                top_n=12,
                value_suffix="s",
                value_format="{:.3f}",
            )
            if fig is not None:
                st.pyplot(fig, width="stretch")
                plt.close(fig)
            with st.expander("Race pace raw data"):
                table = race_results.sort_values("Best Lap").copy()
                display_frame(table[["Driver", "Number", "Team", "Finish", "Start", "Best Lap", "Best Lap Speed", "Average Running Position", "Laps Top 10"]])

    with tab_long_run:
        if race_results.empty:
            st.info("No race result data was available for this race.")
        else:
            long_run = race_results.copy()
            long_run["Long Run Score"] = (
                (long_run["Average Running Position"].rank(ascending=True, method="min") * 0.55)
                + (long_run["Best Lap"].rank(ascending=True, method="min") * 0.25)
                + ((long_run["Laps Top 10"].max() - long_run["Laps Top 10"]).rank(ascending=True, method="min") * 0.20)
            )
            fig = nascar_ranking_graphic(
                long_run,
                "Long Run Pace",
                "Composite from average running position, fastest lap, and laps in top 10",
                "Long Run Score",
                lower_is_better=True,
                top_n=12,
                value_format="{:.1f}",
            )
            if fig is not None:
                st.pyplot(fig, width="stretch")
                plt.close(fig)
            with st.expander("Long run raw data"):
                table = long_run.sort_values("Long Run Score").copy()
                display_frame(table[["Driver", "Number", "Team", "Long Run Score", "Average Running Position", "Best Lap", "Best Lap Speed", "Laps Top 10"]])

    with tab_pits:
        try:
            live_pit_data = load_nascar_cached_file(year, series_id, race_id, "live-pit-data.json")
        except Exception:
            live_pit_data = []

        pitstops = nascar_pitstop_dataframe(live_pit_data, series_id)
        if pitstops.empty:
            st.info("No valid pit stop timing data was available for this race.")
        else:
            st.caption("Source: NASCAR LivePitData cache (`live-pit-data.json`), ranked by documented pit_stop_duration.")
            pit_summary = (
                pitstops.groupby(["Driver", "Number", "Team", "Manufacturer"], as_index=False)
                .agg(**{"Average Box Time": ("Box Time", "mean"), "Stops Counted": ("Box Time", "count")})
            )
            pit_summary["Badge"] = pit_summary["Number"].map(lambda number: nascar_driver_badge_source(series_id, number))
            fig = nascar_ranking_graphic(
                pit_summary,
                "Fastest Pit Crews",
                "Average in-box time from NASCAR LivePitData, excluding invalid and OTHER stops",
                "Average Box Time",
                lower_is_better=True,
                top_n=12,
                value_suffix="s",
                value_format="{:.2f}",
            )
            if fig is not None:
                st.pyplot(fig, width="stretch")
                plt.close(fig)
            with st.expander("Pit stop raw data"):
                display_frame(
                    pitstops.sort_values("Box Time")[
                        [
                            "Driver",
                            "Number",
                            "Manufacturer",
                            "Lap",
                            "Stop Type",
                            "Box Time",
                            "Total Pit Road Time",
                            "Positions Gained/Lost",
                        ]
                    ],
                    height=420,
                )


def indycar_dashboard() -> None:
    seasons = load_indycar_seasons()
    dropdown = load_indycar_season_dropdown()
    current_year = dt.date.today().year

    with st.container(border=True):
        st.subheader("IndyCar Session Picker")
        picker_a, picker_b, picker_c = st.columns([1, 2.4, 1.8])
        with picker_a:
            year = st.selectbox(
                "Year",
                seasons,
                index=seasons.index(current_year) if current_year in seasons else 0,
                key="indycar_year",
            )

        year_entry = next((item for item in dropdown if str(item.get("Year")) == str(year)), None)
        events = (year_entry or {}).get("Events") or []
        if not events:
            st.warning("No IndyCar events were available for this season.")
            return

        with picker_b:
            event_labels = [event["EventName"] for event in events]
            event_label = st.selectbox("Race", event_labels, key="indycar_event")
        event = next(event for event in events if event["EventName"] == event_label)
        sessions = event.get("Sessions") or []
        with picker_c:
            session_label = st.selectbox("Session", [session["SessionName"] for session in sessions], key="indycar_session")
        selected_session = next(session for session in sessions if session["SessionName"] == session_label)

    session = load_indycar_session(selected_session["EventsSessionID"])
    data = indycar_records_dataframe(session)
    st.caption(
        f"NTT INDYCAR SERIES | {session.get('EventName', event_label)} | {session.get('SessionName', session_label)} | "
        f"{session.get('SessionDateFormatted', '')}"
    )

    tab_pace, tab_race, tab_long_run, tab_pit_strategy, tab_standings, tab_reports = st.tabs(
        ["Session Pace", "Race Result", "Long Run", "Pit Strategy", "Standings", "Reports"]
    )

    with tab_pace:
        if data.empty or data["Best Lap"].dropna().empty:
            st.info("No lap-time data was available for this session.")
        else:
            fig = nascar_ranking_graphic(
                data,
                "IndyCar Session Pace",
                "Official best lap from INDYCAR EventsSessionDetails",
                "Best Lap",
                lower_is_better=True,
                top_n=12,
                value_suffix="s",
                value_format="{:.4f}",
            )
            if fig is not None:
                st.pyplot(fig, width="stretch")
                plt.close(fig)
            with st.expander("Session pace raw data"):
                table = data.sort_values("Best Lap").copy()
                display_frame(table[["Driver", "Number", "Team", "Best Lap Text", "Best Speed", "Laps", "Status"]])

    with tab_race:
        if session.get("SessionType") != "R" or data.empty:
            st.info("Select a race session to view race-result graphics.")
        else:
            fig = nascar_ranking_graphic(
                data,
                "IndyCar Race Result",
                "Official finishing order",
                "Finish",
                lower_is_better=True,
                top_n=12,
                value_format="{:.0f}",
            )
            if fig is not None:
                st.pyplot(fig, width="stretch")
                plt.close(fig)
            with st.expander("Race result raw data"):
                table = data.sort_values("Finish").copy()
                display_frame(table[["Finish", "Start", "Driver", "Number", "Team", "Laps", "Laps Led", "Pit Stops", "Average Speed", "Status", "Points"]])

    with tab_long_run:
        if session.get("SessionType") != "R" or data.empty:
            st.info("Select a race session to view long-run strength.")
        else:
            long_run = data.copy()
            long_run["Long Run Score"] = (
                (long_run["Finish"].rank(ascending=True, method="min") * 0.35)
                + (long_run["Average Speed"].rank(ascending=False, method="min") * 0.30)
                + (long_run["Best Lap"].rank(ascending=True, method="min") * 0.20)
                + ((long_run["Laps"].max() - long_run["Laps"]).rank(ascending=True, method="min") * 0.15)
            )
            fig = nascar_ranking_graphic(
                long_run,
                "IndyCar Long Run Strength",
                "Composite from finish, average speed, best lap, and laps completed",
                "Long Run Score",
                lower_is_better=True,
                top_n=12,
                value_format="{:.1f}",
            )
            if fig is not None:
                st.pyplot(fig, width="stretch")
                plt.close(fig)
            with st.expander("Long-run raw data"):
                table = long_run.sort_values("Long Run Score").copy()
                display_frame(table[["Driver", "Number", "Team", "Long Run Score", "Finish", "Average Speed", "Best Lap Text", "Laps", "Laps Led"]])

    with tab_pit_strategy:
        if session.get("SessionType") != "R" or data.empty or data["Pit Stops"].dropna().empty:
            st.info("Select a race session with pit stop counts to view pit strategy.")
        else:
            pit_data = data.dropna(subset=["Pit Stops"]).copy()
            pit_data = pit_data[pit_data["Status"].astype(str).str.casefold().isin(["running", "finished"])].copy()
            if pit_data.empty:
                st.info("No classified finishers with pit stop counts were available for this race.")
            else:
                fig = nascar_ranking_graphic(
                    pit_data,
                    "IndyCar Pit Strategy",
                    "Official pit stop count by classified finishers. IndyCar public JSON does not include stop durations.",
                    "Pit Stops",
                    lower_is_better=True,
                    top_n=12,
                    value_format="{:.0f}",
                )
                if fig is not None:
                    st.pyplot(fig, width="stretch")
                    plt.close(fig)
                with st.expander("Pit strategy raw data"):
                    table = pit_data.sort_values(["Pit Stops", "Finish"]).copy()
                    display_frame(table[["Driver", "Number", "Team", "Finish", "Pit Stops", "Laps", "Status"]])

    with tab_standings:
        payload = load_indycar_standings(year)
        standings = indycar_standings_dataframe(payload)
        if standings.empty:
            st.info("No IndyCar standings data was available.")
        else:
            fig = nascar_ranking_graphic(
                standings,
                "IndyCar Standings",
                str(payload.get("SortTitle") or "Championship points standings"),
                "Points",
                lower_is_better=False,
                top_n=12,
                value_suffix=" pts",
                value_format="{:.0f}",
            )
            if fig is not None:
                st.pyplot(fig, width="stretch")
                plt.close(fig)
            with st.expander("Standings raw data"):
                display_frame(standings[["Pos", "Driver", "Points", "Wins", "Poles", "Top 5s", "Best Finish", "Road Points", "Oval Points"]], height=520)

    with tab_reports:
        reports = pd.DataFrame(session.get("SessionReports") or [])
        if reports.empty:
            st.info("No official report links were available for this session.")
        else:
            reports = reports.copy()
            reports["URL"] = reports["Url"].map(lambda value: f"https://imscdn.com/{value}" if str(value).startswith("INDYCAR/") else value)
            display_frame(reports[["Name", "DocumentType", "FileType", "URL"]], height=420)


def main() -> None:
    with st.sidebar:
        st.header("Motorsport Tools")
        tool = st.radio("Series", ["F1", "NASCAR", "IndyCar"], label_visibility="collapsed")
        st.caption("Pick a series-specific analysis toolkit.")

    if tool == "NASCAR":
        nascar_dashboard()
        return

    if tool == "IndyCar":
        indycar_dashboard()
        return

    if tool != "F1":
        st.subheader(f"{tool} Tools")
        st.info(f"{tool} analysis tools are not wired up yet. Choose F1 in the sidebar to use the current dashboard.")
        return

    current_year = dt.date.today().year
    with st.container(border=True):
        st.subheader("Session Picker")
        picker_left, picker_middle, picker_right = st.columns([1, 2.2, 1.4])
        with picker_left:
            year = st.selectbox("Year", list(range(current_year, 2017, -1)), index=0)
        schedule = load_schedule(year)
        with picker_middle:
            race_label = st.selectbox("Race", schedule["Label"].tolist())
        round_number = int(schedule.loc[schedule["Label"] == race_label, "RoundNumber"].iloc[0])
        with picker_right:
            session_name = st.selectbox("Session", list(SESSION_OPTIONS.keys()), index=5)
        session_code = SESSION_OPTIONS[session_name]

        settings_left, settings_right = st.columns(2)
        with settings_left:
            corner_radius = st.slider("Corner window", 40, 220, 110, 10, help="Meters before and after each mapped corner.")
        with settings_right:
            heatmap_bins = st.slider("Heat map detail", 40, 180, 90, 10, help="Number of track mini-sectors.")

    with st.spinner("Loading timing and telemetry from FastF1..."):
        session = load_session(year, round_number, session_code)
        laps = valid_laps(session)
        telemetry = collect_fastest_lap_telemetry(year, round_number, session_code)

    standings_rounds = {round_number, int(schedule["RoundNumber"].max())}
    standings_cache: dict[tuple[str, int], pd.DataFrame] = {}
    with st.spinner("Preloading championship standings..."):
        for standings_round in standings_rounds:
            for standings_mode in ["Drivers", "Teams"]:
                standings_cache[(standings_mode, standings_round)] = championship_standings(
                    year, standings_mode, standings_round
                )

    if laps.empty or not telemetry:
        st.warning("No usable timing or telemetry was available for this selection.")
        return

    track_length = max(item.frame["Distance"].max() for item in telemetry)
    corners = circuit_corners(session, track_length)
    driver_sections, team_sections = build_section_rankings(telemetry, corners, track_length, corner_radius)
    corner_samples = corner_speed_samples(telemetry, corners, track_length, corner_radius)

    event_name = str(session.event["EventName"])
    st.caption(f"{year} {event_name} - {session_name}")

    fastest = min(telemetry, key=lambda item: item.lap_time)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Fastest driver", fastest.driver, format_time(fastest.lap_time))
    col2.metric("Fastest team", fastest.team)
    col3.metric("Drivers loaded", len(telemetry))
    col4.metric("Track length", f"{track_length / 1000:.2f} km")

    (
        tab_map,
        tab_corners,
        tab_straights,
        tab_efficiency,
        tab_speed,
        tab_theoretical,
        tab_strategy,
        tab_pit_crews,
        tab_delta,
        tab_standings,
    ) = st.tabs(
        [
            "Track Heat Map",
            "Corners",
            "Straights",
            "Efficiency Map",
            "Speed Trap",
            "Theoretical",
            "Race Strategy",
            "Pit Crews",
            "Delta Charts",
            "Standings",
        ]
    )

    with tab_map:
        st.plotly_chart(track_heatmap(telemetry, track_length, heatmap_bins), width="stretch")

    with tab_corners:
        if corner_samples.empty:
            st.info("Corner telemetry was not available for this session.")
        else:
            corner_mode = st.segmented_control("Show", ["Drivers", "Teams"], default="Drivers", key="corner_mode")
            corner_top_n = st.slider("Rankings per graphic", 3, 12, 8, 1, key="corner_top_n")
            corner_headshots = driver_headshot_urls(year, round_number, session_code) if corner_mode == "Drivers" else {}
            for corner_type in ["Slow Speed", "Medium Speed", "High Speed"]:
                driver_data, team_data = corner_type_rankings(corner_samples, corner_type)
                corner_list = (
                    corner_samples.loc[corner_samples["Corner Type"] == corner_type, ["Corner", "Median Minimum Speed"]]
                    .drop_duplicates()
                    .sort_values("Median Minimum Speed")
                )
                if driver_data.empty:
                    st.info(f"No {corner_type.lower()} corners were detected for this track/session.")
                    continue

                label_data = format_speed_columns(corner_list)
                ranking_data = driver_data if corner_mode == "Drivers" else team_data
                graphic = corner_ranking_graphic(
                    ranking_data,
                    corner_type,
                    corner_mode,
                    event_name,
                    session_name,
                    corner_headshots,
                    label_data,
                    int(corner_top_n),
                )
                if graphic is not None:
                    st.pyplot(graphic, width="stretch")
                    plt.close(graphic)

                with st.expander(f"{corner_type} raw ranking data"):
                    if corner_mode == "Drivers":
                        display_frame(
                            format_speed_columns(driver_data)[["Driver", "Team", "Average Speed", "Minimum Speed"]],
                            height=290,
                        )
                    else:
                        display_frame(
                            format_speed_columns(team_data)[["Team", "Average Speed", "Minimum Speed"]],
                            height=290,
                        )

            min_corner = minimum_corner_speed_leaderboard(telemetry, corners, track_length, corner_radius)
            if corner_mode == "Teams":
                team_corner = (
                    corner_samples.groupby(["Corner", "Team"], as_index=False)["Minimum Speed"]
                    .max()
                    .sort_values("Minimum Speed", ascending=False)
                )
                idx = team_corner.groupby("Corner")["Minimum Speed"].idxmax()
                min_corner = team_corner.loc[idx].sort_values("Minimum Speed", ascending=False).reset_index(drop=True)

            leader_graphic = corner_leaders_graphic(
                min_corner,
                corner_mode,
                event_name,
                session_name,
                corner_headshots,
                int(corner_top_n),
            )
            if leader_graphic is not None:
                st.pyplot(leader_graphic, width="stretch")
                plt.close(leader_graphic)

            with st.expander("Apex speed raw ranking data"):
                min_corner_table = min_corner.copy()
                if not min_corner_table.empty:
                    min_corner_table["Minimum Speed"] = min_corner_table["Minimum Speed"].map(format_speed_mph)
                display_frame(min_corner_table, height=300)

    with tab_straights:
        straight_left, straight_right = st.columns([1, 1])
        with straight_left:
            straight_mode = st.segmented_control("Show", ["Drivers", "Teams"], default="Drivers", key="straight_mode")
        with straight_right:
            straight_top_n = st.slider("Rankings per graphic", 3, 12, 10, 1, key="straight_top_n")

        straight_headshots = driver_headshot_urls(year, round_number, session_code) if straight_mode == "Drivers" else {}
        straight_data = driver_sections if straight_mode == "Drivers" else team_sections
        straight_graphic = straight_ranking_graphic(
            straight_data,
            straight_mode,
            event_name,
            session_name,
            straight_headshots,
            int(straight_top_n),
        )
        if straight_graphic is not None:
            st.pyplot(straight_graphic, width="stretch")
            plt.close(straight_graphic)

        with st.expander("Straight-line raw ranking data"):
            if straight_mode == "Drivers":
                data = driver_sections.sort_values("Straight Avg Speed", ascending=False).copy()
                data["Straight Avg Speed"] = data["Straight Avg Speed"].map(format_speed_mph)
                data["Speed Trap"] = data["Speed Trap"].map(format_speed_mph)
                display_frame(data[["Driver", "Team", "Best Lap", "Straight Avg Speed", "Speed Trap"]])
            else:
                data = team_sections.sort_values("Straight Avg Speed", ascending=False).copy()
                data["Straight Avg Speed"] = data["Straight Avg Speed"].map(format_speed_mph)
                data["Speed Trap"] = data["Speed Trap"].map(format_speed_mph)
                display_frame(data[["Team", "Straight Avg Speed", "Speed Trap"]])

    with tab_efficiency:
        efficiency_data = team_efficiency_data(telemetry)
        efficiency_fig = team_efficiency_graphic(efficiency_data, event_name, session_name, year)
        if efficiency_fig is None:
            st.info("Not enough telemetry was available to build the efficiency map.")
        else:
            image = figure_to_png(efficiency_fig)
            image_bytes = image.getvalue()
            st.image(image_bytes, width="stretch")
            st.download_button(
                "Download Efficiency Map PNG",
                data=image_bytes,
                file_name=f"{year}_{event_name}_{session_name}_efficiency_map.png"
                .replace(" ", "_")
                .replace("/", "-")
                .replace("\\", "-")
                .replace(":", "-"),
                mime="image/png",
            )
            plt.close(efficiency_fig)

        with st.expander("Efficiency map raw data"):
            table = efficiency_data.copy()
            if not table.empty:
                table["Lap Time"] = table["Lap Time"].map(format_time)
                table["Mean Speed"] = table["Mean Speed"].map(lambda value: f"{value:.1f} km/h")
                table["Top Speed"] = table["Top Speed"].map(lambda value: f"{value:.1f} km/h")
            display_frame(table[["Team", "Driver", "Lap Time", "Mean Speed", "Top Speed"]])

    with tab_speed:
        speed_top_n = st.slider("Rankings per graphic", 3, 12, 10, 1, key="speed_top_n")
        speed_headshots = driver_headshot_urls(year, round_number, session_code)
        speed_fig = speed_trap_graphic(driver_sections, event_name, session_name, speed_headshots, int(speed_top_n))
        if speed_fig is not None:
            st.pyplot(speed_fig, width="stretch")
            plt.close(speed_fig)

        with st.expander("Speed trap raw ranking data"):
            speed_trap = driver_sections.sort_values("Speed Trap", ascending=False).copy()
            speed_trap["Speed Trap"] = speed_trap["Speed Trap"].map(format_speed_mph)
            display_frame(speed_trap[["Driver", "Team", "Speed Trap"]])

    with tab_theoretical:
        theoretical = theoretical_lap_leaderboard(laps)
        if theoretical.empty:
            st.info("Sector timing was not available for a theoretical lap leaderboard.")
        else:
            theoretical_top_n = st.slider("Rankings per graphic", 3, 12, 10, 1, key="theoretical_top_n")
            theoretical_headshots = driver_headshot_urls(year, round_number, session_code)
            theoretical_fig = theoretical_lap_graphic(
                theoretical,
                event_name,
                session_name,
                theoretical_headshots,
                int(theoretical_top_n),
            )
            if theoretical_fig is not None:
                st.pyplot(theoretical_fig, width="stretch")
                plt.close(theoretical_fig)

            with st.expander("Theoretical lap raw ranking data"):
                display_frame(theoretical[["Driver", "Team", "S1", "S2", "S3", "Best Theoretical"]])

    with tab_strategy:
        if session_code != "R":
            st.info("Race strategy is available for Race sessions.")
        else:
            strategy = f1_strategy_data(laps)
            if strategy.empty:
                st.info("Tyre stint data was not available for this race.")
            else:
                strategy_top_n = st.slider("Drivers per graphic", 5, 24, 20, 1, key="strategy_top_n")
                strategy_fig = f1_strategy_graphic(strategy, event_name, session_name, int(strategy_top_n))
                if strategy_fig is not None:
                    st.pyplot(strategy_fig, width="stretch")
                    plt.close(strategy_fig)
                with st.expander("Race strategy raw data"):
                    display_frame(strategy[["Driver", "Team", "Stint", "Compound", "Start Lap", "End Lap", "Laps"]], height=420)

    with tab_pit_crews:
        if year < 2023:
            st.info("OpenF1 pit stop durations are available from the 2023 season onward.")
        else:
            pit_left, pit_right = st.columns(2)
            with pit_left:
                pit_through_selected = st.toggle("Through Selected Round", value=True, key="pit_crews_through_round")
            with pit_right:
                max_normal_stop = st.slider(
                    "Max normal stop",
                    3.5,
                    10.0,
                    6.0,
                    0.5,
                    key="pit_crews_max_stop",
                    help="Stops above this are treated as abnormal, such as wing changes, penalties, or repairs.",
                )
            through_round = round_number if pit_through_selected else None
            with st.spinner("Loading OpenF1 pit stop durations..."):
                pitstops = f1_season_pitstop_data(year, through_round)
            normal_pitstops = filter_normal_f1_pitstops(pitstops, float(max_normal_stop))
            pit_summary = f1_pit_crews_summary(normal_pitstops)

            if pit_summary.empty:
                st.info("No normal pit stop duration data was available for this season selection.")
            else:
                pit_fig = f1_pit_crews_graphic(pit_summary, year, top_n=10)
                if pit_fig is not None:
                    st.pyplot(pit_fig, width="stretch")
                    plt.close(pit_fig)
                st.caption("Source: OpenF1 `/v1/pit` stop_duration. Filters remove missing stops, very short entries, high-duration stops, and IQR outliers.")
                with st.expander("Pit crew raw stop data"):
                    display_frame(
                        normal_pitstops.sort_values("Stop Duration")[
                            ["Race", "Team", "Driver", "Driver Number", "Lap", "Stop Duration", "Pit Lane Duration"]
                        ],
                        height=420,
                    )
                with st.expander("Pit crew summary data"):
                    display_frame(pit_summary[["Team", "Average Stop", "Median Stop", "Best Stop", "Stops Counted"]], height=360)

    with tab_delta:
        left, middle, right = st.columns(3)
        with left:
            delta_session = st.segmented_control("Pace Type", ["Qualifying", "Race"], default="Qualifying")
        with middle:
            delta_grouping = st.segmented_control("Ranking", ["Drivers", "Teams"], default="Drivers")
        with right:
            top_n = st.number_input("Show Top", min_value=3, max_value=20, value=10, step=1)
        presentation = st.toggle("Presentation size", value=True)

        if st.button("Generate Image", type="primary"):
            with st.spinner("Building image..."):
                headshots = driver_headshot_urls(year, round_number, "Q" if delta_session == "Qualifying" else "R")
                if delta_session == "Qualifying":
                    delta_data, qualifying_session = qualifying_delta_bar_data(year, round_number, delta_grouping, int(top_n))
                    note = "Qualifying uses each driver's or team's fastest lap, plotted as delta to pole."
                else:
                    delta_data = delta_box_data(year, round_number, delta_session, delta_grouping)
                    note = "Race box plots use clean green-flag race laps, ordered by median lap time."

            if delta_data.empty:
                st.info(f"No usable {delta_session.lower()} delta data was available for this race.")
            else:
                title = f"{year} {event_name} {delta_session} Pace Comparison - {delta_grouping}"
                if delta_session == "Qualifying":
                    fig = qualifying_delta_bar_chart(
                        delta_data,
                        qualifying_session,
                        title,
                        delta_grouping,
                        headshots,
                        presentation,
                    )
                    table = delta_data[["Name", "GroupTeam", "Lap Time", "Delta"]].rename(columns={"GroupTeam": "Team"})
                else:
                    fig = delta_box_plot(delta_data, title, delta_grouping, headshots, int(top_n), presentation)
                    visible_names = top_order_from_summary(delta_data, int(top_n))
                    table = (
                        delta_box_summary(delta_data)
                        .query("Name in @visible_names")
                        .rename(columns={"GroupTeam": "Team"})
                    )

                image = figure_to_png(fig)
                image_bytes = image.getvalue()
                st.image(image_bytes, width="stretch")
                st.download_button(
                    "Download PNG",
                    data=image_bytes,
                    file_name=f"{year}_{short_race_name(event_name).lower().replace(' ', '_')}_{delta_session.lower()}_{delta_grouping.lower()}.png",
                    mime="image/png",
                )
                plt.close(fig)
                st.caption(note)
                display_frame(table)

    with tab_standings:
        left, middle, right = st.columns(3)
        with left:
            standings_mode = st.segmented_control("Standings Type", ["Drivers", "Teams"], default="Drivers")
        with middle:
            standings_top_n = st.slider("Rankings per graphic", 3, 20, 10, 1, key="standings_top_n")
        with right:
            through_selected_round = st.toggle("Through Selected Round", value=True)
        standings_round = round_number if through_selected_round else int(schedule["RoundNumber"].max())
        standings = standings_cache.get((standings_mode, standings_round), pd.DataFrame())

        if standings.empty:
            st.info("No championship standings data was available for this selection.")
        else:
            standings_fig = standings_graphic(standings, standings_mode, year, standings_round, int(standings_top_n))
            if standings_fig is not None:
                st.pyplot(standings_fig, width="stretch")
                plt.close(standings_fig)

            with st.expander("Standings raw data"):
                st.dataframe(
                    standings,
                    hide_index=True,
                    width="stretch",
                    height=680,
                    column_config={
                        "Logo": st.column_config.ImageColumn(""),
                        "Points": st.column_config.NumberColumn("Points", format="%.1f"),
                        "Gap to Leader": st.column_config.NumberColumn("Gap to WDC/WCC Leader", format="%.1f"),
                    },
                )


if __name__ == "__main__":
    main()
