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
import plotly.express as px
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
                    "S1": format_time(sectors[0]),
                    "S2": format_time(sectors[1]),
                    "S3": format_time(sectors[2]),
                    "Best Theoretical": sum(sectors),
                }
            )
    result = pd.DataFrame(rows).sort_values("Best Theoretical").reset_index(drop=True)
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
            "Retirements": 0,
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
                    driver_entry["Retirements"] += 1
                    team_entry["DNFs"] += 1
                    team_entry["Retirements"] += 1

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
            "Retirements": entry["Retirements"],
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
    sample = telemetry[0].frame
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
    plot["Winner"] = plot["Driver"] + " - " + plot["Team"]
    plot["Speed mph"] = plot["Speed"] * KMH_TO_MPH

    palette = {}
    for item in telemetry:
        team_color = normalize_color(None, item.team)
        palette[f"{item.driver} - {item.team}"] = team_color

    fig = px.scatter(
        plot,
        x="X",
        y="Y",
        color="Winner",
        color_discrete_map=palette,
        hover_data={"Distance": ":.0f", "Speed mph": ":.1f", "Speed": False, "X": False, "Y": False},
    )
    fig.update_traces(marker={"size": 9, "opacity": 0.95})
    fig.add_trace(
        go.Scatter(
            x=sample["X"] if "X" in sample else sample["Distance"],
            y=sample["Y"] if "Y" in sample else np.zeros(len(sample)),
            mode="lines",
            line={"color": "rgba(255,255,255,0.22)", "width": 2},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.update_layout(
        height=610,
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
        paper_bgcolor="#0b0f14",
        plot_bgcolor="#0b0f14",
        legend_title_text="Fastest mini-sector",
        xaxis={"visible": False, "scaleanchor": "y"},
        yaxis={"visible": False},
    )
    return fig


def main() -> None:
    st.title("F1 Session Intelligence")

    current_year = dt.date.today().year
    with st.sidebar:
        st.header("Session")
        year = st.selectbox("Year", list(range(current_year, 2017, -1)), index=0)
        schedule = load_schedule(year)
        race_label = st.selectbox("Race", schedule["Label"].tolist())
        round_number = int(schedule.loc[schedule["Label"] == race_label, "RoundNumber"].iloc[0])
        session_name = st.selectbox("Session", list(SESSION_OPTIONS.keys()), index=5)
        session_code = SESSION_OPTIONS[session_name]
        corner_radius = st.slider("Corner window", 40, 220, 110, 10, help="Meters before and after each mapped corner.")
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

    tab_map, tab_corners, tab_straights, tab_speed, tab_theoretical, tab_delta, tab_standings = st.tabs(
        ["Track Heat Map", "Corners", "Straights", "Speed Trap", "Theoretical", "Delta Charts", "Standings"]
    )

    with tab_map:
        st.plotly_chart(track_heatmap(telemetry, track_length, heatmap_bins), width="stretch")

    with tab_corners:
        if corner_samples.empty:
            st.info("Corner telemetry was not available for this session.")
        else:
            for corner_type in ["Slow Speed", "Medium Speed", "High Speed"]:
                st.subheader(corner_type)
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
                st.caption("Corners: " + ", ".join(label_data["Corner"].astype(str).tolist()))

                left, right = st.columns(2)
                with left:
                    display_frame(
                        format_speed_columns(driver_data)[["Driver", "Team", "Average Speed", "Minimum Speed"]],
                        height=290,
                    )
                with right:
                    display_frame(
                        format_speed_columns(team_data)[["Team", "Average Speed", "Minimum Speed"]],
                        height=290,
                    )

        st.subheader("Minimum Speed Corner Leaders")
        min_corner = minimum_corner_speed_leaderboard(telemetry, corners, track_length, corner_radius)
        if not min_corner.empty:
            min_corner["Minimum Speed"] = min_corner["Minimum Speed"].map(format_speed_mph)
        display_frame(min_corner, height=300)

    with tab_straights:
        left, right = st.columns(2)
        with left:
            st.subheader("Driver Straight Ranking")
            data = driver_sections.sort_values("Straight Avg Speed", ascending=False).copy()
            data["Straight Avg Speed"] = data["Straight Avg Speed"].map(format_speed_mph)
            display_frame(data[["Driver", "Team", "Best Lap", "Straight Avg Speed"]])
        with right:
            st.subheader("Team Straight Ranking")
            data = team_sections.sort_values("Straight Avg Speed", ascending=False).copy()
            data["Straight Avg Speed"] = data["Straight Avg Speed"].map(format_speed_mph)
            display_frame(data[["Team", "Straight Avg Speed"]])

    with tab_speed:
        speed_trap = driver_sections.sort_values("Speed Trap", ascending=False).copy()
        speed_trap["Speed Trap"] = speed_trap["Speed Trap"].map(format_speed_mph)
        display_frame(speed_trap[["Driver", "Team", "Speed Trap"]])

    with tab_theoretical:
        theoretical = theoretical_lap_leaderboard(laps)
        if theoretical.empty:
            st.info("Sector timing was not available for a theoretical lap leaderboard.")
        else:
            display_frame(theoretical)

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
        left, right = st.columns(2)
        with left:
            standings_mode = st.segmented_control("Standings Type", ["Drivers", "Teams"], default="Drivers")
        with right:
            through_selected_round = st.toggle("Through Selected Round", value=True)
        standings_round = round_number if through_selected_round else int(schedule["RoundNumber"].max())
        standings = standings_cache.get((standings_mode, standings_round), pd.DataFrame())

        if standings.empty:
            st.info("No championship standings data was available for this selection.")
        else:
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
