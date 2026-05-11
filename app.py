"""
EnvLog Analyzer — single-file Streamlit Cloud app
"""

from __future__ import annotations

import io
import warnings
from dataclasses import dataclass

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import requests
import streamlit as st

warnings.filterwarnings("ignore", message="Unverified HTTPS request")


# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_LATITUDE = 44.376290
DEFAULT_LONGITUDE = 9.071358

CORA_URL_TEMPLATE = (
    "https://erddap.emodnet-physics.eu/erddap/griddap/"
    "INSITU_GLO_PHY_TS_OA_MY_013_052_TEMP.csv"
    "?TEMP%5B(1990-01-01T00:00:00Z):1:(2023-06-15T00:00:00Z)%5D"
    "%5B(1.0):1:(1)%5D"
    "%5B({lat}):1:({lat})%5D"
    "%5B({lon}):1:({lon})%5D"
)

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class LoggerMetadata:
    serial: str
    custom_name: str
    sampling_frequency: str
    latitude: float
    longitude: float


# ── Parser ────────────────────────────────────────────────────────────────────

def extract_metadata(df: pd.DataFrame) -> LoggerMetadata:
    serial = df.iloc[9, 1]
    custom_name = df.iloc[10, 1]
    sampling_frequency = df.iloc[13, 1]

    has_latitude = "lat" in str(df.iloc[15, 0]).lower()

    if has_latitude:
        latitude = df.iloc[15, 1]
        longitude = df.iloc[16, 1]
    else:
        latitude = df.iloc[16, 1]
        longitude = df.iloc[17, 1]

    latitude = pd.to_numeric(latitude, errors="coerce")
    longitude = pd.to_numeric(longitude, errors="coerce")

    if pd.isna(latitude) or pd.isna(longitude):
        latitude = DEFAULT_LATITUDE
        longitude = DEFAULT_LONGITUDE

    return LoggerMetadata(
        serial=serial,
        custom_name=custom_name,
        sampling_frequency=sampling_frequency,
        latitude=latitude,
        longitude=longitude,
    )


def parse_envlog_csv(df: pd.DataFrame) -> pd.DataFrame:
    metadata = extract_metadata(df)

    clean_df = (
        df.iloc[21:, :]
        .dropna()
        .reset_index(drop=True)
    )

    clean_df.columns = ["time", "temperature"]

    clean_df["time"] = pd.to_datetime(clean_df["time"], errors="coerce")
    clean_df["temperature"] = pd.to_numeric(clean_df["temperature"], errors="coerce")

    clean_df["serial"] = metadata.serial
    clean_df["custom_name"] = metadata.custom_name
    clean_df["sampling_frequency"] = metadata.sampling_frequency
    clean_df["latitude"] = metadata.latitude
    clean_df["longitude"] = metadata.longitude

    return clean_df.dropna()


# ── Processing ────────────────────────────────────────────────────────────────

def add_rolling_mean(df: pd.DataFrame, window_size: int = 5) -> pd.DataFrame:
    result = df.copy()
    result["temperature_rolling_mean"] = (
        result["temperature"].rolling(window=window_size).mean()
    )
    return result


def add_temperature_summary(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    mean_temperature = (
        result["temperature_rolling_mean"].mean()
        if "temperature_rolling_mean" in result.columns
        else result["temperature"].mean()
    )
    result["temperature_mean"] = mean_temperature
    return result


# ── CORA API ──────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Downloading CORA climatology…")
def fetch_cora_data(latitude: float, longitude: float) -> pd.DataFrame | None:
    url = CORA_URL_TEMPLATE.format(lat=latitude, lon=longitude)
    try:
        response = requests.get(url, verify=False, timeout=60)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text), skiprows=[1])
        df["time"] = pd.to_datetime(df["time"])
        df["TEMP"] = pd.to_numeric(df["TEMP"], errors="coerce")
        return df.dropna()
    except Exception as exc:
        st.warning(f"Could not fetch CORA data: {exc}")
        return None


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_temperature_series(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        df["time"],
        df["temperature"],
        alpha=0.4,
        linewidth=0.8,
        color="steelblue",
        label="Raw temperature",
    )

    if "temperature_rolling_mean" in df.columns:
        ax.plot(
            df["time"],
            df["temperature_rolling_mean"],
            linewidth=2,
            color="tomato",
            label="Rolling mean",
        )
        ax.legend()

    ax.set_xlabel("Time")
    ax.set_ylabel("Temperature °C")
    ax.set_title("Temperature Time Series")
    fig.tight_layout()
    return fig


def plot_doy_climatology(
    cora_df: pd.DataFrame,
    logger_dfs: dict[str, pd.DataFrame],
    latitude: float,
    longitude: float,
) -> plt.Figure:
    """
    Scatter of CORA daily temperatures by day-of-year (one colour per year)
    overlaid with a star marker for each logger's mean temperature.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    years = sorted(cora_df["time"].dt.year.unique())
    colours = cm.tab20(np.linspace(0, 1, len(years)))

    for colour, (year, year_data) in zip(colours, cora_df.groupby(cora_df["time"].dt.year)):
        doy = year_data["time"].dt.dayofyear
        ax.scatter(doy, year_data["TEMP"], label=str(year),
                   marker=".", s=10, color=colour, alpha=0.6)

    # Logger means — one star per file
    star_colours = cm.Set1(np.linspace(0, 1, max(len(logger_dfs), 1)))
    for (fname, sdata), sc in zip(logger_dfs.items(), star_colours):
        d = sdata["time"].iloc[0].timetuple().tm_yday
        tavg = sdata["temperature"].mean()
        ax.plot(d, tavg, "*", markersize=15, color=sc,
                label=fname, markeredgecolor="black", markeredgewidth=0.5)

    ax.set_xlabel("Day of Year")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title(
        f"Temperature vs. Day of Year at ({latitude:.4f}, {longitude:.4f}) by Year"
    )
    ax.legend(title="Year / Logger", bbox_to_anchor=(1.05, 1), loc="upper left",
              fontsize=7, markerscale=1.2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_monthly_climatology(
    cora_df: pd.DataFrame,
    logger_dfs: dict[str, pd.DataFrame],
) -> plt.Figure:
    """
    Monthly mean ± std of CORA data with logger means overlaid as stars.
    """
    cora_df = cora_df.copy()
    cora_df["month"] = cora_df["time"].dt.month
    monthly = (
        cora_df.groupby("month")["TEMP"]
        .agg(["mean", "std"])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.scatter(monthly["month"], monthly["mean"],
               color="steelblue", zorder=3, label="CORA monthly mean")
    ax.errorbar(monthly["month"], monthly["mean"], yerr=monthly["std"],
                fmt="o", color="steelblue", capsize=4, alpha=0.6,
                label="CORA monthly std")

    # Logger means — one star per file
    star_colours = cm.Set1(np.linspace(0, 1, max(len(logger_dfs), 1)))
    for (fname, sdata), sc in zip(logger_dfs.items(), star_colours):
        month = sdata["time"].iloc[0].month
        tavg = sdata["temperature"].mean()
        ax.plot(month, tavg, "*", markersize=15, color=sc,
                label=fname, markeredgecolor="black", markeredgewidth=0.5)

    ax.set_xlabel("Month")
    ax.set_ylabel("Temperature [°C]")
    ax.set_title("Monthly Mean and Standard Deviation of CORA Temperature")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTH_LABELS)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ── Streamlit UI ──────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="EnvLog Analyzer", layout="wide")
    st.title("EnvLog Analyzer")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    window_size = st.sidebar.slider(
        "Rolling window", min_value=1, max_value=20, value=5
    )

    # ── File uploader ─────────────────────────────────────────────────────────
    uploaded_files = st.file_uploader(
        "Upload CSV files",
        type=["csv"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.caption("No files uploaded yet. Use the widget above to select one or more CSV files.")
        return

    # File count badge
    n = len(uploaded_files)
    st.info(
        f"**{n} file{'s' if n != 1 else ''} loaded:** "
        + ", ".join(f.name for f in uploaded_files)
    )

    if not st.button("▶ Process files", type="primary"):
        return

    # ── Parse all files ───────────────────────────────────────────────────────
    logger_dfs: dict[str, pd.DataFrame] = {}

    for uploaded_file in uploaded_files:
        st.subheader(uploaded_file.name)
        try:
            raw_df = pd.read_csv(uploaded_file)
            clean_df = parse_envlog_csv(raw_df)
            processed_df = add_rolling_mean(clean_df, window_size=window_size)
            logger_dfs[uploaded_file.name] = processed_df

            st.dataframe(processed_df.head())

            fig = plot_temperature_series(processed_df)
            st.pyplot(fig)
            plt.close(fig)

        except Exception as exc:
            st.error(f"Failed to process **{uploaded_file.name}**: {exc}")

    if not logger_dfs:
        return

    # ── Infer location from first valid file ──────────────────────────────────
    first_df = next(iter(logger_dfs.values()))
    latitude = float(first_df["latitude"].iloc[0])
    longitude = float(first_df["longitude"].iloc[0])

    # ── CORA climatology ──────────────────────────────────────────────────────
    st.divider()
    st.header("Climatology — CORA comparison")

    cora_df = fetch_cora_data(latitude, longitude)

    if cora_df is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("CORA records", f"{len(cora_df):,}")
        with col2:
            st.metric(
                "Period",
                f"{cora_df['time'].dt.year.min()} – {cora_df['time'].dt.year.max()}",
            )

        st.subheader("Temperature vs. Day of Year")
        fig_doy = plot_doy_climatology(cora_df, logger_dfs, latitude, longitude)
        st.pyplot(fig_doy)
        plt.close(fig_doy)

        st.subheader("Monthly Mean ± Std")
        fig_monthly = plot_monthly_climatology(cora_df, logger_dfs)
        st.pyplot(fig_monthly)
        plt.close(fig_monthly)

    else:
        st.warning(
            "CORA data could not be fetched. "
            "Check your internet connection or try again later."
        )


if __name__ == "__main__":
    main()
