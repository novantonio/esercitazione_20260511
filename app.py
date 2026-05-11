"""
EnvLog Analyzer — single-file Streamlit Cloud app
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st


# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_LATITUDE = 44.376290
DEFAULT_LONGITUDE = 9.071358


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


# ── API ───────────────────────────────────────────────────────────────────────

def download_cora_temperature_data(url: str) -> pd.DataFrame:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return pd.read_csv(io.StringIO(response.text))


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_temperature_series(df: pd.DataFrame):
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


# ── Streamlit UI ──────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="EnvLog Analyzer", layout="wide")
    st.title("EnvLog Analyzer")

    # Sidebar
    window_size = st.sidebar.slider(
        "Rolling window",
        min_value=1,
        max_value=20,
        value=5,
    )

    # File uploader
    uploaded_files = st.file_uploader(
        "Upload CSV files",
        type=["csv"],
        accept_multiple_files=True,
    )

    # File count badge
    if uploaded_files:
        n = len(uploaded_files)
        st.info(
            f"**{n} file{'s' if n != 1 else ''} loaded:** "
            + ", ".join(f.name for f in uploaded_files)
        )

        # Process button
        if st.button("▶ Process files", type="primary"):
            for uploaded_file in uploaded_files:
                st.subheader(uploaded_file.name)
                try:
                    raw_df = pd.read_csv(uploaded_file)
                    clean_df = parse_envlog_csv(raw_df)
                    processed_df = add_rolling_mean(clean_df, window_size=window_size)

                    st.dataframe(processed_df.head())

                    fig = plot_temperature_series(processed_df)
                    st.pyplot(fig)

                except Exception as exc:
                    st.error(f"Failed to process **{uploaded_file.name}**: {exc}")
    else:
        st.caption("No files uploaded yet. Use the widget above to select one or more CSV files.")


if __name__ == "__main__":
    main()
