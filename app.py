#------------------------------------
#  PLOTTING
#------------------------------------


import matplotlib.pyplot as plt
import pandas as pd


def plot_temperature_series(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        df["time"],
        df["temperature"],
    )

    ax.set_xlabel("Time")
    ax.set_ylabel("Temperature °C")
    ax.set_title("Temperature Time Series")

    return fig

#------------------------------------
#  API
#------------------------------------
import pandas as pd
import requests


def download_cora_temperature_data(url: str) -> pd.DataFrame:
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    return pd.read_csv(io.StringIO(response.text))

#------------------------------------
#  Processing
#------------------------------------

import pandas as pd


def add_rolling_mean(
    df: pd.DataFrame,
    window_size: int = 5,
) -> pd.DataFrame:

    result = df.copy()

    result["temperature_rolling_mean"] = (
        result["temperature"]
        .rolling(window=window_size)
        .mean()
    )

    return result


def add_temperature_summary(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    mean_temperature = (
        result["temperature_rolling_mean"]
        .mean()
        if "temperature_rolling_mean" in result.columns
        else result["temperature"].mean()
    )

    result["temperature_mean"] = mean_temperature

    return result

#------------------------------------
#  Parser
#------------------------------------
from dataclasses import dataclass

import pandas as pd

DEFAULT_LATITUDE = 44.376290
DEFAULT_LONGITUDE = 9.071358


@dataclass
class LoggerMetadata:
    serial: str
    custom_name: str
    sampling_frequency: str
    latitude: float
    longitude: float


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

    clean_df["time"] = pd.to_datetime(
        clean_df["time"],
        errors="coerce"
    )

    clean_df["temperature"] = pd.to_numeric(
        clean_df["temperature"],
        errors="coerce"
    )

    clean_df["serial"] = metadata.serial
    clean_df["custom_name"] = metadata.custom_name
    clean_df["sampling_frequency"] = metadata.sampling_frequency
    clean_df["latitude"] = metadata.latitude
    clean_df["longitude"] = metadata.longitude

    return clean_df.dropna()

#------------------------------------
#  main
#------------------------------------

import pandas as pd
import streamlit as st

from src.parser import parse_envlog_csv
from src.processing import add_rolling_mean
from src.plotting import plot_temperature_series


st.set_page_config(
    page_title="EnvLog Analyzer",
    layout="wide",
)

st.title("EnvLog Analyzer")

uploaded_files = st.file_uploader(
    "Upload CSV files",
    type=["csv"],
    accept_multiple_files=True,
)

window_size = st.sidebar.slider(
    "Rolling window",
    min_value=1,
    max_value=20,
    value=5,
)

if uploaded_files:

    for uploaded_file in uploaded_files:

        raw_df = pd.read_csv(uploaded_file)

        clean_df = parse_envlog_csv(raw_df)

        processed_df = add_rolling_mean(
            clean_df,
            window_size=window_size,
        )

        st.subheader(uploaded_file.name)

        st.dataframe(processed_df.head())

        fig = plot_temperature_series(processed_df)

        st.pyplot(fig)
