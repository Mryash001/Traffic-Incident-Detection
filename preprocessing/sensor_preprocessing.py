import pandas as pd
import numpy as np


SENSOR_COLUMNS = [
    "vibration",
    "acoustic",
    "occupancy",
    "ultrasonic",
    "illumination",
    "rainfall"
]


def load_sensor_data(path):

    df = pd.read_csv(path)

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.interpolate()

    return df


def smooth_sensor_data(
    df,
    alpha=0.3
):

    for column in SENSOR_COLUMNS:

        df[column] = df[column].ewm(
            alpha=alpha,
            adjust=False
        ).mean()

    return df


def get_sensor_window(
    df,
    window_size=30
):

    values = df[
        SENSOR_COLUMNS
    ].values

    if len(values) < window_size:

        raise ValueError(
            f"Need {window_size} sensor samples, "
            f"but only {len(values)} available."
        )

    window = values[
        :window_size
    ]

    return window.astype(
        np.float32
    )