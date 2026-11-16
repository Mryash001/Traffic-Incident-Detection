import numpy as np
import torch

from models.incident_model import TrafficIncidentModel

from preprocessing.video_loader import load_video

from preprocessing.sensor_preprocessing import (
    load_sensor_data,
    smooth_sensor_data,
    get_sensor_window
)


VIDEO_PATH = "data/raw/videos/traffic.mp4"

SENSOR_PATH = "data/raw/sensors/sensor_data.csv"


def extract_video_features(model):

    frames, fps = load_video(
        VIDEO_PATH,
        window_seconds=3,
        target_fps=25
    )

    if len(frames) == 0:
        raise RuntimeError(
            "No video frames found."
        )

    video = torch.tensor(
        np.array(frames),
        dtype=torch.float32
    )

    with torch.no_grad():

        frame_features = model.visual_model(
            video
        )

    window_features = torch.mean(
        frame_features,
        dim=0,
        keepdim=True
    )

    return window_features, fps, len(frames)


def extract_sensor_features(model):

    df = load_sensor_data(
        SENSOR_PATH
    )

    df = smooth_sensor_data(
        df
    )

    sensor_window = get_sensor_window(
        df,
        window_size=30
    )

    sensors = torch.tensor(
        sensor_window,
        dtype=torch.float32
    ).unsqueeze(0)

    with torch.no_grad():

        sensor_features = model.sensor_model(
            sensors
        )

    return sensor_features


def main():

    model = TrafficIncidentModel()

    model.eval()

    video_features, fps, frame_count = (
        extract_video_features(model)
    )

    sensor_features = extract_sensor_features(
        model
    )

    with torch.no_grad():

        fused_features, attention_weights = (
            model.fusion(
                video_features,
                sensor_features
            )
        )

        output = model.classifier(
            fused_features
        )

        probability = torch.sigmoid(
            output
        )

    print(
        "Original FPS:",
        fps
    )

    print(
        "Video frames:",
        frame_count
    )

    print(
        "Visual features:",
        video_features.shape
    )

    print(
        "Sensor features:",
        sensor_features.shape
    )

    print(
        "Fused features:",
        fused_features.shape
    )

    print(
        "CCTV attention:",
        round(
            attention_weights[0, 0, 0].item(),
            4
        )
    )

    print(
        "IoT attention:",
        round(
            attention_weights[0, 1, 0].item(),
            4
        )
    )

    print(
        "Incident probability:",
        round(
            probability.item(),
            4
        )
    )


if __name__ == "__main__":
    main()