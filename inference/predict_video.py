import torch
import numpy as np

from models.cnn import VisualCNN
from preprocessing.video_loader import load_video


video_path = "data/raw/videos/traffic.mp4"


frames, fps = load_video(
    video_path,
    window_seconds=3,
    target_fps=25
)


if len(frames) == 0:
    raise RuntimeError(
        "No frames were extracted from the video."
    )


video_tensor = torch.tensor(
    np.array(frames),
    dtype=torch.float32
)


print(
    "Video tensor:",
    video_tensor.shape
)


model = VisualCNN()

model.eval()


with torch.no_grad():

    features = model(
        video_tensor
    )


print(
    "CNN feature shape:",
    features.shape
)

window_features = torch.mean(
    features,
    dim=0,
    keepdim=True
)


print(
    "3-second window feature:",
    window_features.shape
)