import torch

from torch.utils.data import DataLoader

from training.dataset import TrafficDataset


video_data = torch.randn(
    20,
    3,
    224,
    224
)

sensor_data = torch.randn(
    20,
    30,
    6
)

labels = torch.randint(
    0,
    2,
    (20,)
)


dataset = TrafficDataset(
    video_data,
    sensor_data,
    labels
)


loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True
)


for video, sensors, labels in loader:

    print("Video:", video.shape)
    print("Sensors:", sensors.shape)
    print("Labels:", labels.shape)

    break