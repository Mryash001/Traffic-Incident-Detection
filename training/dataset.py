import torch
from torch.utils.data import Dataset


class TrafficDataset(Dataset):

    def __init__(
        self,
        video_data,
        sensor_data,
        labels
    ):
        self.video_data = video_data
        self.sensor_data = sensor_data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        video = self.video_data[index].clone().detach().float()

        sensors = self.sensor_data[index].clone().detach().float()

        label = self.labels[index].clone().detach().float()

        return video, sensors, label