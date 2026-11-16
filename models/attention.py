import torch
import torch.nn as nn


class AttentionFusion(nn.Module):

    def __init__(self, feature_size=128):
        super().__init__()

        self.visual_projection = nn.Linear(
            128,
            feature_size
        )

        self.sensor_projection = nn.Linear(
            96,
            feature_size
        )

        self.attention = nn.Sequential(
            nn.Linear(feature_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(
        self,
        visual_features,
        sensor_features
    ):

        visual = self.visual_projection(
            visual_features
        )

        sensor = self.sensor_projection(
            sensor_features
        )

        features = torch.stack(
            [visual, sensor],
            dim=1
        )

        scores = self.attention(features)

        weights = torch.softmax(
            scores,
            dim=1
        )

        fused = torch.sum(
            weights * features,
            dim=1
        )

        return fused, weights