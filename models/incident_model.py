import torch
import torch.nn as nn

from models.cnn import VisualCNN
from models.lstm import SensorLSTM
from models.attention import AttentionFusion


class TrafficIncidentModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.visual_model = VisualCNN()

        self.sensor_model = SensorLSTM(
            input_size=6,
            hidden_size=96,
            num_layers=2
        )

        self.fusion = AttentionFusion(
            feature_size=128
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, video, sensors):

        visual_features = self.visual_model(video)

        sensor_features = self.sensor_model(sensors)

        fused_features, attention_weights = self.fusion(
            visual_features,
            sensor_features
        )

        output = self.classifier(fused_features)

        probability = torch.sigmoid(output)

        return probability, attention_weights