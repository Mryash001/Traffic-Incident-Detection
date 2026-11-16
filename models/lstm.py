import torch
import torch.nn as nn


class SensorLSTM(nn.Module):

    def __init__(
        self,
        input_size=6,
        hidden_size=96,
        num_layers=2
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

    def forward(self, x):

        output, (hidden, cell) = self.lstm(x)

        return hidden[-1]