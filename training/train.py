import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from models.incident_model import TrafficIncidentModel
from training.dataset import TrafficDataset


def create_dataset():

    video_data = torch.randn(
        100,
        3,
        224,
        224
    )

    sensor_data = torch.randn(
        100,
        30,
        6
    )

    labels = torch.randint(
        0,
        2,
        (100,)
    ).float()

    return TrafficDataset(
        video_data,
        sensor_data,
        labels
    )


def train():

    print("Creating dataset...")

    dataset = create_dataset()

    print("Dataset size:", len(dataset))

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True
    )

    model = TrafficIncidentModel()

    criterion = nn.BCELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    epochs = 5

    print("Starting training...")

    for epoch in range(epochs):

        model.train()

        total_loss = 0

        for video, sensors, labels in loader:

            optimizer.zero_grad()

            predictions, attention = model(
                video,
                sensors
            )

            predictions = predictions.squeeze()

            loss = criterion(
                predictions,
                labels
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(loader)

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"Loss: {average_loss:.4f}"
        )

    torch.save(
        model.state_dict(),
        "models/traffic_incident_model.pth"
    )

    print("Model saved successfully.")


if __name__ == "__main__":
    train()