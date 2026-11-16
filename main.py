import torch

from models.incident_model import TrafficIncidentModel


model = TrafficIncidentModel()

video = torch.randn(
    1,
    3,
    224,
    224
)

sensors = torch.randn(
    1,
    30,
    6
)


probability, attention_weights = model(
    video,
    sensors
)


print("Incident probability:", probability.item())

print("CCTV attention:", attention_weights[0, 0, 0].item())

print("IoT attention:", attention_weights[0, 1, 0].item())