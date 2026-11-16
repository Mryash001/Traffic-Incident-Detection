import torch

from preprocessing.sensor_preprocessing import (
    load_sensor_data,
    smooth_sensor_data,
    get_sensor_window
)

from models.lstm import SensorLSTM


sensor_path = (
    "data/raw/sensors/"
    "sensor_data.csv"
)


df = load_sensor_data(
    sensor_path
)

df = smooth_sensor_data(
    df
)

sensor_window = get_sensor_window(
    df,
    window_size=30
)


sensor_tensor = torch.tensor(
    sensor_window,
    dtype=torch.float32
).unsqueeze(0)


print(
    "Sensor tensor:",
    sensor_tensor.shape
)


model = SensorLSTM()

model.eval()


with torch.no_grad():

    features = model(
        sensor_tensor
    )


print(
    "LSTM feature shape:",
    features.shape
)