
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from models.incident_model import TrafficIncidentModel

from preprocessing.video_loader import load_video

from preprocessing.sensor_preprocessing import (
    load_sensor_data,
    smooth_sensor_data,
    get_sensor_window
)


app = FastAPI(
    title="Traffic Incident Detection API",
    description="Multimodal CCTV and IoT Traffic Incident Detection System",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


BASE_DIR = Path(__file__).resolve().parent.parent

UPLOAD_DIR = BASE_DIR / "data" / "uploads"

SENSOR_PATH = (
    BASE_DIR
    / "data"
    / "raw"
    / "sensors"
    / "sensor_data.csv"
)

CHECKPOINT_PATH = (
    BASE_DIR
    / "models"
    / "traffic_incident_model.pth"
)

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm"
}

MAX_UPLOAD_SIZE = 100 * 1024 * 1024

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def load_trained_model():

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            "Model checkpoint not found: "
            + str(CHECKPOINT_PATH)
        )

    model = TrafficIncidentModel().to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    if "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    elif "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    model.load_state_dict(checkpoint)

    model.eval()

    return model


def run_inference(video_path):

    if not SENSOR_PATH.exists():
        raise FileNotFoundError(
            "Sensor data CSV file not found."
        )

    model = load_trained_model()

    frames, fps = load_video(
        str(video_path),
        window_seconds=3,
        target_fps=25
    )

    if len(frames) == 0:
        raise RuntimeError(
            "No video frames could be extracted."
        )

    video = torch.tensor(
        np.asarray(frames),
        dtype=torch.float32
    ).to(DEVICE)

    with torch.no_grad():

        frame_features = model.visual_model(
            video
        )

        video_features = torch.mean(
            frame_features,
            dim=0,
            keepdim=True
        )

        df = load_sensor_data(
            str(SENSOR_PATH)
        )

        df = smooth_sensor_data(df)

        sensor_window = get_sensor_window(
            df,
            window_size=30
        )

        sensors = torch.tensor(
            sensor_window,
            dtype=torch.float32
        ).unsqueeze(0).to(DEVICE)

        sensor_features = model.sensor_model(
            sensors
        )

        fused_features, attention_weights = model.fusion(
            video_features,
            sensor_features
        )

        output = model.classifier(
            fused_features
        )

        probability = torch.sigmoid(
            output
        )

    incident_probability = probability.item()

    return {
        "video_fps": float(fps),
        "frames_processed": len(frames),

        "visual_feature_dimensions": list(
            video_features.shape
        ),

        "sensor_feature_dimensions": list(
            sensor_features.shape
        ),

        "fused_feature_dimensions": list(
            fused_features.shape
        ),

        "cctv_attention": round(
            attention_weights[0, 0, 0].item(),
            4
        ),

        "iot_attention": round(
            attention_weights[0, 1, 0].item(),
            4
        ),

        "incident_probability": round(
            incident_probability,
            4
        )
    }


@app.get("/")
def root():

    return {
        "message": "Traffic Incident Detection API is running",
        "status": "online",
        "device": str(DEVICE)
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "device": str(DEVICE),
        "checkpoint_exists": CHECKPOINT_PATH.exists(),
        "sensor_data_exists": SENSOR_PATH.exists()
    }


@app.post("/analyze-video")
async def analyze_video(
    file: UploadFile = File(...)
):

    extension = Path(
        file.filename or ""
    ).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:

        raise HTTPException(
            status_code=400,
            detail="Unsupported video format."
        )

    video_id = str(uuid4())

    destination = (
        UPLOAD_DIR
        / f"{video_id}{extension}"
    )

    size = 0

    try:

        with destination.open("wb") as output_file:

            while True:

                chunk = await file.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                size += len(chunk)

                if size > MAX_UPLOAD_SIZE:

                    raise HTTPException(
                        status_code=413,
                        detail="Video exceeds the 100 MB upload limit."
                    )

                output_file.write(chunk)

        await file.close()

        results = run_inference(
            destination
        )

        return {
            "video_id": video_id,
            "filename": file.filename,
            "size_bytes": size,
            "upload_status": "successful",
            "analysis_status": "completed",
            "model_status": "checkpoint_loaded",
            "validation_note": (
                "The saved model checkpoint was loaded successfully. "
                "Its predictions are not independently validated "
                "on real-world traffic incident data."
            ),
            "results": results
        }

    except HTTPException:

        destination.unlink(missing_ok=True)
        raise

    except Exception as error:

        destination.unlink(missing_ok=True)

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )

    finally:

        await file.close()
