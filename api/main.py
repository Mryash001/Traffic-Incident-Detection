
import os
import uuid
from pathlib import Path

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


BASE_DIR = Path(__file__).resolve().parent.parent

UPLOAD_DIR = BASE_DIR / "data" / "uploads"
SENSOR_PATH = BASE_DIR / "data" / "raw" / "sensors" / "sensor_data.csv"
CHECKPOINT_PATH = BASE_DIR / "models" / "traffic_incident_model.pth"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MAX_UPLOAD_SIZE = 100 * 1024 * 1024

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FRONTEND_URL = os.getenv(
    "FRONTEND_URL",
    "http://localhost:5173"
)

app = FastAPI(
    title="Traffic Incident Detection API",
    description="Multi-modal traffic incident detection using CCTV video and IoT roadside sensors",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list({
        "http://localhost:5173",
        FRONTEND_URL
    }),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


def load_trained_model():

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {CHECKPOINT_PATH}"
        )

    model = TrafficIncidentModel().to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    if isinstance(checkpoint, dict):
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
            f"Sensor data not found: {SENSOR_PATH}"
        )

    model = load_trained_model()

    frames, fps = load_video(
        str(video_path),
        window_seconds=3,
        target_fps=25
    )

    if frames is None or len(frames) == 0:
        raise ValueError("No video frames could be extracted")

    video_tensor = torch.tensor(
        np.asarray(frames),
        dtype=torch.float32
    ).to(DEVICE)

    with torch.no_grad():

        visual_features = model.visual_model(video_tensor)

        if visual_features.ndim == 3:
            visual_features = visual_features.mean(
                dim=1,
                keepdim=False
            ).unsqueeze(0)

        elif visual_features.ndim == 2:
            visual_features = visual_features.mean(
                dim=0,
                keepdim=True
            )

        sensor_df = load_sensor_data(
            str(SENSOR_PATH)
        )

        sensor_df = smooth_sensor_data(sensor_df)

        sensor_window = get_sensor_window(
            sensor_df,
            window_size=30
        )

        sensor_tensor = torch.tensor(
            sensor_window,
            dtype=torch.float32
        ).unsqueeze(0).to(DEVICE)

        sensor_features = model.sensor_model(
            sensor_tensor
        )

        fused_features, attention_weights = model.fusion(
            visual_features,
            sensor_features
        )

        output = model.classifier(fused_features)

        probability = torch.sigmoid(output).item()

    return {
        "video_fps": float(fps),
        "frames_processed": int(len(frames)),
        "visual_feature_dimensions": list(visual_features.shape),
        "sensor_feature_dimensions": list(sensor_features.shape),
        "fused_feature_dimensions": list(fused_features.shape),
        "cctv_attention": round(
            float(attention_weights[0, 0, 0].item()),
            4
        ),
        "iot_attention": round(
            float(attention_weights[0, 1, 0].item()),
            4
        ),
        "incident_probability": round(
            float(probability),
            4
        )
    }


@app.get("/")
def home():

    return {
        "message": "Traffic Incident Detection API is running",
        "status": "online"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "device": str(DEVICE),
        "model_checkpoint_exists": CHECKPOINT_PATH.exists(),
        "sensor_data_exists": SENSOR_PATH.exists()
    }


@app.post("/analyze-video")
async def analyze_video(file: UploadFile = File(...)):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename provided"
        )

    extension = Path(file.filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported video format"
        )

    video_id = str(uuid.uuid4())
    video_path = UPLOAD_DIR / f"{video_id}{extension}"

    try:

        contents = await file.read()

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="Uploaded video is empty"
            )

        if len(contents) > MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=413,
                detail="Video exceeds the maximum allowed size of 100 MB"
            )

        with open(video_path, "wb") as buffer:
            buffer.write(contents)

        results = run_inference(video_path)

        return {
            "video_id": video_id,
            "filename": file.filename,
            "size_bytes": len(contents),
            "upload_status": "successful",
            "analysis_status": "completed",
            "model_status": "checkpoint_loaded",
            "validation_note": (
                "The saved model checkpoint was loaded successfully. "
                "Its predictions are not independently validated "
                "on real-world traffic data."
            ),
            "results": results
        }

    except HTTPException:
        raise

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Video analysis failed: {str(error)}"
        )

    finally:

        await file.close()

        if video_path.exists():
            video_path.unlink()
