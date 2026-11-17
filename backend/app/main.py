from pathlib import Path
import io
import os
import uuid

import numpy as np
from PIL import Image, ImageDraw, UnidentifiedImageError
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .model import predict, CLASS_NAMES, MODEL_SIZE, DEVICE
from .services import vessel_candidates, drift_track

BASE = Path(__file__).resolve().parent.parent
OUTPUTS = BASE / "outputs"
OUTPUTS.mkdir(exist_ok=True)

MAX_UPLOAD_BYTES = 30 * 1024 * 1024
DEFAULT_ORIGIN = {"lat": 10.50, "lon": 72.50}
allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]

app = FastAPI(title="OILTRACE API", version="1.0.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
app.mount("/media", StaticFiles(directory=str(OUTPUTS)), name="media")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "model": "POSEatSea / best_sar_model.pth",
    }


def create_visuals(original: Image.Image, labels: np.ndarray, job_id: str):
    base = original.convert("RGB").resize(
        (MODEL_SIZE, MODEL_SIZE), Image.Resampling.BILINEAR
    )
    mask = (labels == 1).astype(np.uint8) * 255
    mask_img = Image.fromarray(mask, mode="L")

    overlay = base.convert("RGBA")
    red = Image.new("RGBA", overlay.size, (255, 70, 55, 0))
    red.putalpha(Image.fromarray((mask * 0.58).astype(np.uint8), mode="L"))
    overlay = Image.alpha_composite(overlay, red).convert("RGB")

    ys, xs = np.where(mask > 0)
    if len(xs):
        ImageDraw.Draw(overlay).rectangle(
            [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
            outline=(255, 255, 0),
            width=3,
        )

    mask_path = OUTPUTS / f"{job_id}_mask.png"
    overlay_path = OUTPUTS / f"{job_id}_overlay.jpg"
    mask_img.save(mask_path)
    overlay.save(overlay_path, quality=92)
    return mask_path, overlay_path


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if not raw:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image is larger than 30 MB.")

    try:
        with Image.open(io.BytesIO(raw)) as source:
            source.verify()
        with Image.open(io.BytesIO(raw)) as source:
            image = source.convert("RGB")
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError):
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid, supported image.")

    job_id = uuid.uuid4().hex[:10]
    try:
        labels, oil_probs, oil_mask = predict(image)
    except Exception:
        # Avoid returning internal paths or implementation details to clients.
        raise HTTPException(
            status_code=500,
            detail="Model inference failed. Check the model checkpoint, architecture and backend logs.",
        )

    mask_path, overlay_path = create_visuals(image, labels, job_id)
    total_pixels = int(oil_mask.size)
    oil_pixels = int(oil_mask.sum())
    oil_ratio = oil_pixels / max(total_pixels, 1)

    class_distribution = {
        name: round(float((labels == idx).mean() * 100), 3)
        for idx, name in enumerate(CLASS_NAMES)
    }
    # oil_probs is a 2-D probability map; mean the full map when no oil pixels
    # are predicted instead of accidentally averaging only one image row.
    confidence = float(oil_probs[oil_mask].mean()) if oil_pixels else float(oil_probs.mean())

    # No square-kilometre estimate is returned: uploaded images have no
    # guaranteed geographic pixel scale or georeferencing metadata.
    origin = DEFAULT_ORIGIN
    return {
        "job_id": job_id,
        "model": "POSEatSea best_sar_model.pth",
        "device": str(DEVICE),
        "input_size": [MODEL_SIZE, MODEL_SIZE],
        "oil_detected": bool(oil_ratio > 0.005),
        "oil_confidence": round(confidence * 100, 2),
        "oil_pixel_ratio": round(oil_ratio * 100, 3),
        "estimated_area_km2": None,
        "area_status": "unavailable_without_georeferencing",
        "class_distribution": class_distribution,
        "origin": origin,
        "origin_status": "demonstration_coordinate_not_image_derived",
        "drift_track": drift_track(origin["lat"], origin["lon"]),
        "vessels": vessel_candidates(origin["lat"], origin["lon"]),
        "ais_status": "synthetic_demonstration_data",
        "mask_url": f"/media/{mask_path.name}",
        "overlay_url": f"/media/{overlay_path.name}",
    }


@app.get("/api/vessels")
def vessels():
    return {
        "status": "synthetic_demonstration_data",
        "vessels": vessel_candidates(DEFAULT_ORIGIN["lat"], DEFAULT_ORIGIN["lon"]),
    }


@app.get("/api/drift")
def drift():
    return {
        "status": "demonstration_track_not_scientific_hindcast",
        "origin": DEFAULT_ORIGIN,
        "track": drift_track(DEFAULT_ORIGIN["lat"], DEFAULT_ORIGIN["lon"]),
    }
