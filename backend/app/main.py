from pathlib import Path
import uuid
import io
import json
import numpy as np
from PIL import Image, ImageDraw
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .model import predict, CLASS_NAMES, MODEL_SIZE, DEVICE
from .services import vessel_candidates, drift_track

BASE = Path(__file__).resolve().parent.parent
UPLOADS = BASE / "uploads"
OUTPUTS = BASE / "outputs"
UPLOADS.mkdir(exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)

app = FastAPI(title="OILTRACE API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/media", StaticFiles(directory=str(OUTPUTS)), name="media")


@app.get("/api/health")
def health():
    return {"status": "ok", "device": str(DEVICE), "model": "POSEatSea / best_sar_model.pth"}


def create_visuals(original: Image.Image, labels: np.ndarray, oil_probs: np.ndarray, job_id: str):
    base = original.convert("RGB").resize((MODEL_SIZE, MODEL_SIZE), Image.Resampling.BILINEAR)

    mask = (labels == 1).astype(np.uint8) * 255
    mask_img = Image.fromarray(mask, mode="L")

    # Semi-transparent red overlay for detected oil.
    overlay = base.convert("RGBA")
    red = Image.new("RGBA", overlay.size, (255, 70, 55, 0))
    alpha = Image.fromarray((mask * 0.58).astype(np.uint8), mode="L")
    red.putalpha(alpha)
    overlay = Image.alpha_composite(overlay, red).convert("RGB")

    # Bounding box.
    ys, xs = np.where(mask > 0)
    if len(xs):
        draw = ImageDraw.Draw(overlay)
        draw.rectangle([xs.min(), ys.min(), xs.max(), ys.max()], outline=(255, 255, 0), width=3)

    original_path = OUTPUTS / f"{job_id}_input.jpg"
    mask_path = OUTPUTS / f"{job_id}_mask.png"
    overlay_path = OUTPUTS / f"{job_id}_overlay.jpg"

    base.save(original_path, quality=92)
    mask_img.save(mask_path)
    overlay.save(overlay_path, quality=92)

    return original_path, mask_path, overlay_path, mask, overlay


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    raw = await file.read()
    if len(raw) > 30 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image is larger than 30 MB.")

    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}")

    job_id = uuid.uuid4().hex[:10]

    try:
        labels, oil_probs, oil_mask = predict(image)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Model inference failed: {exc}. Check model architecture/dependencies."
        )

    _, mask_path, overlay_path, mask, _ = create_visuals(image, labels, oil_probs, job_id)

    oil_pixels = int(oil_mask.sum())
    total_pixels = int(oil_mask.size)
    oil_ratio = oil_pixels / max(total_pixels, 1)

    # This is a relative image-space estimate because a normal uploaded PNG/JPEG
    # has no guaranteed geographic pixel scale.
    estimated_area_km2 = round(oil_ratio * 100.0, 2)

    class_distribution = {}
    for idx, name in enumerate(CLASS_NAMES):
        class_distribution[name] = round(float((labels == idx).mean() * 100), 3)

    # Demo origin around a configurable default AOI.
    origin = {"lat": 10.50, "lon": 72.50}
    vessels = vessel_candidates(origin["lat"], origin["lon"])
    track = drift_track(origin["lat"], origin["lon"])

    confidence = float(oil_probs[oil_mask].mean()) if oil_pixels else float(oil_probs[1].mean())

    return {
        "job_id": job_id,
        "model": "POSEatSea best_sar_model.pth",
        "device": str(DEVICE),
        "input_size": [MODEL_SIZE, MODEL_SIZE],
        "oil_detected": bool(oil_ratio > 0.005),
        "oil_confidence": round(confidence * 100, 2),
        "oil_pixel_ratio": round(oil_ratio * 100, 3),
        "estimated_area_km2": estimated_area_km2,
        "class_distribution": class_distribution,
        "origin": origin,
        "drift_track": track,
        "vessels": vessels,
        "mask_url": f"/media/{mask_path.name}",
        "overlay_url": f"/media/{overlay_path.name}",
    }


@app.get("/api/vessels")
def vessels():
    return {"vessels": vessel_candidates()}


@app.get("/api/drift")
def drift():
    origin = {"lat": 10.50, "lon": 72.50}
    return {"origin": origin, "track": drift_track(origin["lat"], origin["lon"])}
