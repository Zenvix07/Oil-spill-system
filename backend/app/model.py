import os
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from huggingface_hub import hf_hub_download
import segmentation_models_pytorch as smp

MODEL_REPO = os.getenv("SAR_MODEL_REPO", "23f2003521/poseatsea-weights")
MODEL_FILE = os.getenv("SAR_MODEL_FILE", "best_sar_model.pth")
MODEL_SIZE = 512

# The POSEatSea checkpoint is a 5-class segmentation model.
# Keep this order configurable in case your team retrains the model.
CLASS_NAMES = [
    "background",
    "oil_spill",
    "ship",
    "look_alike",
    "wake",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_MODEL = None


def _download_checkpoint() -> str:
    local_dir = Path(__file__).resolve().parent.parent / "models"
    local_dir.mkdir(parents=True, exist_ok=True)
    local_file = local_dir / MODEL_FILE

    if not local_file.exists():
        downloaded = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_FILE,
            local_dir=str(local_dir),
        )
        return downloaded

    return str(local_file)


def load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    checkpoint_path = _download_checkpoint()

    model = smp.Unet(
        encoder_name="mit_b2",
        encoder_weights=None,
        in_channels=3,
        classes=5,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict):
        state = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))
    else:
        state = checkpoint

    cleaned = {}
    for key, value in state.items():
        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[7:]
        if new_key.startswith("model."):
            new_key = new_key[6:]
        cleaned[new_key] = value

    missing, unexpected = model.load_state_dict(cleaned, strict=False)

    # Fail loudly if the checkpoint is clearly incompatible.
    if len(missing) > 20:
        raise RuntimeError(
            f"Checkpoint architecture mismatch. Missing {len(missing)} parameters. "
            "Check backend/app/model.py against the checkpoint architecture."
        )

    model.to(DEVICE)
    model.eval()
    _MODEL = model

    print(f"[OILTRACE] Loaded {MODEL_REPO}/{MODEL_FILE} on {DEVICE}")
    if missing:
        print(f"[OILTRACE] Warning: {len(missing)} missing parameters")
    if unexpected:
        print(f"[OILTRACE] Warning: {len(unexpected)} unexpected parameters")

    return _MODEL


def preprocess(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB").resize((MODEL_SIZE, MODEL_SIZE), Image.Resampling.BILINEAR)
    arr = np.asarray(image).astype(np.float32) / 255.0

    # The checkpoint was trained as an RGB 3-channel segmentation model.
    # ImageNet normalization is used by the application adapter.
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std

    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(DEVICE)


@torch.inference_mode()
def predict(image: Image.Image):
    model = load_model()
    tensor = preprocess(image)

    logits = model(tensor)
    probabilities = torch.softmax(logits, dim=1)[0]
    labels = probabilities.argmax(dim=0).cpu().numpy().astype(np.uint8)
    probs = probabilities.cpu().numpy()

    oil_probability = probs[1]
    oil_mask = labels == 1

    return labels, oil_probability, oil_mask
