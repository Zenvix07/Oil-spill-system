# OILTRACE — SIH NTRO Oil Spill Intelligence Console

Full-stack React + FastAPI prototype for:
1. Uploading a satellite/SAR image.
2. Running an existing pretrained oil-spill segmentation checkpoint.
3. Returning an oil mask + overlay + area/confidence estimates.
4. Displaying the result on an operational dashboard.
5. Ranking candidate vessels using a transparent AIS correlation score.

## Existing model used

The backend is wired to the POSEatSea SAR checkpoint:

- Hugging Face repository: `23f2003521/poseatsea-weights`
- File: `best_sar_model.pth`
- Architecture: U-Net + MiT-B2 encoder
- 5 classes
- Required inference size: 512×512

The model is downloaded automatically on first inference into `backend/models/`.

This is deliberately kept as an adapter so you can replace the checkpoint later without changing the React UI.

## Requirements

- Python 3.10+
- Node.js 18+
- 4 GB+ RAM recommended
- NVIDIA GPU recommended for fast inference, but CPU is supported
- Internet connection on first model download

## Backend

```bash
cd backend
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows
# .venv\Scripts\activate

pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## Frontend

```bash
cd frontend
npm install
npm run dev
```

Open:

http://localhost:5173

## Model notes

The checkpoint is expected to work with the exact 512×512 input size. Do not silently change that size.

If the checkpoint architecture in your downloaded version differs, the only file that normally needs modification is:

`backend/app/model.py`

The web application itself does not depend on the model implementation.

## API

### Health

GET `/api/health`

### Analyze satellite image

POST `/api/analyze`

Multipart form field:

`file`

Response includes:

- oil confidence
- oil pixel ratio
- estimated area
- mask image URL
- overlay image URL
- class distribution

### AIS candidate ranking demo

GET `/api/vessels`

This endpoint currently generates deterministic demonstration AIS candidates. Replace the service with your historical AIS database/API when available.

## Important

The displayed vessel score is an investigation aid, not proof that a vessel caused a spill.

The satellite segmentation model also detects visual patterns that can have look-alikes. Human/analyst review and additional evidence are required for operational attribution.
