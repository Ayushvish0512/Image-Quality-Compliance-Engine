# BBIPL Vision Compliance Engine (VCE)

A lightweight Computer Vision API that validates employee attendance selfies before they are accepted by the ERP system.

## Features
- Face Detection & Visibility
- Person Detection
- Blur & Lighting Detection
- Face Distance & Orientation
- Uniform, Cap & Accessory Detection
- Screenshot Risk Detection
- Configurable Compliance Scoring

## Tech Stack
- **Backend**: FastAPI
- **Image Processing**: OpenCV
- **Face Detection**: MediaPipe
- **Config**: YAML

## Setup
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## API Endpoints
- `GET /health` - Health check
- `POST /validate` - Full compliance check
- `POST /check/*` - Individual checks

## Configuration
Edit `config/rules.yaml` to update business rules without code changes.

