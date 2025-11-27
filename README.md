# SAM3 RTSP Object Detection API

FastAPI + Uvicorn microservice that samples frames from an RTSP stream and runs Meta's SAM3 "Segment Anything with Concepts" model for open-vocabulary object detections (COCO + industrial safety prompts out of the box).

## Features
- REST endpoint (`POST /detect`) for per-request RTSP inference with optional concept overrides.
- OpenCV-powered RTSP sampler with timeout protection and configurable frame skipping.
- Official `facebookresearch/sam3` integration for open-vocabulary detections (COCO + safety prompts by default).
- Live MJPEG preview (`GET /live-stream`) that restreams RTSP frames with detections drawn on every frame.
- Label Studio-compatible ML backend endpoint (`POST /predict`) for human-in-the-loop workflows.
- **YOLO Dataset Preparation Tool** - Auto-label images from Label Studio using SAM3 and export to YOLO format.
- Health probe (`GET /healthz`) for liveness checks.

## YOLO Dataset Preparation Tool

A powerful command-line tool to create YOLO-format datasets from Label Studio projects with SAM3 auto-labeling support.

### Features
- Fetches images from Label Studio projects (streaming, one-by-one)
- Auto-labels images using SAM3 detection server
- Converts annotations to YOLO format
- Splits dataset into train/val/test sets
- Generates `dataset.yaml` for YOLO training
- Saves preview images with bounding boxes for verification
- Progress bars with colored output and statistics

### Usage

```bash
# Basic usage with auto-labeling
python tools/prepare_yolo_dataset.py -p PROJECT_ID --auto-label -o ./datasets/mydata

# Use existing Label Studio annotations
python tools/prepare_yolo_dataset.py -p PROJECT_ID --use-existing -o ./datasets/mydata

# Auto-label with preview images (to verify detection quality)
python tools/prepare_yolo_dataset.py -p PROJECT_ID --auto-label -o ./datasets/mydata \
  --save-preview --max-tasks 20 --sam3-url http://localhost:8080

# Full example with all options
python tools/prepare_yolo_dataset.py \
  -p 1 \
  --auto-label \
  -o ./datasets/traffic \
  --force \
  --sam3-url http://localhost:8080 \
  --save-preview \
  --max-tasks 100 \
  --train-split 0.8 \
  --val-split 0.15 \
  --test-split 0.05
```

### Options

| Option | Description | Default |
| --- | --- | --- |
| `-p, --project-id` | Label Studio project ID (required) | - |
| `-o, --output-dir` | Output directory for YOLO dataset (required) | - |
| `--ls-url` | Label Studio URL | From `.env` |
| `--ls-token` | Label Studio API token | From `.env` |
| `--sam3-url` | SAM3 server URL | `http://localhost:8000` |
| `--use-existing` | Use existing annotations from Label Studio | `false` |
| `--auto-label` | Auto-label images using SAM3 | `false` |
| `--save-preview` | Save labeled preview images to `~/labeled_previews` | `false` |
| `--max-tasks` | Maximum tasks to process (0 = all) | `0` |
| `--train-split` | Train split ratio | `0.8` |
| `--val-split` | Validation split ratio | `0.15` |
| `--test-split` | Test split ratio | `0.05` |
| `--force` | Overwrite existing output directory | `false` |

### SAM3 Detection Configuration

The tool reads SAM3 configuration from `.env` file for better detection quality:

```env
SAM3_SCORE_THRESHOLD=0.35    # Lower = more detections
SAM3_NMS_THRESHOLD=0.3       # Non-max suppression threshold
SAM3_MIN_BOX_AREA=500        # Minimum bounding box area (pixels)
SAM3_MAX_DETECTIONS=250      # Max detections per image
SAM3_CROSS_CLASS_NMS=true    # Cross-class NMS to reduce duplicates
```

### Output Structure

```
datasets/mydata/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── dataset.yaml
```

### Train YOLO Model

After preparing the dataset:

```bash
yolo detect train data=./datasets/mydata/dataset.yaml model=yolov8n.pt epochs=100
```

## Prerequisites
1. Python 3.10+
2. PyTorch 2.7.0 with CUDA 12.6 (or CPU-only build) installed manually, e.g.
   ```bash
   pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
     --index-url https://download.pytorch.org/whl/cu126
  ```

## Live stream preview
- Endpoint: `GET /live-stream`
- Required query parameter: `rtsp_url=rtsp://...`
- Optional query parameters:
  - `frame_skip` (defaults to `1`) to throttle how often detections/frames are emitted.
  - Repeatable `concepts` parameters to override the default prompts (same limits as `/detect`).
- Response: `multipart/x-mixed-replace` (MJPEG). Each JPEG has the detected bounding boxes, labels, and scores rendered directly on the frame.

The index page automatically wires your form input to this endpoint and displays the MJPEG feed under the “Live stream preview” heading, so you can watch the annotated stream while the JSON results continue to refresh.

## Label Studio ML backend

- Endpoint: `POST /predict` (Label Studio’s default) — `/label-studio/predict` stays available for backward compatibility.
- Body: `{ "tasks": [ { "id": "...", "data": { "rtsp_url": "rtsp://...", "frame_skip": 1, "concepts": ["person"] } } ] }`
- Returns prediction objects that Label Studio can ingest directly. Each bounding box is emitted as a `rectanglelabels` result using `from_name="label"` and `to_name="image"`, so configure your project with matching control/input names.

Example request:
```bash
curl -X POST http://localhost:8000/label-studio/predict \
  -H "Content-Type: application/json" \
  -d '{
        "tasks": [
          {
            "id": "task-1",
            "data": {
              "rtsp_url": "rtsp://192.168.1.50:8554/live",
              "frame_skip": 1,
              "concepts": ["person", "safety helmet"]
            }
          }
        ]
      }'
```

Register the backend in the Label Studio UI under *Settings → Machine Learning* by pointing to `http://<server>:8000/predict`. Use the same `from_name`/`to_name` pairing in your labeling config so the rectangles attach to the intended image or video node. (If you already configured `/label-studio/predict`, it will continue to work.)
3. Hugging Face access to [`facebook/sam3`](https://huggingface.co/facebook/sam3) plus an access token saved as `SAM3_HF_TOKEN` (see `.env.example`).
4. (Optional) Local checkpoint path if you prefer not to download from Hugging Face at runtime.

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Configuration
The server reads environment variables via `.env` (all variables are automatically prefixed with `SAM3_`). Key options:

| Variable | Description | Default |
| --- | --- | --- |
| `SAM3_CHECKPOINT_PATH` | Optional path to a local `sam3.pt`. If omitted, the HF checkpoint is downloaded (requires token). | `None` |
| `SAM3_HF_TOKEN` | Hugging Face token used when auto-downloading checkpoints. | `None` |
| `SAM3_DEVICE` | `cuda` or `cpu`. Falls back to `cpu` if CUDA is unavailable. | `cuda` |
| `SAM3_SCORE_THRESHOLD` | Minimum confidence score to keep detections. | `0.50` |
| `SAM3_MAX_DETECTIONS` | Max detections returned per frame. | `25` |
| `SAM3_RTSP_TIMEOUT` | Seconds to keep trying to read RTSP before aborting. | `5` |
| `SAM3_FRAME_SKIP` | Frames skipped between samples (set to `1` for realtime). | `1` |
| `SAM3_CONCEPTS_PATH` | Optional newline-delimited file listing concept prompts to run. Overrides defaults when set. | `None` |
| `SAM3_EXTRA_CONCEPTS` | Comma-separated prompts appended after the defaults/file concepts. | `None` |
| `SAM3_USE_DEFAULT_CONCEPTS` | `true/false` flag to include the built-in COCO + safety prompts. | `true` |
| `SAM3_MAX_CONCEPTS_PER_REQUEST` | Safety guard for per-request concept overrides. | `160` |
| `SAM3_LABELSTUDIO_FROM_NAME` | Label Studio control tag (`from_name`) to attach detections to. | `label` |
| `SAM3_LABELSTUDIO_TO_NAME` | Label Studio object tag (`to_name`) that receives rectangle results. | `image` |
| `SAM3_LABELSTUDIO_MODEL_VERSION` | Version string returned in Label Studio prediction responses. | `sam3-v0.1` |
| `SAM3_LABELSTUDIO_API_BASE` | Base URL of your Label Studio instance (e.g., `http://10.10.18.20:8080`). | `None` |
| `SAM3_LABELSTUDIO_API_TOKEN` | Personal access token used when calling the Label Studio REST API. | `None` |

Example `.env`:
```env
SAM3_CHECKPOINT_PATH=./weights/sam3.pt
SAM3_DEVICE=cuda
SAM3_SCORE_THRESHOLD=0.5
SAM3_MAX_DETECTIONS=25
SAM3_FRAME_SKIP=1
SAM3_EXTRA_CONCEPTS=fire alarm,worker without helmet
```

## Running the server
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Once the server is running, open `http://localhost:8000/` to use the built-in test page. Enter any RTSP URL, optional
concept overrides, and start the polling loop to see detections refresh in near real-time.

## Sample request
```bash
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{
        "rtsp_url": "rtsp://192.168.1.50:8554/live",
        "max_frames": 2,
        "concepts": ["fire", "smoke", "safety helmet"]
      }'
```

Sample response:
```json
{
  "frames_analyzed": 2,
  "detections": [
    {
      "frame_index": 0,
      "label": "fire",
      "bbox": {"x": 120, "y": 33, "width": 188, "height": 201},
      "area": 29000,
      "score": 0.93
    }
  ]
}
```

## Concept prompts
- By default the detector runs the 80 COCO classes plus a curated set of road-safety, industrial, fire, and PPE prompts (see `app/models/sam3_detector.py`).
- Override the prompt list globally by pointing `SAM3_CONCEPTS_PATH` at a newline-separated file, or append ad-hoc prompts via `SAM3_EXTRA_CONCEPTS`.
- You can also send a `concepts` array inside the `/detect` payload. The server enforces `SAM3_MAX_CONCEPTS_PER_REQUEST` to avoid runaway inference times.

## Notes
- The detector uses Meta's official `facebookresearch/sam3` implementation. Make sure your environment satisfies the PyTorch/CUDA requirements listed above.
- RTSP streams vary widely; if you need to throttle processing you can raise `frame_skip`, but the default `1` keeps every frame for realtime detection.
- The published `sam3` wheel omits the `sam3.sam` subpackage, so this repo vendors those files under `third_party/sam3_patch`. Re-copy `sam3/sam` from upstream whenever you upgrade the dependency.
