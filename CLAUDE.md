# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

FastAPI service that wraps Meta's SAM3 (Segment Anything Model 3) as a text-prompted, open-vocabulary object detector. It exposes REST endpoints for RTSP/image detection, an MJPEG live-preview stream, and a Label Studio ML-backend endpoint for human-in-the-loop annotation. A companion CLI tool converts Label Studio projects into YOLO training datasets (optionally auto-labeling via this server), and an optional NVIDIA Cosmos Predict2.5 service generates synthetic images to balance weak classes.

## Commands

```bash
# Install (Python 3.10+; PyTorch must match your CUDA version — see GPU gotchas below)
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129
pip install -r requirements.txt
cp .env.example .env   # then edit; SAM3_HF_TOKEN is required to download weights

# Run the API server (loads/downloads the ~2GB SAM3 model on startup)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Docker (needs NVIDIA Container Toolkit)
docker compose up -d
docker compose logs -f
docker compose down

# Syntax-check tools without running them
python -m py_compile tools/prepare_yolo_dataset.py tools/cosmos_predict25_batch.py
docker compose config

# YOLO dataset tool
python tools/prepare_yolo_dataset.py -p <PROJECT_ID> -o ./datasets/out --auto-label --force

# Full balance loop (Label Studio + Cosmos + SAM3)
python tools/prepare_yolo_dataset.py -p 1 --use-existing --auto-label \
  -o ./datasets/my_project --save-preview --balance-classes --cosmos-augment \
  --max-workers --sam3-batch-size auto --force
```

There is **no test suite, linter, or build step** configured despite what the README's "Development Setup" section implies (`tests/` does not exist, `pytest` is not a dependency). Do not assume `pytest tests/` works.

## Configuration

All server settings come from environment variables with the `SAM3_` prefix, loaded via `pydantic-settings` from `.env` (see `app/core/config.py` for the full list and defaults). `get_settings()` is `@lru_cache()`-ed — call `get_settings.cache_clear()` if you need to reload settings in tests. Note the config defaults differ from README examples — e.g. `score_threshold` defaults to `0.5`, `max_detections` to `25`. The YOLO tool reads a broader set of env vars including `SAM3_LABELSTUDIO_API_BASE`, `SAM3_LABELSTUDIO_API_TOKEN`, `SAM3_SERVER_URL`, and `YOLO_BATCH_SIZE`/`YOLO_WORKERS`.

Copy `.env.example` to `.env`. Minimum required:

```env
SAM3_HF_TOKEN=your_huggingface_token   # needed to download the ~2GB SAM3 checkpoint
SAM3_DEVICE=cuda                        # or cpu
```

Label Studio integration requires `SAM3_LABELSTUDIO_API_BASE` and `SAM3_LABELSTUDIO_API_TOKEN`. The model checkpoint is downloaded automatically from HuggingFace on first run if `SAM3_CHECKPOINT_PATH` is not set or the file doesn't exist.

## Architecture

**Request flow:** `app/main.py` (routes) → `RtspFrameSampler`/image fetch → `Sam3Detector.detect()` → NMS/filtering → response schema. The detector is a process-wide singleton cached via `@lru_cache` in `_get_detector()` and pre-loaded on FastAPI startup. Detection runs in a threadpool (`run_in_threadpool`) since the model is synchronous and GPU-bound.

Key routes:
- `POST /detect` — RTSP stream detection
- `GET /live-stream` — MJPEG stream with live annotations
- `POST /predict` and `POST /label-studio/predict` — Label Studio ML backend
- `POST /setup` and `POST /refresh-labels` — clear the Label Studio label cache
- `GET /labels` — inspect cached labels

**`app/models/sam3_detector.py` is the core.** Key points:
- SAM3 detection is text-prompted: each concept string is a separate prompt run against the image. The image embedding is computed once (`set_image`), then prompts are iterated with `set_text_prompt` + `reset_all_prompts` between each.
- Concept sources (in `_build_default_concepts`): `SAM3_CONCEPTS_PATH` file → else `DEFAULT_CONCEPTS` (COCO 80 + safety/PPE + Indian-traffic customs) → plus `SAM3_EXTRA_CONCEPTS`. Per-request `concepts` override the defaults entirely.
- Post-processing pipeline after raw detections: min-box-area filter → per-class NMS → cross-class NMS (removes the same object detected under different prompts, e.g. "car" vs "truck") → `max_detections` cap by score. IoU is computed manually in `_compute_iou`.

**SAM3 wheel workaround (`third_party/sam3_patch/`):** The published `sam3` wheel omits the `sam3.sam` subpackage needed at runtime. `_extend_sam3_namespace()` in the detector copies the vendored `third_party/sam3_patch/sam3/sam` into the installed package on first import failure. `_get_bpe_path()` locates the BPE vocab file across `sam3`/`clip`/`open_clip` to sidestep a `pkg_resources` lookup bug. If you upgrade `sam3`, re-vendor `sam3/sam` to match. Do not remove `third_party/sam3_patch/`.

**Label Studio integration (two directions):**
- `POST /predict` (also `/label-studio/predict`) is the ML-backend endpoint. `_predict_for_task` resolves the image from many possible field names, resizes large images (scaling boxes back afterward), runs detection, and formats results as `rectanglelabels` percentages.
- Concept/label resolution: if a task carries no `concepts`, labels are fetched from the Label Studio project config via `app/services/labelstudio.py` (XML-parsed, cached 1h; cleared on every `/setup` call). `_format_label_studio_results` in `main.py` maps SAM3's free-form prompt outputs back onto the project's exact label set using a large hardcoded synonym `label_mapping` (e.g. "flames"→"fire", "sedan"→"car"), and **drops any detection whose label isn't in the valid set**. When adding/renaming concepts, update both `DEFAULT_CONCEPTS` and this mapping or detections will be silently discarded.

### Tools

**`tools/prepare_yolo_dataset.py`** — standalone CLI (not imported by the app) that talks to Label Studio and the SAM3 `/predict` endpoint. Streams tasks in batches, downloads images in parallel (`ThreadPoolExecutor`), either uses existing annotations (`--use-existing`) or auto-labels (`--auto-label`), converts `RectangleLabels` annotations to YOLO format, splits into train/val/test, and writes `dataset.yaml`. With `--balance-classes --cosmos-augment` it also invokes LM Studio for prompt generation and triggers Cosmos jobs.

**`tools/cosmos_predict25_batch.py`** — creates and optionally runs NVIDIA Cosmos Predict2.5 generation jobs. Cosmos runs as a separate Docker service (`cosmos` in `docker-compose.yml`) built from a sibling repo checkout at `../cosmos-predict2.5`. Synthetic images are always placed in `train/` only, never `val/` or `test/`.

### Docker services

`docker-compose.yml` defines two services:
- `sam3-auto-labeler` — the FastAPI server, mounts `./weights`, `./datasets`, and the `hf_cache` named volume
- `cosmos` (optional) — Cosmos Predict2.5, expects `../cosmos-predict2.5` checked out as a sibling directory; communicates with the labeler via the `/auto-labeler` volume mount

## GPU / inference gotchas (learned the hard way)

- **Blackwell GPUs (RTX 50-series, sm_120) need cu128+ wheels.** The Dockerfile defaults to `torch==2.8.0` from the `cu129` index (overridable via the `PYTORCH_VERSION`/`PYTORCH_CUDA_INDEX` build args); the older `cu126` build only ships kernels up to sm_90 and fails at model load with `CUDA error: no kernel image is available for execution on the device`. Verify support with `torch.cuda.get_arch_list()` (must include `sm_120`).
- **Detection must run under CUDA autocast(bfloat16).** SAM3's fused/flash-attention kernels emit BFloat16 activations that otherwise hit FP32 linear weights and raise `mat1 and mat2 must have the same dtype, but got BFloat16 and Float`. `Sam3Detector.detect()` wraps the forward pass in `torch.autocast(device_type="cuda", dtype=torch.bfloat16)` (skipped on CPU). Don't remove this.
- **Model weights persist in the `hf_cache` named volume** (`/root/.cache/huggingface`), NOT in `./weights`. The model downloads into the HF cache; without that volume every `docker compose up --force-recreate` re-downloads ~2GB.

## Safety / PPE detection recall (don't regress)

Two post-processing steps in `Sam3Detector.detect()` were silently deleting safety-critical detections; both are now guarded and must stay that way:

- **Cross-class NMS must not suppress nested part-of objects.** A helmet/hard hat sits inside a person box, a license plate inside a car. `_apply_cross_class_nms` uses `_is_contained()` to skip suppression when a smaller different-class box is mostly inside a larger one (≤50% of its area, ≥70% contained). It still dedupes genuine same-object hits (e.g. "car" vs "truck" on one vehicle, which are similar-sized + high IoU). If you rewrite NMS, keep both behaviors — there's a regression test pattern in the git history of this change.
- **Safety-critical labels bypass `min_box_area`.** Hard hats seen from above can be ~120–190 px², well under the default `min_box_area=500`, so they were all filtered out. Labels matching `SAFETY_CRITICAL_KEYWORDS` (helmet, hat, vest, goggles, mask, fire, smoke, gun, weapon, knife, …) only need to clear a 64px floor. Non-safety classes keep the full `min_box_area`.

Also note `SAM3_SCORE_THRESHOLD=0.35` in `.env` (not the 0.5 config default) — a deliberate recall-favoring setting for safety use. It lets through occasional weak false positives (~0.37); that's the intended precision/recall tradeoff since humans review in Label Studio.

## Multi-GPU (auto-scales to all visible GPUs)

`Sam3DetectorPool` (in `sam3_detector.py`) loads **one full SAM3 model per visible GPU** — `torch.cuda.device_count()` drives the count, so 1/2/4 cards → 1/2/4 models with no code change. `docker-compose.yml` uses `count: all` and defaults `SAM3_*_VISIBLE_DEVICES` to `all` to expose every host GPU. Each `detect()` checks out an idle model from a `queue.Queue` (blocking when all are busy) and returns it — that's the "route by load" scheduler. `/readyz` reports `{"gpus": N, "devices": [...]}`.

Two placement gotchas that WILL silently break multi-GPU if refactored:
- **`build_sam3_image_model` only moves the model to GPU when `device == "cuda"` exactly** (`if device == "cuda": model.cuda()`). Passing `"cuda:1"` leaves it on CPU → `Input type CUDABFloat16 and weight type torch.FloatTensor`. So `_build_processor` passes `"cuda"` but wraps the build in `torch.cuda.device(index)` so `.cuda()` lands on the right card.
- `detect()` wraps inference in `torch.cuda.device(self.device.index)` so ambient-device ops go to the right card, not default cuda:0.

`/predict` fans its tasks out concurrently (`asyncio.gather` + semaphore of `2 × pool.size`), so a **single batch request** spreads across all GPUs. Verified: a 24-task batch kept both RTX 5090s busy simultaneously in 56/100 utilization samples. To use only specific cards, set `SAM3_NVIDIA_VISIBLE_DEVICES`/`SAM3_CUDA_VISIBLE_DEVICES` (or swap `count: all` for `device_ids: [...]`) in compose — e.g. to leave GPU 1 free for the `cosmos` service (which defaults to GPU 1 via `COSMOS_*_VISIBLE_DEVICES`).

## Health / readiness

`/healthz` (and `/health`, `/predict/health`) is liveness only — always 200 once the process is up. `/readyz` is readiness: it 503s until the SAM3 model actually loaded (`_model_ready` in `main.py`). Docker/compose healthchecks probe `/readyz` so a container that can't load the model is correctly unhealthy. `/predict` returns 503 (not 500) when the model is unavailable. When changing startup/model-load code, keep `_model_ready` accurate or the deployment will lie about its state.

## Key constraints

- `sam3` must be installed from source: `sam3 @ git+https://github.com/facebookresearch/sam3.git` (see `requirements.txt`)
- PyTorch must be installed separately before `pip install -r requirements.txt` because it requires a CUDA-specific index URL
- The `third_party/sam3_patch/` directory vendors the missing `sam3.sam` subpackage — do not remove it

## Conventions

- Modules use `from __future__ import annotations` and PEP 604 unions (`str | None`); target Python 3.10+.
- Detection I/O uses OpenCV BGR `np.ndarray` internally; convert to PIL RGB only at the SAM3 boundary.
- Startup/model-load code uses `print("[STARTUP]"/"[INIT]"...)`. The per-request `/predict` path uses the `logging` module (`logger = logging.getLogger("sam3")`) at DEBUG so production runs stay quiet and don't dump base64 image payloads; raise `LOG_LEVEL`/logger level to DEBUG to see per-task detail.
