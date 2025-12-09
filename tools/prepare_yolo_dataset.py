#!/usr/bin/env python3
"""
YOLO Dataset Preparation Tool

This tool fetches annotated tasks from Label Studio and converts them to YOLO format.
It can also auto-label unannotated images using the SAM3 detection server.

Features:
- Fetches tasks from Label Studio project
- Converts RectangleLabels to YOLO format
- Auto-labels images using SAM3 (optional)
- Splits dataset into train/val/test
- Creates dataset.yaml for YOLO training
- Progress bars and colored output
- Statistics and class distribution

Usage:
    python tools/prepare_yolo_dataset.py -p PROJECT_ID -o OUTPUT_DIR [options]

Examples:
    # Use existing annotations only
    python tools/prepare_yolo_dataset.py -p 1 -o ./datasets/traffic --use-existing

    # Auto-label all images with SAM3
    python tools/prepare_yolo_dataset.py -p 1 -o ./datasets/traffic --auto-label

    # Custom SAM3 server URL
    python tools/prepare_yolo_dataset.py -p 1 -o ./datasets/traffic --auto-label --sam3-url http://localhost:8080
"""

import argparse
import os
import sys
import shutil
import random
import time
import base64
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Load .env file automatically
try:
    from dotenv import load_dotenv
    # Try to find .env in current dir or parent dirs
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()  # Try current directory
except ImportError:
    pass  # dotenv not installed, use environment variables directly

import requests


# Default values
DEFAULT_LABEL_STUDIO_URL = os.environ.get("SAM3_LABELSTUDIO_API_BASE", "http://localhost:8080")
DEFAULT_SAM3_SERVER_URL = os.environ.get("SAM3_SERVER_URL", "http://localhost:8000")
DEFAULT_API_TOKEN = os.environ.get("SAM3_LABELSTUDIO_API_TOKEN", "")

# SAM3 detection configuration from .env
SAM3_SCORE_THRESHOLD = float(os.environ.get("SAM3_SCORE_THRESHOLD", "0.5"))
SAM3_NMS_THRESHOLD = float(os.environ.get("SAM3_NMS_THRESHOLD", "0.3"))
SAM3_MIN_BOX_AREA = int(os.environ.get("SAM3_MIN_BOX_AREA", "500"))
SAM3_MAX_DETECTIONS = int(os.environ.get("SAM3_MAX_DETECTIONS", "25"))
SAM3_CROSS_CLASS_NMS = os.environ.get("SAM3_CROSS_CLASS_NMS", "true").lower() == "true"

# Batch processing configuration
DEFAULT_BATCH_SIZE = int(os.environ.get("YOLO_BATCH_SIZE", "10"))
DEFAULT_WORKERS = int(os.environ.get("YOLO_WORKERS", "4"))


class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_banner():
    """Print the tool banner."""
    print(f"""
{Colors.CYAN}╔══════════════════════════════════════════════════════════════╗
║  {Colors.BOLD}YOLO Dataset Preparation Tool{Colors.ENDC}{Colors.CYAN}                               ║
║  Label Studio → YOLO Format Converter                         ║
║  With SAM3 Auto-Labeling Support                              ║
╚══════════════════════════════════════════════════════════════╝{Colors.ENDC}
""")


def print_section(title: str):
    """Print a section header."""
    print(f"\n{Colors.BOLD}{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}{Colors.ENDC}")


def format_time(seconds: float) -> str:
    """Format seconds into human readable time."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def format_size(bytes: int) -> str:
    """Format bytes into human readable size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024:
            return f"{bytes:.1f} {unit}"
        bytes /= 1024
    return f"{bytes:.1f} TB"


class Stats:
    """Track processing statistics (thread-safe)."""
    
    def __init__(self):
        self.start_time = time.time()
        self.total_tasks = 0
        self.processed = 0
        self.success = 0
        self.skipped = 0
        self.failed = 0
        self.total_annotations = 0
        self.total_bytes_downloaded = 0
        self.class_counts: Dict[str, int] = {}
        self._lock = Lock()
    
    def add_annotations(self, count: int, labels: List[str]):
        with self._lock:
            self.total_annotations += count
            for label in labels:
                self.class_counts[label] = self.class_counts.get(label, 0) + 1
    
    def add_bytes(self, bytes_count: int):
        with self._lock:
            self.total_bytes_downloaded += bytes_count
    
    def increment_success(self):
        with self._lock:
            self.processed += 1
            self.success += 1
    
    def increment_skipped(self):
        with self._lock:
            self.processed += 1
            self.skipped += 1
    
    def increment_failed(self):
        with self._lock:
            self.processed += 1
            self.failed += 1
    
    def elapsed(self) -> float:
        return time.time() - self.start_time
    
    def eta(self) -> str:
        with self._lock:
            processed = self.processed
        if processed == 0:
            return "calculating..."
        elapsed = self.elapsed()
        rate = processed / elapsed
        remaining = (self.total_tasks - processed) / rate
        return format_time(remaining)
    
    def print_progress(self, task_id: Any = None, batch_info: str = ""):
        """Print current progress."""
        with self._lock:
            processed = self.processed
            success = self.success
            skipped = self.skipped
            failed = self.failed
        
        pct = (processed / self.total_tasks * 100) if self.total_tasks > 0 else 0
        elapsed = format_time(self.elapsed())
        eta = self.eta()
        
        # Create progress bar
        bar_width = 30
        filled = int(bar_width * processed / self.total_tasks) if self.total_tasks > 0 else 0
        bar = '█' * filled + '░' * (bar_width - filled)
        
        status = f"\r{Colors.CYAN}[{bar}]{Colors.ENDC} {pct:5.1f}% | "
        if batch_info:
            status += f"{batch_info} | "
        elif task_id:
            status += f"Task: {task_id} | "
        status += f"{Colors.GREEN}✓{success}{Colors.ENDC} "
        status += f"{Colors.YELLOW}⊘{skipped}{Colors.ENDC} "
        status += f"{Colors.RED}✗{failed}{Colors.ENDC} | "
        status += f"⏱ {elapsed} | ETA: {eta}  "
        
        print(status, end='', flush=True)
    
    def print_summary(self):
        """Print final summary."""
        print_section("Processing Summary")
        
        elapsed = format_time(self.elapsed())
        rate = self.processed / self.elapsed() if self.elapsed() > 0 else 0
        
        print(f"""
  {Colors.BOLD}Tasks Processed:{Colors.ENDC}
    • Total:     {self.total_tasks}
    • Success:   {Colors.GREEN}{self.success}{Colors.ENDC}
    • Skipped:   {Colors.YELLOW}{self.skipped}{Colors.ENDC}
    • Failed:    {Colors.RED}{self.failed}{Colors.ENDC}
  
  {Colors.BOLD}Annotations:{Colors.ENDC}
    • Total boxes: {self.total_annotations}
    • Downloaded:  {format_size(self.total_bytes_downloaded)}
  
  {Colors.BOLD}Performance:{Colors.ENDC}
    • Total time:  {elapsed}
    • Rate:        {rate:.2f} tasks/sec
""")
        
        if self.class_counts:
            print(f"  {Colors.BOLD}Class Distribution:{Colors.ENDC}")
            sorted_classes = sorted(self.class_counts.items(), key=lambda x: -x[1])
            max_count = max(self.class_counts.values())
            bar_max_width = 30
            
            for label, count in sorted_classes:
                bar_width = int(bar_max_width * count / max_count)
                bar = '█' * bar_width
                print(f"    {label:20s} {Colors.CYAN}{bar}{Colors.ENDC} {count}")


class LabelStudioClient:
    """Client for Label Studio API."""

    def __init__(self, base_url: str, api_token: str):
        self.base_url = base_url.rstrip("/")
        self.headers = {"Authorization": f"Token {api_token}"}
        # Create a session for connection pooling (faster for multiple requests)
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def get_project(self, project_id: int) -> Dict[str, Any]:
        """Get project details including label config."""
        url = f"{self.base_url}/api/projects/{project_id}"
        resp = self.session.get(url)
        resp.raise_for_status()
        return resp.json()

    def get_task_count(self, project_id: int) -> int:
        """Get total number of tasks in project."""
        url = f"{self.base_url}/api/projects/{project_id}"
        resp = self.session.get(url)
        resp.raise_for_status()
        data = resp.json()
        return data.get("task_number", 0)

    def get_all_tasks(self, project_id: int, max_tasks: int = 0, show_progress: bool = True) -> List[Dict[str, Any]]:
        """Fetch ALL tasks at once with large page size for efficiency."""
        all_tasks = []
        page = 1
        page_size = 100  # Fetch 100 at a time
        
        # Get total count first for progress bar
        total_count = self.get_task_count(project_id)
        if max_tasks > 0:
            total_count = min(total_count, max_tasks)
        
        while True:
            url = f"{self.base_url}/api/projects/{project_id}/tasks"
            params = {"page": page, "page_size": page_size}
            resp = self.session.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            
            if isinstance(data, list):
                batch = data
            else:
                batch = data.get("tasks", data.get("results", []))
            
            if not batch:
                break
            
            all_tasks.extend(batch)
            
            # Show progress bar
            if show_progress:
                fetched = len(all_tasks)
                pct = (fetched / total_count * 100) if total_count > 0 else 100
                bar_width = 30
                filled = int(bar_width * fetched / total_count) if total_count > 0 else bar_width
                bar = '█' * filled + '░' * (bar_width - filled)
                print(f"\r  {Colors.CYAN}[{bar}]{Colors.ENDC} {pct:5.1f}% | Fetched {fetched}/{total_count} tasks", end='', flush=True)
            
            # Check max_tasks limit
            if max_tasks > 0 and len(all_tasks) >= max_tasks:
                all_tasks = all_tasks[:max_tasks]
                break
            
            # Check if there are more pages
            if isinstance(data, dict) and data.get("next"):
                page += 1
            elif len(batch) < page_size:
                break
            else:
                page += 1
        
        if show_progress:
            print()  # New line after progress bar
        
        return all_tasks

    def iter_tasks(self, project_id: int, page_size: int = 1):
        """Iterate over tasks one by one (generator) - legacy method."""
        page = 1
        while True:
            url = f"{self.base_url}/api/projects/{project_id}/tasks"
            params = {"page": page, "page_size": page_size}
            resp = self.session.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            
            if isinstance(data, list):
                batch = data
            else:
                batch = data.get("tasks", data.get("results", []))
            
            if not batch:
                break
            
            for task in batch:
                yield task
            
            # Check if there are more pages
            if isinstance(data, dict) and data.get("next"):
                page += 1
            elif len(batch) < page_size:
                break
            else:
                page += 1

    def iter_task_batches(self, project_id: int, batch_size: int = 10):
        """Iterate over tasks in batches (generator yielding lists)."""
        page = 1
        while True:
            url = f"{self.base_url}/api/projects/{project_id}/tasks"
            params = {"page": page, "page_size": batch_size}
            resp = self.session.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            
            if isinstance(data, list):
                batch = data
            else:
                batch = data.get("tasks", data.get("results", []))
            
            if not batch:
                break
            
            yield batch
            
            # Check if there are more pages
            if isinstance(data, dict) and data.get("next"):
                page += 1
            elif len(batch) < batch_size:
                break
            else:
                page += 1

    def get_tasks(self, project_id: int, page_size: int = 100) -> List[Dict[str, Any]]:
        """Get all tasks from a project (for backward compatibility)."""
        return list(self.iter_tasks(project_id, page_size))

    def get_image_url(self, task: Dict[str, Any]) -> Optional[str]:
        """Extract image URL from task data."""
        data = task.get("data", {})
        
        # Try common field names
        for field in ["image", "img", "photo", "file", "url"]:
            if field in data:
                url = data[field]
                if isinstance(url, str):
                    # Handle Label Studio local file URLs
                    if url.startswith("/data/"):
                        return f"{self.base_url}{url}"
                    return url
        
        return None


class SAM3Client:
    """Client for SAM3 detection server."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        # Load SAM3 config from environment
        self.score_threshold = SAM3_SCORE_THRESHOLD
        self.nms_threshold = SAM3_NMS_THRESHOLD
        self.min_box_area = SAM3_MIN_BOX_AREA
        self.max_detections = SAM3_MAX_DETECTIONS
        self.cross_class_nms = SAM3_CROSS_CLASS_NMS

    def health_check(self) -> bool:
        """Check if SAM3 server is running."""
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def predict_with_base64(self, image_base64: str, concepts: List[str]) -> List[Dict[str, Any]]:
        """Get predictions from SAM3 server using base64 image."""
        # Format: data:image/jpeg;base64,<base64_data>
        if not image_base64.startswith("data:"):
            image_base64 = f"data:image/jpeg;base64,{image_base64}"
        
        payload = {
            "tasks": [{
                "id": 0,
                "data": {
                    "image": image_base64,
                    "concepts": concepts,
                    # Pass SAM3 config for better detection
                    "score_threshold": self.score_threshold,
                    "nms_threshold": self.nms_threshold,
                    "min_box_area": self.min_box_area,
                    "max_detections": self.max_detections,
                    "cross_class_nms": self.cross_class_nms,
                }
            }]
        }
        
        resp = requests.post(
            f"{self.base_url}/predict",
            json=payload,
            timeout=120,
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        
        data = resp.json()
        
        # Handle different response formats
        if "results" in data:
            results = data["results"]
        elif isinstance(data, list):
            results = data
        else:
            results = [data]
        
        if results and "result" in results[0]:
            return results[0]["result"]
        
        return []

    def predict(self, image_url: str, concepts: List[str]) -> List[Dict[str, Any]]:
        """Get predictions from SAM3 server (legacy URL method)."""
        payload = {
            "tasks": [{
                "id": 0,
                "data": {
                    "image": image_url,
                    "concepts": concepts,
                    # Pass SAM3 config for better detection
                    "score_threshold": self.score_threshold,
                    "nms_threshold": self.nms_threshold,
                    "min_box_area": self.min_box_area,
                    "max_detections": self.max_detections,
                    "cross_class_nms": self.cross_class_nms,
                }
            }]
        }
        
        resp = requests.post(
            f"{self.base_url}/predict",
            json=payload,
            timeout=120,
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        
        data = resp.json()
        
        # Handle different response formats
        if "results" in data:
            results = data["results"]
        elif isinstance(data, list):
            results = data
        else:
            results = [data]
        
        if results and "result" in results[0]:
            return results[0]["result"]
        
        return []

    def predict_batch(self, images_base64: List[Tuple[int, str]], concepts: List[str]) -> Dict[int, List[Dict[str, Any]]]:
        """
        Batch prediction for multiple images.
        
        Args:
            images_base64: List of (task_id, base64_image_data) tuples
            concepts: List of concept labels to detect
            
        Returns:
            Dict mapping task_id to list of detection results
        """
        # Build batch request
        tasks = []
        for task_id, img_b64 in images_base64:
            if not img_b64.startswith("data:"):
                img_b64 = f"data:image/jpeg;base64,{img_b64}"
            
            tasks.append({
                "id": task_id,
                "data": {
                    "image": img_b64,
                    "concepts": concepts,
                    "score_threshold": self.score_threshold,
                    "nms_threshold": self.nms_threshold,
                    "min_box_area": self.min_box_area,
                    "max_detections": self.max_detections,
                    "cross_class_nms": self.cross_class_nms,
                }
            })
        
        payload = {"tasks": tasks}
        
        resp = requests.post(
            f"{self.base_url}/predict",
            json=payload,
            timeout=300,  # Longer timeout for batch
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        
        data = resp.json()
        
        # Parse response into dict by task_id
        results_map = {}
        
        if "results" in data:
            results_list = data["results"]
        elif isinstance(data, list):
            results_list = data
        else:
            results_list = [data]
        
        # Match results back to task IDs
        for i, result in enumerate(results_list):
            if i < len(images_base64):
                task_id = images_base64[i][0]
                results_map[task_id] = result.get("result", [])
        
        return results_map


def parse_labels_from_config(label_config: str) -> List[str]:
    """Parse label names from Label Studio XML config."""
    import re
    
    labels = []
    
    # Find all Label tags with value attribute
    pattern = r'<Label[^>]+value="([^"]+)"'
    matches = re.findall(pattern, label_config)
    labels.extend(matches)
    
    # Also check for labels in Choices
    pattern = r'<Choice[^>]+value="([^"]+)"'
    matches = re.findall(pattern, label_config)
    labels.extend(matches)
    
    return list(dict.fromkeys(labels))  # Remove duplicates, preserve order


def get_annotations_from_task(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract rectangle annotations from task."""
    annotations = []
    
    for annotation in task.get("annotations", []):
        for result in annotation.get("result", []):
            if result.get("type") == "rectanglelabels":
                value = result.get("value", {})
                labels = value.get("rectanglelabels", [])
                
                if labels:
                    annotations.append({
                        "label": labels[0],
                        "x": value.get("x", 0),
                        "y": value.get("y", 0),
                        "width": value.get("width", 0),
                        "height": value.get("height", 0),
                    })
    
    return annotations


def convert_to_yolo_format(
    annotations: List[Dict[str, Any]],
    label_to_id: Dict[str, int],
) -> List[str]:
    """Convert annotations to YOLO format strings."""
    yolo_lines = []
    
    for ann in annotations:
        label = ann["label"]
        if label not in label_to_id:
            continue
        
        class_id = label_to_id[label]
        
        # Label Studio uses percentages (0-100), YOLO uses normalized (0-1)
        x_center = (ann["x"] + ann["width"] / 2) / 100
        y_center = (ann["y"] + ann["height"] / 2) / 100
        width = ann["width"] / 100
        height = ann["height"] / 100
        
        # Clamp values to valid range
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        width = max(0, min(1, width))
        height = max(0, min(1, height))
        
        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    return yolo_lines


# Colors for drawing bounding boxes (BGR for different classes)
BBOX_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
    (128, 0, 128), (0, 128, 128), (255, 128, 0), (255, 0, 128), (128, 255, 0),
]


def draw_preview_image(
    image_data: bytes,
    annotations: List[Dict[str, Any]],
    label_to_id: Dict[str, int],
) -> Optional[bytes]:
    """Draw bounding boxes on image and return as bytes."""
    if not PIL_AVAILABLE:
        return None
    
    try:
        # Load image
        img = Image.open(io.BytesIO(image_data))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        draw = ImageDraw.Draw(img)
        width, height = img.size
        
        # Try to load a font, fall back to default
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        for ann in annotations:
            label = ann["label"]
            if label not in label_to_id:
                continue
            
            class_id = label_to_id[label]
            color = BBOX_COLORS[class_id % len(BBOX_COLORS)]
            
            # Convert percentage to pixels
            x1 = int(ann["x"] * width / 100)
            y1 = int(ann["y"] * height / 100)
            x2 = int((ann["x"] + ann["width"]) * width / 100)
            y2 = int((ann["y"] + ann["height"]) * height / 100)
            
            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # Draw label background
            text = f"{label}"
            bbox = draw.textbbox((x1, y1 - 20), text, font=font)
            draw.rectangle([bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2], fill=color)
            draw.text((x1, y1 - 20), text, fill=(255, 255, 255), font=font)
        
        # Save to bytes
        output = io.BytesIO()
        img.save(output, format='JPEG', quality=90)
        return output.getvalue()
    
    except Exception as e:
        print(f"\n  {Colors.YELLOW}Warning: Could not draw preview: {e}{Colors.ENDC}")
        return None


def download_image(url: str, output_path: Path, headers: Dict[str, str] = None, session: requests.Session = None) -> int:
    """Download image and return bytes downloaded."""
    try:
        requester = session if session else requests
        resp = requester.get(url, headers=headers, timeout=30, stream=True)
        resp.raise_for_status()
        
        total_bytes = 0
        with open(output_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                total_bytes += len(chunk)
        
        return total_bytes
    except Exception as e:
        print(f"\n  {Colors.RED}✗ Download failed: {e}{Colors.ENDC}")
        return 0


def download_image_to_memory(url: str, headers: Dict[str, str] = None, session: requests.Session = None) -> Optional[bytes]:
    """Download image to memory and return bytes."""
    try:
        requester = session if session else requests
        resp = requester.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        return resp.content
    except Exception:
        return None


def process_task_download(
    task: Dict[str, Any],
    ls_client: LabelStudioClient,
) -> Tuple[int, Optional[bytes], Optional[str], str]:
    """
    Download image for a task (thread-safe, uses connection pooling).
    Returns (task_id, image_data, filename, status) tuple.
    """
    task_id = task.get("id", 0)
    
    # Get image URL
    image_url = ls_client.get_image_url(task)
    if not image_url:
        return task_id, None, None, "no_image_url"
    
    # Skip invalid URLs
    if not image_url.startswith(("http://", "https://")):
        return task_id, None, None, "invalid_url"
    
    # Determine output filename
    parsed = urlparse(image_url)
    original_name = Path(parsed.path).stem
    if not original_name or original_name in ["image", "img", "file"]:
        original_name = f"task_{task_id}"
    
    # Add task_id to ensure uniqueness
    filename = f"{original_name}_{task_id}"
    
    # Determine image extension
    image_ext = Path(parsed.path).suffix or ".jpg"
    if image_ext.lower() not in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        image_ext = ".jpg"
    
    filename_with_ext = f"{filename}{image_ext}"
    
    # Download image using session for connection pooling
    image_data = download_image_to_memory(image_url, session=ls_client.session)
    if image_data is None:
        return task_id, None, filename_with_ext, "download_failed"
    
    return task_id, image_data, filename_with_ext, "success"


def process_batch(
    tasks: List[Dict[str, Any]],
    ls_client: LabelStudioClient,
    sam3_client: Optional[SAM3Client],
    images_dir: Path,
    labels_dir: Path,
    label_to_id: Dict[str, int],
    use_existing: bool,
    auto_label: bool,
    stats: Stats,
    project_labels: List[str] = None,
    preview_dir: Optional[Path] = None,
    num_workers: int = 4,
) -> None:
    """
    Process a batch of tasks with parallel downloads and batch SAM3 prediction.
    """
    # Step 1: Download all images in parallel
    downloaded = {}  # task_id -> (image_data, filename, task)
    task_map = {t.get("id", i): t for i, t in enumerate(tasks)}
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_task_download, task, ls_client): task.get("id", i)
            for i, task in enumerate(tasks)
        }
        
        for future in as_completed(futures):
            task_id, image_data, filename, status = future.result()
            
            if status == "success" and image_data is not None:
                downloaded[task_id] = (image_data, filename, task_map[task_id])
            else:
                stats.increment_skipped()
    
    if not downloaded:
        return
    
    # Step 2: Get existing annotations or prepare for auto-labeling
    annotations_map = {}  # task_id -> list of annotations
    need_auto_label = []  # list of (task_id, image_base64) for SAM3
    
    for task_id, (image_data, filename, task) in downloaded.items():
        annotations = []
        
        if use_existing:
            annotations = get_annotations_from_task(task)
        
        if annotations:
            annotations_map[task_id] = annotations
        elif auto_label and sam3_client:
            # Queue for batch auto-labeling
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            need_auto_label.append((task_id, image_base64))
        else:
            # No annotations and no auto-label - mark as skipped
            stats.increment_skipped()
    
    # Step 3: Batch SAM3 prediction for tasks needing auto-labeling
    if need_auto_label and sam3_client:
        concepts = project_labels or list(label_to_id.keys())
        try:
            sam3_results = sam3_client.predict_batch(need_auto_label, concepts)
            
            for task_id, results in sam3_results.items():
                annotations = []
                for result in results:
                    value = result.get("value", {})
                    labels = value.get("rectanglelabels", [])
                    if labels:
                        annotations.append({
                            "label": labels[0],
                            "x": value.get("x", 0),
                            "y": value.get("y", 0),
                            "width": value.get("width", 0),
                            "height": value.get("height", 0),
                        })
                
                if annotations:
                    annotations_map[task_id] = annotations
                else:
                    stats.increment_skipped()
                    
        except requests.exceptions.ConnectionError:
            for task_id, _ in need_auto_label:
                stats.increment_failed()
        except requests.exceptions.Timeout:
            for task_id, _ in need_auto_label:
                stats.increment_failed()
        except Exception as e:
            for task_id, _ in need_auto_label:
                stats.increment_failed()
    
    # Step 4: Save images and labels (can be done in parallel)
    def save_task(task_id: int) -> Tuple[bool, str]:
        if task_id not in downloaded or task_id not in annotations_map:
            return False, "missing_data"
        
        image_data, filename, task = downloaded[task_id]
        annotations = annotations_map[task_id]
        
        # Convert to YOLO format
        yolo_lines = convert_to_yolo_format(annotations, label_to_id)
        if not yolo_lines:
            return False, "no_valid_boxes"
        
        # Save image
        image_path = images_dir / filename
        with open(image_path, "wb") as f:
            f.write(image_data)
        
        # Save labels
        stem = Path(filename).stem
        label_path = labels_dir / f"{stem}.txt"
        with open(label_path, "w") as f:
            f.write("\n".join(yolo_lines))
        
        # Save preview image with bounding boxes
        if preview_dir is not None:
            preview_data = draw_preview_image(image_data, annotations, label_to_id)
            if preview_data:
                preview_path = preview_dir / f"{stem}_preview.jpg"
                with open(preview_path, "wb") as f:
                    f.write(preview_data)
        
        # Update stats
        stats.add_bytes(len(image_data))
        labels_in_task = [ann["label"] for ann in annotations if ann["label"] in label_to_id]
        stats.add_annotations(len(yolo_lines), labels_in_task)
        
        return True, "success"
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(save_task, task_id): task_id
            for task_id in annotations_map.keys()
        }
        
        for future in as_completed(futures):
            success, reason = future.result()
            if success:
                stats.increment_success()
            else:
                stats.increment_skipped()


def process_task(
    task: Dict[str, Any],
    ls_client: LabelStudioClient,
    sam3_client: Optional[SAM3Client],
    images_dir: Path,
    labels_dir: Path,
    label_to_id: Dict[str, int],
    use_existing: bool,
    auto_label: bool,
    stats: Stats,
    project_labels: List[str] = None,
    preview_dir: Optional[Path] = None,
) -> Tuple[bool, str]:
    """
    Process a single task and save image + labels.
    Returns (success, reason) tuple.
    """
    task_id = task.get("id", "unknown")
    
    # Get image URL
    image_url = ls_client.get_image_url(task)
    if not image_url:
        return False, "no_image_url"
    
    # Determine output filename early
    parsed = urlparse(image_url)
    original_name = Path(parsed.path).stem
    if not original_name or original_name in ["image", "img", "file"]:
        original_name = f"task_{task_id}"
    
    # Add task_id to ensure uniqueness
    filename = f"{original_name}_{task_id}"
    
    # Determine image extension
    image_ext = Path(parsed.path).suffix or ".jpg"
    if image_ext.lower() not in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        image_ext = ".jpg"
    
    image_path = images_dir / f"{filename}{image_ext}"
    
    # Download image first (needed for both auto-label and saving)
    image_data = None
    try:
        # Skip invalid URLs
        if not image_url.startswith(("http://", "https://")):
            return False, "invalid_url"
        resp = requests.get(image_url, headers=ls_client.headers, timeout=30)
        resp.raise_for_status()
        image_data = resp.content
        stats.total_bytes_downloaded += len(image_data)
    except Exception as e:
        return False, "download_failed"
    
    # Get annotations
    annotations = []
    
    if use_existing:
        annotations = get_annotations_from_task(task)
    
    if not annotations and auto_label and sam3_client:
        # Use project labels for auto-labeling
        concepts = project_labels or list(label_to_id.keys())
        try:
            # Convert image to base64 and send to SAM3
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            sam3_results = sam3_client.predict_with_base64(image_base64, concepts)
            for result in sam3_results:
                value = result.get("value", {})
                labels = value.get("rectanglelabels", [])
                if labels:
                    annotations.append({
                        "label": labels[0],
                        "x": value.get("x", 0),
                        "y": value.get("y", 0),
                        "width": value.get("width", 0),
                        "height": value.get("height", 0),
                    })
        except requests.exceptions.ConnectionError:
            return False, "sam3_connection_error"
        except requests.exceptions.Timeout:
            return False, "sam3_timeout"
        except Exception as e:
            return False, f"sam3_error:{str(e)[:50]}"
    
    if not annotations:
        return False, "no_annotations"
    
    # Convert to YOLO format
    yolo_lines = convert_to_yolo_format(annotations, label_to_id)
    if not yolo_lines:
        return False, "no_valid_boxes"
    
    # Save image
    with open(image_path, "wb") as f:
        f.write(image_data)
    
    # Save labels
    label_path = labels_dir / f"{filename}.txt"
    with open(label_path, "w") as f:
        f.write("\n".join(yolo_lines))
    
    # Save preview image with bounding boxes
    if preview_dir is not None:
        preview_data = draw_preview_image(image_data, annotations, label_to_id)
        if preview_data:
            preview_path = preview_dir / f"{filename}_preview.jpg"
            with open(preview_path, "wb") as f:
                f.write(preview_data)
    
    # Update stats
    labels_in_task = [ann["label"] for ann in annotations if ann["label"] in label_to_id]
    stats.add_annotations(len(yolo_lines), labels_in_task)
    
    return True, "success"


def split_dataset(
    images_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    train_split: float,
    val_split: float,
    test_split: float,
) -> Dict[str, int]:
    """Split dataset into train/val/test sets."""
    
    # Get all image files
    image_files = list(images_dir.glob("*"))
    image_files = [f for f in image_files if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]]
    
    random.shuffle(image_files)
    
    n = len(image_files)
    n_train = int(n * train_split)
    n_val = int(n * val_split)
    
    train_files = image_files[:n_train]
    val_files = image_files[n_train:n_train + n_val]
    test_files = image_files[n_train + n_val:]
    
    # Create output directories
    for split in ["train", "val", "test"]:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)
    
    def copy_files(files: List[Path], split: str):
        for img_file in files:
            # Copy image
            dst_img = output_dir / split / "images" / img_file.name
            shutil.copy2(img_file, dst_img)
            
            # Copy label
            label_file = labels_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                dst_label = output_dir / split / "labels" / label_file.name
                shutil.copy2(label_file, dst_label)
    
    copy_files(train_files, "train")
    copy_files(val_files, "val")
    copy_files(test_files, "test")
    
    return {
        "train": len(train_files),
        "val": len(val_files),
        "test": len(test_files),
    }


def create_dataset_yaml(output_dir: Path, labels: List[str], project_name: str):
    """Create dataset.yaml for YOLO training."""
    yaml_content = f"""# Dataset config for {project_name}
# Generated by prepare_yolo_dataset.py

path: {output_dir.absolute()}
train: train/images
val: val/images
test: test/images

# Classes
names:
"""
    for i, label in enumerate(labels):
        yaml_content += f"  {i}: {label}\n"
    
    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    
    print(f"{Colors.GREEN}✓ Created dataset.yaml at {yaml_path}{Colors.ENDC}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare YOLO dataset from Label Studio annotations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use existing annotations only
  python tools/prepare_yolo_dataset.py -p 1 -o ./datasets/traffic --use-existing

  # Auto-label all images with SAM3
  python tools/prepare_yolo_dataset.py -p 1 -o ./datasets/traffic --auto-label

  # Both: use existing, auto-label the rest
  python tools/prepare_yolo_dataset.py -p 1 -o ./datasets/traffic --use-existing --auto-label
        """,
    )
    
    parser.add_argument("-p", "--project-id", type=int, required=True,
                        help="Label Studio project ID")
    parser.add_argument("-o", "--output-dir", type=str, required=True,
                        help="Output directory for YOLO dataset")
    parser.add_argument("--ls-url", type=str, default=DEFAULT_LABEL_STUDIO_URL,
                        help=f"Label Studio URL (default: {DEFAULT_LABEL_STUDIO_URL})")
    parser.add_argument("--ls-token", type=str, default=DEFAULT_API_TOKEN,
                        help="Label Studio API token")
    parser.add_argument("--sam3-url", type=str, default=DEFAULT_SAM3_SERVER_URL,
                        help=f"SAM3 server URL (default: {DEFAULT_SAM3_SERVER_URL})")
    parser.add_argument("--use-existing", action="store_true",
                        help="Use existing annotations from Label Studio")
    parser.add_argument("--auto-label", action="store_true",
                        help="Auto-label images using SAM3")
    parser.add_argument("--train-split", type=float, default=0.8,
                        help="Train split ratio (default: 0.8)")
    parser.add_argument("--val-split", type=float, default=0.15,
                        help="Validation split ratio (default: 0.15)")
    parser.add_argument("--test-split", type=float, default=0.05,
                        help="Test split ratio (default: 0.05)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing output directory")
    parser.add_argument("--save-preview", action="store_true",
                        help="Save labeled preview images to ~/labeled_previews")
    parser.add_argument("--max-tasks", type=int, default=0,
                        help="Maximum number of tasks to process (0 = all)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Number of tasks to process per batch (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"Number of parallel workers for downloads (default: {DEFAULT_WORKERS})")
    parser.add_argument("--no-batch", action="store_true",
                        help="Disable batch processing (process one-by-one)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.use_existing and not args.auto_label:
        print(f"{Colors.YELLOW}⚠ Neither --use-existing nor --auto-label specified.")
        print(f"  Defaulting to --use-existing{Colors.ENDC}")
        args.use_existing = True
    
    if not args.ls_token:
        print(f"{Colors.RED}✗ Label Studio API token required. Set SAM3_LABELSTUDIO_API_TOKEN or use --ls-token{Colors.ENDC}")
        sys.exit(1)
    
    # Print banner
    print_banner()
    
    # Print configuration
    print_section("Configuration")
    print(f"""
  Project ID:      {args.project_id}
  Label Studio:    {args.ls_url}
  SAM3 Server:     {args.sam3_url}
  Output Dir:      {Path(args.output_dir).absolute()}
  Use Existing:    {'Yes' if args.use_existing else 'No'}
  Auto-Label:      {'Yes' if args.auto_label else 'No'}
  Split Ratio:     train={args.train_split}, val={args.val_split}, test={args.test_split}
  Batch Mode:      {'Disabled' if args.no_batch else f'Yes (batch_size={args.batch_size}, workers={args.workers})'}
""")
    
    if args.auto_label:
        print(f"  {Colors.BOLD}SAM3 Detection Config (from .env):{Colors.ENDC}")
        print(f"    • Score Threshold:  {SAM3_SCORE_THRESHOLD}")
        print(f"    • NMS Threshold:    {SAM3_NMS_THRESHOLD}")
        print(f"    • Min Box Area:     {SAM3_MIN_BOX_AREA}")
        print(f"    • Max Detections:   {SAM3_MAX_DETECTIONS}")
        print(f"    • Cross-Class NMS:  {SAM3_CROSS_CLASS_NMS}")
        print()
    
    # Initialize clients
    ls_client = LabelStudioClient(args.ls_url, args.ls_token)
    sam3_client = None
    
    if args.auto_label:
        sam3_client = SAM3Client(args.sam3_url)
        if not sam3_client.health_check():
            print(f"{Colors.RED}✗ SAM3 server not responding at {args.sam3_url}{Colors.ENDC}")
            print(f"  Start the server with: uvicorn app.main:app --port 8080")
            sys.exit(1)
        print(f"{Colors.GREEN}✓ SAM3 server connected{Colors.ENDC}")
    
    # Get project info
    print_section("Fetching Project Info")
    try:
        project = ls_client.get_project(args.project_id)
        print(f"{Colors.GREEN}✓ Connected to project: {project.get('title', 'Unknown')}{Colors.ENDC}")
    except Exception as e:
        print(f"{Colors.RED}✗ Failed to get project: {e}{Colors.ENDC}")
        sys.exit(1)
    
    # Parse labels from config
    label_config = project.get("label_config", "")
    labels = parse_labels_from_config(label_config)
    
    if not labels:
        print(f"{Colors.RED}✗ No labels found in project config{Colors.ENDC}")
        sys.exit(1)
    
    print(f"{Colors.GREEN}✓ Found {len(labels)} labels (will use these for auto-labeling):{Colors.ENDC}")
    for i, label in enumerate(labels):
        print(f"    {i}: {label}")
    
    label_to_id = {label: i for i, label in enumerate(labels)}
    
    # Fetch all tasks with progress bar
    print_section("Fetching Tasks from Label Studio")
    try:
        total_tasks = ls_client.get_task_count(args.project_id)
        print(f"{Colors.CYAN}ℹ Project has {total_tasks} total tasks{Colors.ENDC}")
        
        if args.max_tasks > 0:
            print(f"{Colors.CYAN}ℹ Will fetch up to {args.max_tasks} tasks (--max-tasks){Colors.ENDC}")
        
        print(f"{Colors.CYAN}ℹ Fetching tasks...{Colors.ENDC}")
        all_tasks = ls_client.get_all_tasks(args.project_id, max_tasks=args.max_tasks, show_progress=True)
        print(f"{Colors.GREEN}✓ Fetched {len(all_tasks)} tasks successfully!{Colors.ENDC}")
        
        total_tasks = len(all_tasks)
        if args.max_tasks > 0 and total_tasks > args.max_tasks:
            all_tasks = all_tasks[:args.max_tasks]
            total_tasks = args.max_tasks
            print(f"{Colors.YELLOW}  (limited to {args.max_tasks} by --max-tasks){Colors.ENDC}")
            
    except Exception as e:
        print(f"{Colors.RED}✗ Failed to fetch tasks: {e}{Colors.ENDC}")
        sys.exit(1)
    
    if total_tasks == 0:
        print(f"{Colors.YELLOW}⚠ No tasks found in project{Colors.ENDC}")
        sys.exit(0)
    
    # Prepare output directory
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        if args.force:
            shutil.rmtree(output_dir)
        else:
            print(f"{Colors.RED}✗ Output directory exists. Use --force to overwrite{Colors.ENDC}")
            sys.exit(1)
    
    # Create preview directory if requested
    preview_dir = None
    if args.save_preview:
        preview_dir = Path.home() / "labeled_previews"
        if preview_dir.exists():
            shutil.rmtree(preview_dir)
        preview_dir.mkdir(parents=True, exist_ok=True)
        print(f"{Colors.GREEN}✓ Preview images will be saved to: {preview_dir}{Colors.ENDC}")
        
        if not PIL_AVAILABLE:
            print(f"{Colors.YELLOW}⚠ PIL/Pillow not installed. Install with: pip install Pillow{Colors.ENDC}")
            preview_dir = None
    
    # Create temp directories for processing
    temp_images_dir = output_dir / "_temp_images"
    temp_labels_dir = output_dir / "_temp_labels"
    temp_images_dir.mkdir(parents=True, exist_ok=True)
    temp_labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Process tasks
    print_section("Processing Tasks" + (" (Batch Mode)" if not args.no_batch else " (Sequential)"))
    if args.auto_label:
        concepts_preview = ", ".join(labels[:5])
        if len(labels) > 5:
            concepts_preview += "..."
        print(f"{Colors.CYAN}ℹ Auto-labeling with {len(labels)} concepts: {concepts_preview}{Colors.ENDC}")
    
    if not args.no_batch:
        print(f"{Colors.CYAN}ℹ Batch size: {args.batch_size}, Workers: {args.workers}{Colors.ENDC}")
    
    stats = Stats()
    stats.total_tasks = total_tasks
    
    print()  # Empty line for progress bar
    
    if args.no_batch:
        # Legacy one-by-one processing (uses pre-fetched tasks)
        tasks_processed = 0
        for task in all_tasks:
            task_id = task.get('id', 'unknown')
            
            success, reason = process_task(
                task,
                ls_client,
                sam3_client,
                temp_images_dir,
                temp_labels_dir,
                label_to_id,
                args.use_existing,
                args.auto_label,
                stats,
                project_labels=labels,
                preview_dir=preview_dir,
            )
            
            tasks_processed += 1
            stats.processed += 1
            if success:
                stats.success += 1
            elif reason in ("no_annotations", "no_valid_boxes", "no_image_url", "download_failed", "invalid_url"):
                stats.skipped += 1
            elif reason and reason.startswith("sam3_"):
                # SAM3 related errors - these are failures, not skips
                stats.failed += 1
                print(f"\n  {Colors.RED}✗ Task {task_id}: {reason}{Colors.ENDC}")
            else:
                stats.failed += 1
                if reason:
                    print(f"\n  {Colors.RED}✗ Task {task_id}: {reason}{Colors.ENDC}")
            
            stats.print_progress(task_id)
    else:
        # Batch processing mode - process pre-fetched tasks in chunks
        batch_num = 0
        batch_size = args.batch_size
        
        # Split all_tasks into batches
        for i in range(0, len(all_tasks), batch_size):
            batch = all_tasks[i:i + batch_size]
            batch_num += 1
            batch_info = f"Batch {batch_num} ({len(batch)} tasks)"
            
            process_batch(
                batch,
                ls_client,
                sam3_client,
                temp_images_dir,
                temp_labels_dir,
                label_to_id,
                args.use_existing,
                args.auto_label,
                stats,
                project_labels=labels,
                preview_dir=preview_dir,
                num_workers=args.workers,
            )
            
            stats.print_progress(batch_info=batch_info)
    
    print()  # New line after progress bar
    
    # Print processing summary
    stats.print_summary()
    
    # Split dataset
    print_section("Splitting Dataset")
    counts = split_dataset(
        temp_images_dir,
        temp_labels_dir,
        output_dir,
        args.train_split,
        args.val_split,
        args.test_split,
    )
    
    # Clean up temp directories
    shutil.rmtree(temp_images_dir, ignore_errors=True)
    shutil.rmtree(temp_labels_dir, ignore_errors=True)
    
    # Create dataset.yaml
    create_dataset_yaml(output_dir, labels, project.get("title", "yolo_dataset"))
    
    # Print final summary
    print_section("Dataset Ready!")
    print(f"""
  Output Directory: {output_dir.absolute()}
  
  Dataset Split:
    • Train: {counts['train']} images
    • Val:   {counts['val']} images  
    • Test:  {counts['test']} images
    • Total: {sum(counts.values())} images
  
  Classes ({len(labels)}):""")
    
    for i, label in enumerate(labels):
        print(f"    {i}: {label}")
    
    print(f"""
  {Colors.GREEN}To train YOLO:{Colors.ENDC}
  yolo detect train data={output_dir.absolute()}/dataset.yaml model=yolov8n.pt epochs=100
""")


if __name__ == "__main__":
    main()
