"""Label Studio API integration for fetching project labels."""
from __future__ import annotations

import re
import time
from typing import List, Set, Tuple
import requests

from app.core.config import Settings


# Cache storage: (labels, timestamp)
_labels_cache: Tuple[frozenset[str], float] | None = None
_CACHE_TTL_SECONDS = 3600  # 1 hour


def parse_label_config(label_config: str) -> Set[str]:
    """Extract label values from Label Studio XML config."""
    labels: Set[str] = set()
    
    # Match <Label value="..."/> tags
    label_pattern = re.compile(r'<Label\s+[^>]*value\s*=\s*["\']([^"\']+)["\']', re.IGNORECASE)
    for match in label_pattern.finditer(label_config):
        labels.add(match.group(1))
    
    # Also match Choice tags for classification
    choice_pattern = re.compile(r'<Choice\s+[^>]*value\s*=\s*["\']([^"\']+)["\']', re.IGNORECASE)
    for match in choice_pattern.finditer(label_config):
        labels.add(match.group(1))
    
    return labels


def fetch_project_labels(settings: Settings, project_id: int | None = None) -> Set[str]:
    """Fetch all labels from Label Studio project(s)."""
    if not settings.labelstudio_api_base or not settings.labelstudio_api_token:
        return set()
    
    api_base = settings.labelstudio_api_base.rstrip("/")
    headers = {"Authorization": f"Token {settings.labelstudio_api_token}"}
    
    all_labels: Set[str] = set()
    
    try:
        if project_id:
            # Fetch specific project
            url = f"{api_base}/api/projects/{project_id}/"
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            project = resp.json()
            label_config = project.get("label_config", "")
            all_labels.update(parse_label_config(label_config))
        else:
            # Fetch all projects
            url = f"{api_base}/api/projects/"
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            
            projects = data.get("results", []) if isinstance(data, dict) else data
            for project in projects:
                label_config = project.get("label_config", "")
                all_labels.update(parse_label_config(label_config))
    except Exception as e:
        print(f"[WARNING] Failed to fetch Label Studio labels: {e}")
    
    return all_labels


def get_labelstudio_concepts(settings: Settings) -> List[str]:
    """Get concepts from Label Studio, with time-based caching (1 hour TTL)."""
    global _labels_cache
    
    if not settings.labelstudio_api_base or not settings.labelstudio_api_token:
        return []
    
    current_time = time.time()
    
    # Check if cache is valid
    if _labels_cache is not None:
        cached_labels, cached_time = _labels_cache
        if current_time - cached_time < _CACHE_TTL_SECONDS:
            return list(cached_labels)
        else:
            print(f"[INFO] Label Studio labels cache expired, refreshing...")
    
    try:
        labels = fetch_project_labels(settings)
        _labels_cache = (frozenset(labels), current_time)
        print(f"[INFO] Cached {len(labels)} labels from Label Studio")
        return list(labels)
    except Exception as e:
        print(f"[WARNING] Failed to get Label Studio concepts: {e}")
        # Return cached labels if available, even if expired
        if _labels_cache is not None:
            return list(_labels_cache[0])
        return []


def clear_labels_cache():
    """Clear the cached labels (call this when you want to force refresh)."""
    global _labels_cache
    _labels_cache = None
    print("[INFO] Label Studio labels cache cleared")
