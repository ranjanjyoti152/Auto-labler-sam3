from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np


class RtspStreamError(RuntimeError):
    """Raised when frames cannot be sampled from the RTSP stream."""


@dataclass(slots=True)
class SampledFrame:
    index: int
    image: np.ndarray


class RtspFrameSampler:
    def __init__(self, rtsp_url: str, timeout_seconds: int = 5) -> None:
        self.rtsp_url = rtsp_url
        self.timeout_seconds = timeout_seconds

    def sample(self, max_frames: int, frame_skip: int) -> List[SampledFrame]:
        if frame_skip < 1:
            raise ValueError("frame_skip must be >= 1")
        if max_frames < 1:
            raise ValueError("max_frames must be >= 1")
        capture = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        if not capture.isOpened():
            capture.release()
            raise RtspStreamError(f"Could not open RTSP stream {self.rtsp_url}")

        sampled: List[SampledFrame] = []
        frame_index = 0
        deadline = time.monotonic() + self.timeout_seconds

        try:
            while len(sampled) < max_frames:
                if time.monotonic() > deadline:
                    raise RtspStreamError("Timed out while reading RTSP stream")

                ok, frame = capture.read()
                if not ok or frame is None:
                    break

                if frame_index % frame_skip == 0:
                    sampled.append(SampledFrame(index=frame_index, image=frame.copy()))

                frame_index += 1
        finally:
            capture.release()

        if not sampled:
            raise RtspStreamError("Failed to sample any frames from the RTSP stream")

        return sampled
