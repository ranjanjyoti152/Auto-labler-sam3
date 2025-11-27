from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field, field_validator


class DetectRequest(BaseModel):
    rtsp_url: str = Field(..., description="RTSP endpoint to sample frames from.")
    max_frames: int | None = Field(
        default=None,
        gt=0,
        le=30,
        description="Optional override for the number of frames pulled per request.",
    )
    frame_skip: int | None = Field(
        default=None,
        ge=1,
        le=600,
        description="Optional override for how many frames to skip between samples.",
    )
    concepts: List[str] | None = Field(
        default=None,
        description="Optional list of text prompts (concepts) to detect. Defaults to the server's configured COCO + safety prompts.",
    )

    @field_validator("rtsp_url")
    @classmethod
    def validate_rtsp_url(cls, value: str) -> str:
        if not value.startswith("rtsp://"):
            raise ValueError("Only RTSP urls starting with rtsp:// are supported")
        return value

    @field_validator("concepts")
    @classmethod
    def validate_concepts(cls, value: List[str] | None) -> List[str] | None:
        if value is None:
            return value
        cleaned = [concept.strip() for concept in value if concept.strip()]
        if not cleaned:
            raise ValueError("At least one non-empty concept must be provided when overriding prompts")
        return cleaned


class BoundingBox(BaseModel):
    x: int
    y: int
    width: int
    height: int


class DetectionResult(BaseModel):
    frame_index: int
    label: str
    bbox: BoundingBox
    area: int
    score: float


class DetectResponse(BaseModel):
    frames_analyzed: int
    detections: List[DetectionResult]


class LabelStudioTask(BaseModel):
    id: str | int | None = None
    data: dict = Field(default_factory=dict, description="Raw task payload from Label Studio.")


class LabelStudioPredictRequest(BaseModel):
    tasks: List[LabelStudioTask] = Field(
        ..., min_length=1, description="Batch of Label Studio tasks to score."
    )
