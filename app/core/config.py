from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    checkpoint_path: str | None = Field(
        default=None,
        description="Optional local path to a SAM3 checkpoint. When omitted the official HF weights are downloaded automatically.",
    )
    device: str = Field(default="cuda", description="Device to run SAM3 on (cuda or cpu).")
    score_threshold: float = Field(default=0.5, description="Minimum confidence score to keep SAM3 detections.")
    max_detections: int = Field(default=25, description="Maximum number of detections returned per frame.")
    rtsp_timeout: int = Field(default=5, description="Seconds to wait when connecting to RTSP stream.")
    frame_skip: int = Field(
        default=1,
        description="Number of frames to skip before sampling the next frame for detection.",
    )
    max_frames: int = Field(default=3, description="Maximum frames sampled from the RTSP stream per request.")
    max_image_size: int = Field(
        default=1024,
        description="Maximum image dimension (width or height). Larger images are resized for faster inference.",
    )
    nms_threshold: float = Field(
        default=0.3,
        description="IoU threshold for Non-Maximum Suppression. Lower = more aggressive suppression.",
    )
    cross_class_nms: bool = Field(
        default=True,
        description="Apply NMS across different classes to remove duplicate detections on same object.",
    )
    min_box_area: int = Field(
        default=500,
        description="Minimum bounding box area in pixels. Smaller boxes are filtered as false positives.",
    )
    concepts_path: str | None = Field(
        default=None,
        description="Optional path to a newline-delimited list of concept prompts that override the built-in defaults.",
    )
    extra_concepts: str | None = Field(
        default=None,
        description="Comma-separated prompts appended after the default or file-based concept list.",
    )
    use_default_concepts: bool = Field(
        default=True,
        description="Include built-in COCO + safety concept prompts unless an override file is supplied.",
    )
    max_concepts_per_request: int = Field(
        default=160,
        description="Maximum number of concept prompts allowed per API request.",
    )
    hf_token: str | None = Field(
        default=None,
        description="Optional Hugging Face access token used when downloading SAM3 checkpoints.",
    )
    labelstudio_from_name: str = Field(
        default="label",
        description="Label Studio control tag 'from_name' used when emitting rectanglelabels results.",
    )
    labelstudio_to_name: str = Field(
        default="image",
        description="Label Studio object tag 'to_name' that receives rectangle predictions.",
    )
    labelstudio_model_version: str = Field(
        default="sam3-v0.1",
        description="Version string advertised back to Label Studio clients.",
    )
    labelstudio_api_base: str | None = Field(
        default=None,
        description="Base URL of the Label Studio instance (used for future integrations).",
    )
    labelstudio_api_token: str | None = Field(
        default=None,
        description="Personal access token for the Label Studio API.",
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        env_prefix = "SAM3_"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
