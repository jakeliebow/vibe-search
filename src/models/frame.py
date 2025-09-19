import numpy as np

from typing import Optional, List
from pydantic import BaseModel, Field


class Frame(BaseModel):
    """
    Represents a single video frame with associated metadata.

    Attributes:
        frame_number (int): The sequential number of the frame in the video.
        timestamp (float): The timestamp of the frame in seconds.
        image_data (np.ndarray): The image data of the frame as a NumPy array.
    """

    frame_number: int = Field(
        ..., description="The sequential number of the frame in the video."
    )
    timestamp: float = Field(..., description="The timestamp of the frame in seconds.")
    detections: List = Field(default=[], description="List of detections in this frame")
    diarized_audio_segments: List = Field(
        default_factory=lambda _:[],
        description="List of diarized audio segments overlapping with this frame",
    )
    transcribed_audio_segments: List = Field(
        default_factory=lambda _:[],
        description="List of diarized audio segments overlapping with this frame",
    )
    video_path:str =Field(..., description='just the string path to video file')
    image_data: np.ndarray = Field(..., description="2d nd array of pixel data")
    class Config:
        arbitrary_types_allowed = True
