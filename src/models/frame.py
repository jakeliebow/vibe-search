import numpy as np

from typing import Optional,List
from pydantic import BaseModel, Field

class Frame(BaseModel):
    """
    Represents a single video frame with associated metadata.

    Attributes:
        frame_number (int): The sequential number of the frame in the video.
        timestamp (float): The timestamp of the frame in seconds.
        image (np.ndarray): The image data of the frame as a NumPy array.
    """

    frame_number: int = Field(..., description="The sequential number of the frame in the video.")
    timestamp: float = Field(..., description="The timestamp of the frame in seconds.")
    detections: List = Field(default=[], description="List of detections in this frame")
    diarized_audio_segments: List = Field(default=[], description="List of diarized audio segments overlapping with this frame")

    class Config:
        arbitrary_types_allowed = True