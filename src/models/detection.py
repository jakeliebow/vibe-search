import numpy as np

from typing import Optional
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """
    Represents a rectangular bounding box in 2D coordinate space.

    Uses the standard computer vision convention where (0,0) is the top-left corner,
    x increases rightward, and y increases downward. Coordinates are stored as
    integer pixel values.

    Attributes:
        x1 (int): Left edge x-coordinate of the bounding box
        y1 (int): Top edge y-coordinate of the bounding box
        x2 (int): Right edge x-coordinate of the bounding box
        y2 (int): Bottom edge y-coordinate of the bounding box
    """

    x1: int = Field(..., description="Left edge x-coordinate of the bounding box")
    y1: int = Field(..., description="Top edge y-coordinate of the bounding box")
    x2: int = Field(..., description="Right edge x-coordinate of the bounding box")
    y2: int = Field(..., description="Bottom edge y-coordinate of the bounding box")
    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2.0




class FaceData(BaseModel):
    """
    Represents facial detection data including bounding box, landmarks, and analysis metrics.

    Attributes:
        face_box (BoundingBox): Face bounding box (left=x1, top=y1, right=x2, bottom=y2)
        landmarks (np.ndarray): Array of facial landmark coordinates
        squeezed_signature (np.ndarray | None): Compressed facial signature for recognition
        normalized_landmarks (np.ndarray): Normalized landmark coordinates
        mar (float): Mouth aspect ratio for expression analysis
    """

    face_box: BoundingBox = Field(..., description="Face bounding box (x1,y1,x2,y2)")
    landmarks: np.ndarray = Field(
        ..., description="Array of facial landmark coordinates"
    )
    embedding: Optional[np.ndarray] = Field(
        None, description="Compressed facial signature for recognition"
    )
    normalized_landmarks: np.ndarray = Field(
        ..., description="Normalized landmark coordinates"
    )
    mar: float = Field(..., description="Mouth aspect ratio for expression analysis")

    class Config:
        arbitrary_types_allowed = True


class Detection(BaseModel):
    """
    Represents a single object detection result in a video frame.

    This model captures the essential information about a detected object,
    including its spatial location, temporal context, and detection confidence.
    Used as a building block for tracking objects across multiple frames.

    Attributes:
        detection_id (str): random id to refer to this detection in this frame, not a true object id
        box (BoundingBox): Spatial coordinates of the detected object
        frame (int): Zero-indexed frame number where detection occurred
        timestamp (int): Approximate Timestamp in seconds when detection was made
        confidence (float): Detection confidence score between 0.0 and 1.0
    """
    
    detection_id: str = Field(
        ...,
        description="random id to refer to this detection in this frame, not a true object id",
    )
    box: BoundingBox = Field(
        ..., description="Spatial coordinates of the detected object"
    )
    frame_number: int = Field(
        ..., ge=0, description="Zero-indexed frame number where detection occurred"
    )
    timestamp: float = Field(
        ...,
        ge=0.0,
        description="Approximate Timestamp in seconds when detection was made",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Detection confidence score between 0.0 and 1.0",
    )
    recognized_object_type: str = Field(
        ..., description="Human-readable classification of the object"
    )

    face: Optional[FaceData] = Field(
        ..., description="Human-readable classification of the object"
    )
