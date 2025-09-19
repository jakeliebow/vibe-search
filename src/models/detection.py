import numpy as np

from typing import Optional, List, Any
from pydantic import BaseModel, Field
from src.models.embedding import Embedding


class ImageSample(BaseModel):
    """
    Represents a sample image from a detection.

    This model stores an image sample along with metadata about when it was captured
    and the confidence level of the detection.

    Attributes:
        frame_index (int): Zero-indexed frame number of sample image
        confidence (float): YOLO confidence score for this detection
        image_data (np.ndarray): Raw image array data of the sample
    """

    frame_index: int = Field(
        ..., ge=0, description="Zero-indexed frame number of sample image"
    )
    confidence: float = Field(..., description="confidence of image")
    image_data: np.ndarray = Field(..., description="image array of sample image")
    class Config:
        arbitrary_types_allowed = True


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
    embedding: Optional[Embedding] = Field(
        None, description="Compressed facial signature for recognition"
    )
    normalized_landmarks: np.ndarray = Field(
        ..., description="Normalized landmark coordinates"
    )
    mar: float = Field(..., description="Mouth aspect ratio for expression analysis")
    mar_derivative: Optional[float] = (
        Field(  # CRITICAL FOR ANNOYING JOE GUY AND THE PIPELINE
            ...,
            description="Mouth aspect ratio derivative for expression analysis like isTalking",
        )
    )

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
        image (Any): Detected cropped image
        class_type (int): type
        
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
    image: Any = Field(
        ..., description="Detected cropped image of the object"
    )
    class_type: int = Field(
        ..., description="type"
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
    yolo_uuid: Optional[str] = Field(
        ..., description="Global Unique identifier assigned by the tracking algorithm"
    )
    yolo_object_id: Optional[str] = Field(
        ..., description="Unique identifier assigned by the tracking algorithm"
    )


class ObjectTrack(BaseModel):
    """
    Represents a tracked object across multiple frames in a video.

    This model aggregates individual detections of the same object over time,
    allowing for analysis of its movement and behavior throughout the video.

    Attributes:
        yolo_object_id (str): Unique identifier assigned by the tracking algorithm
        detections (List[Detection]): List of Detection instances for this object
        type (str): Type of the detected object, e.g., 'person', 'car', etc.
        face_embeddings (Optional[List[np.ndarray]]): List of face embeddings associated with this object
        sample (ImageSample): Sample image data for this object
        action (Any): Action stuff
    """

    yolo_object_id: str = Field(
        ..., description="Unique identifier assigned by the tracking algorithm"
    )
    detections: List[Detection] = Field(
        ..., description="List of Detection instances for this object"
    )
    object_type: str = Field(
        ..., description="Type of the detected object, e.g., 'person', 'car', etc."
    )
    face_embeddings: Optional[List[np.ndarray]] = Field(..., description="List of face embeddings associated with this object"
    )
    sample: ImageSample = Field(..., description="sample image object data")
    action: Optional[Any] = Field(
        default_factory=lambda _:[], description="Action classifier stuff"
    )

    class Config:
        arbitrary_types_allowed = True


class MarAtIndex(BaseModel):
    """
    Represents the Mouth Aspect Ratio (MAR) at a specific frame index.

    This model captures the MAR value, which is useful for analyzing facial expressions,
    particularly in the context of detecting yawns or other mouth movements.

    Attributes:
        frame_index (int): Zero-indexed frame number where the MAR was calculated
        mar (float): Mouth aspect ratio value at the specified frame
    """

    frame_index: int = Field(
        ..., ge=0, description="Zero-indexed frame number where the MAR was calculated"
    )
    mar: float = Field(
        ..., description="Mouth aspect ratio value at the specified frame"
    )
