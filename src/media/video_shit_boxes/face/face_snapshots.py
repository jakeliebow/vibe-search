#!/usr/bin/env python3
"""
Face snapshot extraction using high confidence person detections with timestamps.
Uses RecognizedObject timestamps to get face snapshots at specific moments.
"""

import cv2
import numpy as np
from typing import Optional

from src.models.detection import  FaceData, BoundingBox
from src.state.AI.face_models import detector, predictor
from src.media.video_shit_boxes.face.mar import calculate_mar
from src.media.video_shit_boxes.misc.image_helpers import ensure_gray_scale


def head_focused_upscale(
    person_crop: np.ndarray,
    head_ratio: float = 0.7,
    min_width: int = 300,
    min_height: int = 300,
    top_pad_ratio: float = 0.1,
    interpolation: int = cv2.INTER_CUBIC,
) -> tuple[np.ndarray, float, int, int]:
    """
    Extract the head-focused region from a person crop (top portion), then upscale to
    a minimum size while preserving aspect ratio. Only upscales; never downscales.

    Args:
        person_crop: Numpy image (grayscale or BGR) of the person bounding box.
        head_ratio: Fraction of height from the top to include (e.g., 0.7 = top 70%).
        min_width: Minimum output width in pixels. If <= current width, no width upscale.
        min_height: Minimum output height in pixels. If <= current height, no height upscale.
        top_pad_ratio: Additional fraction of height to include beyond head_ratio from the top.
        interpolation: OpenCV interpolation method for resizing.

    Returns:
        Tuple of (image, scale, x_offset, y_offset) where image is the head-focused crop
        possibly upscaled, and the scale/offset describe how to map image coords back to
        the original person_crop coords: orig = (resized/scale) + (x_offset, y_offset)
    """
    if person_crop is None:
        raise ValueError("head_focused_upscale: person_crop is None")

    if not hasattr(person_crop, "shape") or len(person_crop.shape) < 2:
        raise ValueError(f"head_focused_upscale: invalid image shape {getattr(person_crop, 'shape', None)}")

    h, w = person_crop.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("head_focused_upscale: person_crop has zero height or width")

    # Determine the vertical extent to capture the head region from the top
    head_h = int(round(h * head_ratio))
    pad = int(round(h * top_pad_ratio))
    y_end = min(h, head_h + pad)
    if y_end <= 0:
        return person_crop, 1.0, 0, 0

    head_crop = person_crop[0:y_end, 0:w]

    # Check if upscale is needed
    if w >= min_width and y_end >= min_height:
        return head_crop, 1.0, 0, 0

    # Compute scale to meet both constraints, preserving aspect ratio
    scale = max(min_width / float(w), min_height / float(y_end))
    new_w = int(round(w * scale))
    new_h = int(round(y_end * scale))

    # Ensure we don't end up just shy of the minimum due to rounding
    if new_w < min_width:
        new_w = min_width
    if new_h < min_height:
        new_h = min_height

    resized = cv2.resize(head_crop, (new_w, new_h), interpolation=interpolation)
    return resized, scale, 0, 0






def run_dlib_on_person_to_get_face_data(detected_cropped_image) -> Optional[FaceData]:

    # Detect faces in the cropped person image
    detected_cropped_image_gray = ensure_gray_scale(detected_cropped_image)
    faces = detector(detected_cropped_image_gray)
    if not faces:
        return None

    # Get image dimensions
    image_height, image_width = detected_cropped_image.shape[:2]

    # Calculate scores for all faces and find the best one
    face_scores = []
    for face in faces:
        score = _calculate_face_score(face, image_width, image_height)
        face_scores.append((face, score))

    # Sort by score (highest first) and take the best face
    face_scores.sort(key=lambda x: x[1], reverse=True)
    best_face = face_scores[0][0]

    # Process only the best face
    face = best_face

    # Get facial landmarks
    landmarks = predictor(detected_cropped_image, face)

    # Convert landmarks to numpy array
    landmarks_array = np.array([[p.x, p.y] for p in landmarks.parts()])
    # Create squeezed signature by taking key landmark distances and ratios
    # This creates a more compact representation than full landmarks
    # Create squeezed signature from normalized landmarks
    # Let the model figure out what's important rather than hand-crafting features
    face_width = face.right() - face.left()
    face_height = face.bottom() - face.top()

    # Normalize landmarks relative to face bounding box and flatten
    normalized_landmarks = landmarks_array.copy().astype(np.float32)
    normalized_landmarks[:, 0] = (normalized_landmarks[:, 0] - face.left()) / face_width  # x coords
    normalized_landmarks[:, 1] = (normalized_landmarks[:, 1] - face.top()) / face_height  # y coords

    # Compute robust 128D embedding using dlib face recognition model
    


    mar = calculate_mar(landmarks_array)

    # Create FaceData model instance
    face_data_instance = FaceData(
        face_box=BoundingBox(x1=face.left(), y1=face.top(), x2=face.right(), y2=face.bottom()),
        landmarks=landmarks_array,
        normalized_landmarks=normalized_landmarks,
        mar=mar
    )

    return face_data_instance
def _calculate_face_score(face, image_width: int, image_height: int, confidence_weight: float = 0.7, center_weight: float = 0.3) -> float:
    """
    Calculate a composite score for face selection based on confidence and centeredness.

    Args:
        face: dlib face detection object
        image_width: Width of the image
        image_height: Height of the image
        confidence_weight: Weight for confidence score (0-1)
        center_weight: Weight for centeredness score (0-1)

    Returns:
        float: Composite score (higher is better)
    """
    # Get face center
    face_center_x = (face.left() + face.right()) / 2
    face_center_y = (face.top() + face.bottom()) / 2

    # Get image center
    image_center_x = image_width / 2
    image_center_y = image_height / 2

    # Calculate distance from center (normalized by image diagonal)
    distance_from_center = np.sqrt((face_center_x - image_center_x)**2 + (face_center_y - image_center_y)**2)
    max_distance = np.sqrt(image_width**2 + image_height**2) / 2
    normalized_distance = distance_from_center / max_distance

    # Centeredness score (1.0 = perfectly centered, 0.0 = at corner)
    centeredness_score = 1.0 - normalized_distance

    # For dlib, we don't have direct confidence, so we use face size as a proxy
    # Larger faces are typically more confident detections
    face_area = (face.right() - face.left()) * (face.bottom() - face.top())
    image_area = image_width * image_height
    size_score = min(face_area / (image_area * 0.1), 1.0)  # Cap at 1.0, normalize by 10% of image

    # Composite score
    composite_score = (size_score * confidence_weight) + (centeredness_score * center_weight)

    return composite_score