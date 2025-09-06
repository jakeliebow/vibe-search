#!/usr/bin/env python3
"""
Face embedding utilities using dlib's 128D face recognition model.

These helpers compute robust face embeddings suitable for RAG storage and
similarity search, analogous in spirit to the audio embeddings you use via
voice_embedding_model.encode_batch(...).

They are not wired into the pipeline; import and call them where needed.
"""
from typing import Optional, Tuple, List

import cv2
import numpy as np
import dlib

from src.models.objects import BoundingBox
from src.utils.AI_models import detector, predictor, face_recognition_model
from src.processing.video_shit_boxes.misc.image_helpers import (
    ensure_rgb,
    ensure_gray_scale,
)


def compute_face_embedding_from_rect(
    image: np.ndarray, face_rect: BoundingBox
) -> np.ndarray:
    """
    Compute a 128D face embedding given an image and a face rectangle.

    Steps:
    - If image is BGR, convert to RGB (dlib expects RGB order)
    - Predict 68 landmarks
    - Optionally align to a face chip for stability
    - Compute 128D embedding with face_recognition_model

    Returns:
    - np.ndarray shape (128,), dtype float64
    """
    rgb = ensure_rgb(image)
    face_rect = dlib.rectangle(face_rect.x1, face_rect.y1, face_rect.x2, face_rect.y2)
    shape = predictor(rgb, face_rect)

    # Align to a standard face chip for better invariance
    face_chip = dlib.get_face_chip(rgb, shape, size=150)
    # Re-run landmarks on the aligned chip (optional but typical)
    shape_chip = predictor(
        face_chip, dlib.rectangle(0, 0, face_chip.shape[1], face_chip.shape[0])
    )

    # Compute embedding
    descriptor = face_recognition_model.compute_face_descriptor(face_chip, shape_chip)
    embedding = np.array(descriptor, dtype=np.float32)
    return embedding


def compute_best_face_embedding(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Detect faces and return the embedding for the best (largest) face.

    Returns None if no face is detected.
    """
    # dlib accepts RGB or grayscale. We'll let detector work on grayscale for robustness.
    gray = ensure_gray_scale(image)

    faces = detector(gray)
    if len(faces) == 0:
        return None

    def area(r: dlib.rectangle) -> int:
        return (r.right() - r.left()) * (r.bottom() - r.top())

    best = max(faces, key=area)
    return compute_face_embedding_from_rect(image, best)
