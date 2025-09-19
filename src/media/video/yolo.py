#!/usr/bin/env python3
"""
Yolo Based object processing - gives you list of frames where each frame has its own lists of detection objects, based on how many objects in this frame
"""

from typing import Dict, List, Tuple, Optional
from src.state.AI.object_detection_models import yolo_model
from src.models.frame import Frame
from src.models.detection import (
    BoundingBox,
    Detection,
    FaceData,
    ObjectTrack,
    ImageSample,
)
from uuid import uuid4
from src.media.video.face.face_embeddings import (
    compute_face_embedding_from_rect,

)
from src.models.embedding import Embedding
from src.media.video.misc.image_helpers import (
    get_detection_image
)
from src.media.video.face.main import (
    get_face_data_from_person_detection,
)
from diskcache import Cache

cache = Cache("/tmp/yolo___")


# @cache.memoize()
def process_yolo_detection(Frame, tracker, identities, processed_frames, name_map, detection):
    if name_map[detection.class_type] == "person":
        face_data_from_detection: Optional[FaceData] = (
            get_face_data_from_person_detection(
                detection
            )
        )

        if face_data_from_detection:
            face_data_from_detection.embedding = Embedding(
                embedding=compute_face_embedding_from_rect(
                    Frame.image_data, face_data_from_detection.face_box
                ),
                image_data=detection.image
            )
            detection.face = face_data_from_detection

    Frame.detections.append(detection)
    processed_frames.append(Frame)
    