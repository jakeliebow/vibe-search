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
    YoloObjectTrack,
    ImageSample,
)
from uuid import uuid4
from src.media.video.face.face_embeddings import (
    compute_face_embedding_from_rect,

)
from src.models.embedding import Embedding
from src.media.video.misc.image_helpers import (
    get_cropped_image_by_detection_bounded_box
)
from src.media.video.face.main import (
    get_face_data_from_person_detection,
)
from diskcache import Cache

cache = Cache("/tmp/yolo___")


# @cache.memoize()
def run_yolo(Frame, tracker, identities, processed_frames, name_map, detection):
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

    if detection.yolo_object_id:

        bb_image_data = Frame.image_data[
                        detection.box.y1 : detection.box.y2,
                        detection.box.x1 : detection.box.x2,
                    ]

        if detection.yolo_object_id not in tracker:
            id = str(uuid4())
            tracker[detection.yolo_object_id] = id
            identities[id] = YoloObjectTrack(
                face_embeddings=[],
                yolo_object_id=detection.yolo_object_id,
                detections=[],
                object_type=name_map[detection.class_type],
                sample=ImageSample(
                    confidence=detection.confidence,
                    frame_index=Frame.frame_number,
                    image_data=bb_image_data
                )
            )
        yolo_uuid = tracker[detection.yolo_object_id]
        if identities[yolo_uuid].sample.confidence < detection.confidence:
            identities[yolo_uuid].sample = ImageSample(
                confidence=detection.confidence,
                frame_index=Frame.frame_number,
                image_data=bb_image_data
            )
        detection.yolo_uuid = yolo_uuid
        identities[yolo_uuid].detections.append(detection)
    Frame.detections.append(detection)
    processed_frames.append(Frame)