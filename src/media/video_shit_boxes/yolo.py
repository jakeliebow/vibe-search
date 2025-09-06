#!/usr/bin/env python3
"""
Yolo Based object processing - gives you list of frames where each frame has its own lists of detection objects, based on how many objects in this frame
"""

from typing import Dict, List

from pydantic import UUID4
from src.AI.object_detection_models import yolo_model
from src.models.detection import BoundingBox, Detection, FaceData
from src.utils.cache import cache
import cv2
from uuid import uuid4
from src.media.video_shit_boxes.face.face_embeddings import (
    compute_face_embedding_from_rect,
)
from typing import List, Optional
from src.media.video_shit_boxes.misc.image_helpers import (
    get_frame_image,
    get_cropped_image_by_detection_bounded_box,
)
from src.media.video_shit_boxes.face.main import (
    get_face_data_from_person_detection,
)

from src.utils.cache import cache


@cache.memoize()
def process_yolo_boxes_to_get_inferenced_detections(
    yolo_tagged_frames,
    video_path="/Users/jakeliebow/milli/tests/test_data/chunks/test_chunk_000.mp4",
) -> List[List[Detection]]:

    inferenced_frames = []
    for frame_number, frame_detections in enumerate(yolo_tagged_frames):
        inferenced_detections = []
        frame_image = get_frame_image(frame_number, video_path)
        for frame_detection in frame_detections:
            detected_cropped_image = get_cropped_image_by_detection_bounded_box(
                frame_image, frame_detection.box
            )

            face_data_from_detection: Optional[FaceData] = (
                get_face_data_from_person_detection(
                    frame_detection, detected_cropped_image
                )
            )

            if face_data_from_detection:
                face_data_from_detection.embedding = compute_face_embedding_from_rect(
                    frame_image, face_data_from_detection.face_box
                )
                frame_detection.face = face_data_from_detection

            # Append all confident detections (aggregator will use person only)
            inferenced_detections.append(frame_detection)
        inferenced_frames.append(inferenced_detections)

    return inferenced_frames


@cache.memoize()
def extract_object_boxes_and_tag_objects_yolo(video_path: str) -> List[List[Detection]]:
    yolo_results = yolo_model(video_path, verbose=False)

    # Get video FPS to calculate timestamps
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    detections = []
    frame_number = 0

    for yolo_result in yolo_results:
        frame = []
        # ultralytics.engine.results removed ULTRA-LYTICS
        timestamp = frame_number / fps
        name_map = yolo_result.names
        for box in yolo_result.boxes:
            confidence = float(box.conf[0])
            type = round(float(box.cls[0]))  # extract from tensor and round
            detection_id = str(uuid4())
            frame.append(
                Detection(
                    detection_id=detection_id,
                    box=BoundingBox(
                        x1=int(box.xyxy[0][0]),
                        y1=int(box.xyxy[0][1]),
                        x2=int(box.xyxy[0][2]),
                        y2=int(box.xyxy[0][3]),
                    ),
                    confidence=confidence,
                    timestamp=timestamp,
                    frame_number=frame_number,
                    recognized_object_type=name_map[type],
                    face=None,
                )
            )
        detections.append(frame)
        frame_number += 1
    return detections
