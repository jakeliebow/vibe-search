#!/usr/bin/env python3
"""
Yolo Based object processing - gives you list of frames where each frame has its own lists of detection objects, based on how many objects in this frame
"""

from typing import Dict, List, Tuple

from pydantic import UUID4
from src.state.AI.object_detection_models import yolo_model
from src.models.frame import Frame
from src.models.detection import BoundingBox, Detection, FaceData, YoloObjectTrack
from src.utils.cache import cache
import cv2
from uuid import uuid4
from src.media.video_shit_boxes.face.face_embeddings import (
    compute_face_embedding_from_rect,
)
from src.models.embedding import Embedding
from typing import List, Optional
from src.media.video_shit_boxes.misc.image_helpers import (
    get_frame_image,
    get_cropped_image_by_detection_bounded_box,
)
from src.media.video_shit_boxes.face.main import (
    get_face_data_from_person_detection,
)
from diskcache import Cache

cache = Cache("/tmp/yolo___")


def process_and_inject_yolo_boxes_frame_by_frame(
    yolo_tagged_frames,
    video_path,
) -> List[List[Detection]]:
    for frame_number, frame_detections in enumerate(yolo_tagged_frames):
        frame_image = get_frame_image(frame_number, video_path)
        for frame_detection in frame_detections.detections:
            detected_cropped_image = get_cropped_image_by_detection_bounded_box(
                frame_image, frame_detection.box
            )

            face_data_from_detection: Optional[FaceData] = (
                get_face_data_from_person_detection(
                    frame_detection, detected_cropped_image
                )
            )

            if face_data_from_detection:
                face_data_from_detection.embedding = Embedding(embedding=compute_face_embedding_from_rect(
                    frame_image, face_data_from_detection.face_box
                ))
                frame_detection.face = face_data_from_detection



#@cache.memoize()
def extract_object_boxes_and_tag_objects_yolo(
    video_path: str,
) -> Tuple[List[Frame], Dict[str, YoloObjectTrack], float]:
    yolo_results = yolo_model.track(
        source=video_path,
        verbose=False,
        stream=True,
        show=False,
        tracker="bytetrack.yaml",
        conf=0.3,
    )

    # Get video FPS to calculate timestamps
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    frames = []
    identities = {}
    tracker = {}
    for frame_number, yolo_result in enumerate(yolo_results):

        timestamp = frame_number / fps
        name_map = yolo_result.names
        frame = Frame(frame_number=frame_number, timestamp=timestamp)
        for box in yolo_result.boxes:
            confidence = float(box.conf[0])
            type = round(float(box.cls[0]))  # extract from tensor and round
            detection_id = str(uuid4())
            yolo_object_id = str(int(box.id[0].item())) if box.id is not None else None
            detection = Detection(
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
                yolo_object_id=None,
                yolo_uuid=None,
            )
            if yolo_object_id:
                if yolo_object_id not in tracker:
                    id = str(uuid4())
                    tracker[yolo_object_id] = id

                    identities[id] = YoloObjectTrack(
                        yolo_object_id=yolo_object_id,
                        detections=[],
                        type=name_map[type],
                    )
                yolo_uuid = tracker[yolo_object_id]
                detection.yolo_uuid = yolo_uuid
                detection.yolo_object_id = yolo_object_id
                identities[yolo_uuid].detections.append(detection)
            frame.detections.append(detection)
        frames.append(frame)
    return (frames, identities, fps)
