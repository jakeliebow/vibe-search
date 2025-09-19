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
import av
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
def run_yolo(Frame, tracker, identities, processed_frames):
        yolo_results = yolo_model.track(
            source=Frame.image_data,
            tracker="bytetrack.yaml",
            conf=0.4,
            verbose=False,
            persist=True
        )
        yolo_result=yolo_results[-1]

        name_map = yolo_result.names
        for box in yolo_result.boxes:

            confidence = float(box.conf[0])
            type = round(float(box.cls[0]))
            detection_id = str(uuid4())
            yolo_object_id = str(int(box.id[0].item())) if box.id is not None else None
            bounding_box = BoundingBox(
                x1=int(box.xyxy[0][0]),
                y1=int(box.xyxy[0][1]),
                x2=int(box.xyxy[0][2]),
                y2=int(box.xyxy[0][3]),
            )
            detected_cropped_image = get_cropped_image_by_detection_bounded_box(
                Frame.image_data, bounding_box
            )

            detection = Detection(
                detection_id=detection_id,
                box=bounding_box,
                confidence=confidence,
                timestamp=Frame.timestamp,
                frame_number=int(Frame.frame_number),
                recognized_object_type=name_map[type],
                face=None,
                yolo_object_id=None,
                yolo_uuid=None,
            )
            if name_map[type] == "person":
                face_data_from_detection: Optional[FaceData] = (
                    get_face_data_from_person_detection(
                        detection, detected_cropped_image
                    )
                )

                if face_data_from_detection:
                    face_data_from_detection.embedding = Embedding(
                        embedding=compute_face_embedding_from_rect(
                            Frame.image_data, face_data_from_detection.face_box
                        ),
                        image_data=detected_cropped_image
                    )
                    detection.face = face_data_from_detection

            if yolo_object_id:

                if yolo_object_id not in tracker:
                    id = str(uuid4())
                    tracker[yolo_object_id] = id
                    identities[id] = YoloObjectTrack(
                        face_embeddings=[],
                        yolo_object_id=yolo_object_id,
                        detections=[],
                        type=name_map[type],
                        sample=ImageSample(
                            confidence=confidence,
                            frame_index=Frame.frame_number,
                            image_data=Frame.image_data[
                                bounding_box.y1 : bounding_box.y2,
                                bounding_box.x1 : bounding_box.x2,
                            ],
                        )
                    )

                yolo_uuid = tracker[yolo_object_id]
                if identities[yolo_uuid].sample.confidence < confidence:
                    identities[yolo_uuid].sample = ImageSample(
                        confidence=confidence,
                        frame_index=Frame.frame_number,
                        image_data=Frame.image_data[
                            bounding_box.y1 : bounding_box.y2,
                            bounding_box.x1 : bounding_box.x2,
                        ],
                    )
                detection.yolo_uuid = yolo_uuid
                detection.yolo_object_id = yolo_object_id
                identities[yolo_uuid].detections.append(detection)
            Frame.detections.append(detection)
        processed_frames.append(Frame)

