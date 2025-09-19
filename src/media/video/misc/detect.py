
#!/usr/bin/env python3
"""
Yolo Based frame processing - gives you name_map and a list of detections for a given frame
"""

from typing import Dict, List, Tuple, Optional
from src.state.AI.object_detection_models import yolo_model
from src.models.frame import Frame
from src.models.detection import BoundingBox, Detection, ImageSample
from src.models.detection import ObjectTrack
from uuid import uuid4
from src.media.video.misc.image_helpers import get_detection_image

def get_detections_per_frame(Frame, identities, tracker):
    yolo_results = yolo_model.track(
        source=Frame.image_data,
        tracker="bytetrack.yaml",
        conf=0.4,
        verbose=False,
        persist=True
    )
    yolo_result=yolo_results[-1]

    name_map = yolo_result.names
    detections = []
    for box in yolo_result.boxes:

        confidence = float(box.conf[0])
        class_type = round(float(box.cls[0]))
        yolo_object_id = str(int(box.id[0].item())) if box.id is not None else None
        bounding_box = BoundingBox(
            x1=int(box.xyxy[0][0]),
            y1=int(box.xyxy[0][1]),
            x2=int(box.xyxy[0][2]),
            y2=int(box.xyxy[0][3]),
        )

    if yolo_object_id and yolo_object_id not in tracker:

        track_uuid = str(uuid4())
        tracker[yolo_object_id] = track_uuid
        identities[track_uuid] = ObjectTrack(
            face_embeddings=[],
            yolo_object_id=yolo_object_id,
            detections=[],
            object_type=name_map[class_type],
            sample=ImageSample(
                confidence=confidence,
                frame_index=Frame.frame_number,
                image_data=Frame.image_data
            )
        )
    yolo_uuid = tracker[yolo_object_id]
    if identities[yolo_uuid].sample.confidence < confidence:
        identities[yolo_uuid].sample = ImageSample(
            confidence=confidence,
            frame_index=Frame.frame_number,
            image_data=Frame.image_data
        )

        detections.append(Detection(
            detection_id=str(uuid4()),
            box=bounding_box,
            image=get_detection_image(Frame.image_data, bounding_box),
            class_type=class_type,
            confidence=float(box.conf[0]),
            timestamp=Frame.timestamp,
            frame_number=int(Frame.frame_number),
            recognized_object_type=name_map[class_type],
            face=None,
            yolo_object_id=yolo_object_id,
            yolo_uuid=yolo_uuid,
            )
        )
    return name_map, detections
