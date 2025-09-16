from typing import Dict
from src.models.detection import YoloObjectTrack, MarAtIndex
import numpy as np


def process_and_inject_identity_heuristics(yolo_objects: Dict[str, YoloObjectTrack]):
    for yolo_id in yolo_objects:
        yolo_object_track = yolo_objects[yolo_id]

        last_mar = MarAtIndex(frame_index=0, mar=0.0)

        for index, detection in enumerate(yolo_object_track.detections):
            face_embeddings = []
            face = detection.face

            if face:
                mar = face.mar
                if mar:
                    mar_derivative = (mar - last_mar.mar) / (
                        max(1, index - last_mar.frame_index)
                    )
                    face.mar_derivative = mar_derivative
                    last_mar = MarAtIndex(frame_index=index, mar=mar)
                if face.embedding is not None:
                    if yolo_object_track.face_embeddings is None:
                        yolo_object_track.face_embeddings = []
                    face_embeddings.append(face.embedding)
        yolo_objects[yolo_id].face_embeddings=face_embeddings