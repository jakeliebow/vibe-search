from typing import Dict
from src.models.detection import ObjectTrack, MarAtIndex

def process_and_inject_identity_heuristics(yolo_objects: Dict[str, ObjectTrack]):
    for track_uuid in yolo_objects:
        yolo_object_track = yolo_objects[track_uuid]
        last_mar = MarAtIndex(frame_index=0, mar=0.0)

        for index, detection in enumerate(yolo_object_track.detections):

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
                    if yolo_objects[track_uuid].face_embeddings is None:
                        yolo_objects[track_uuid].face_embeddings = []
                    yolo_objects[track_uuid].face_embeddings.append(face.embedding)
