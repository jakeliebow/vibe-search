from typing import Optional
from src.models.objects import FaceData, BoundingBox
from src.processing.video_shit_boxes.face.face_snapshots import (
    run_dlib_on_person_to_get_face_data,
    head_focused_upscale,
)
from src.processing.video_shit_boxes.face.face_embeddings import (
    compute_face_embedding_from_rect,
)


def get_face_data_from_person_detection(
    frame_detection, detected_cropped_image
) -> Optional[FaceData]:
    if frame_detection.recognized_object_type != "person":
        return None

    # If small, focus on head and upscale; keep mapping info to project detections back
    used_image = detected_cropped_image
    scale = 1.0
    x_off = 0
    y_off = 0
    if detected_cropped_image.shape[0] < 150 or detected_cropped_image.shape[1] < 150:
        used_image, scale, x_off, y_off = head_focused_upscale(
            detected_cropped_image, head_ratio=0.95, min_width=200, min_height=200
        )

    face_data_from_detection: Optional[FaceData] = run_dlib_on_person_to_get_face_data(
        used_image
    )
    if face_data_from_detection:
        if scale != 1.0 or x_off != 0 or y_off != 0:
            fb = face_data_from_detection.face_box
            l = int(round(fb.x1 / scale)) + x_off
            t = int(round(fb.y1 / scale)) + y_off
            r = int(round(fb.x2 / scale)) + x_off
            b = int(round(fb.y2 / scale)) + y_off
            face_data_from_detection.face_box = BoundingBox(x1=l, y1=t, x2=r, y2=b)
    return face_data_from_detection
