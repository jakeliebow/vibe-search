import cv2 
import numpy as np
from src.models.detection import BoundingBox
from src.utils.cache import cache
from functools import lru_cache
def ensure_gray_scale(image:np.ndarray):
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return gray

def ensure_rgb(image: np.ndarray) -> np.ndarray:
    """
    Ensure the input image is 3-channel RGB (uint8 expected by dlib/OpenCV).
    - Grayscale (H, W) -> RGB via COLOR_GRAY2RGB
    - BGR (H, W, 3) -> RGB via COLOR_BGR2RGB (OpenCV default is BGR)
    - BGRA (H, W, 4) -> RGB via COLOR_BGRA2RGB
    Raises ValueError on unsupported shapes.
    """
    if image is None:
        raise ValueError("ensure_rgb: image is None")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.ndim == 3:
        if image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    raise ValueError(f"ensure_rgb: unsupported image shape {getattr(image, 'shape', None)}")
@cache.memoize()
def get_frame_image(frame_number:int, video_path:str)->np.ndarray:
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, image = cap.read()
    if not ret:
        raise Exception("dick butt")

    return image

def get_cropped_image_by_detection_bounded_box(frame_image:np.ndarray,box:BoundingBox)->np.ndarray:
    cropped_image = frame_image[box.y1:box.y2, box.x1:box.x2]
    return cropped_image
