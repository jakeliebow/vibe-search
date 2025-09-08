from ultralytics import YOLO
from .lazy_loader import LazyLoader

# yolo_model = YOLO('yolov8n.pt')  # General object detection model
yolo_model = LazyLoader(YOLO, "yolo11s.pt")
