from src.media.video_shit_boxes.yolo import extract_object_boxes_and_tag_objects_yolo, process_yolo_boxes_to_get_inferenced_detections
from src.classification.Node import Node
from src.classification.tree import build_classification_tree

video_path = "/Users/jakeliebow/milli/tests/test_data/chunks/test_chunk_010.mp4"

def main():
    frames = extract_object_boxes_and_tag_objects_yolo(video_path)

    inferenced_frames = process_yolo_boxes_to_get_inferenced_detections(
        frames, video_path=video_path
    )
    root = Node(None, 0)

    build_classification_tree(inferenced_frames)