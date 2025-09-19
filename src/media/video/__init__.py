#!/usr/bin/env python3
"""
Main video provessing pipeline - runs yolo and action classifier
"""

from typing import Dict, List, Tuple
from collections import defaultdict, deque
from src.models.frame import Frame
from src.media.video.yolo import process_yolo_detection
from src.models.detection import ObjectTrack
from src.media.video.misc.detect import get_detections_per_frame
from src.media.video.action import ActionClassifier
from src.media.video.action import process_action_detections
import av
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_video(
    video_path: str,
    *,
    start_seconds: float = 0.0,
    target_fps: float = 15.0,
) -> Tuple[List[Frame], Dict[str, ObjectTrack], float]:

    processed_frames = []
    identities = {}
    tracker = {}

    container = av.open(video_path)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"

    action_head = ActionClassifier()
    clip_buffers = defaultdict(lambda: deque(maxlen=action_head.T))
    object_counter = defaultdict(int)
    # KINETICS_400 = None  # load list to map class_id -> label

    fps = float(stream.average_rate) if stream.average_rate else 30.0
    if target_fps<fps:
        skip=round(fps/target_fps)
        fps=target_fps
    else:
        skip=1
    frame_number_counter=0

    for decoded_frame_number,decoded_frame in enumerate(container.decode(stream)):
        if decoded_frame_number%skip!=0:
            continue
        print(frame_number_counter)
        timestamp = float(decoded_frame.pts * stream.time_base) if decoded_frame.pts is not None else frame_number_counter / fps
        frame_image = decoded_frame.to_ndarray(format="rgb24")
        current_frame = Frame(
            frame_number=frame_number_counter, 
            timestamp=timestamp,
            video_path=video_path, 
            image_data=frame_image
            )
        
        name_map, detections_per_frame = get_detections_per_frame(current_frame, identities, tracker)

        for detection in detections_per_frame:
            process_yolo_detection(current_frame, tracker, identities, processed_frames, name_map, detection)
            process_action_detections(detection, name_map, object_counter, action_head, clip_buffers)
            identities[tracker[detection.yolo_object_id]].detections.append(detection)
            logger.info(f"{timestamp}: Processed {len(detections_per_frame)} detections for frame {frame_number_counter}")

        # identities[detection.yolo_object_id].detections.append(detection)
        frame_number_counter+=1

    return (processed_frames, identities, fps)
