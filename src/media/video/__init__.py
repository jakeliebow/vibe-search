#!/usr/bin/env python3
"""
Yolo Based object processing - gives you list of frames where each frame has its own lists of detection objects, based on how many objects in this frame
"""

from typing import Dict, List, Tuple
from src.models.frame import Frame
from src.media.video.yolo import run_yolo
from src.models.detection import ObjectTrack
from src.media.video.misc.detect import get_detection
from src.media.video.action import ActionClassifier
import av

import torch, torch.nn.functional as F
from torchvision import transforms
from torchvision.models.video import r3d_18
from collections import defaultdict, deque
from diskcache import Cache

cache = Cache("/tmp/yolo______")

@cache.memoize()
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
    frame_counters = defaultdict(int)

    fps = float(stream.average_rate) if stream.average_rate else 30.0
    if target_fps<fps:
        skip=round(fps/target_fps)
        fps=target_fps
    else:
        skip=1
    frame_number=0
    for decoded_frame_number,decoded_frame in enumerate(container.decode(stream)):
        if decoded_frame_number%skip!=0:
            continue
        print(frame_number)
        timestamp = float(decoded_frame.pts * stream.time_base) if decoded_frame.pts is not None else frame_number / fps
        frame_image = decoded_frame.to_ndarray(format="rgb24")
        current_frame = Frame(
            frame_number=frame_number, 
            timestamp=timestamp,
            video_path=video_path, 
            image_data=frame_image
            )
        name_map, detection = get_detection(current_frame)
        run_yolo(current_frame, tracker, identities, processed_frames, name_map, detection)
        frame_number+=1
    return (processed_frames, identities, fps)
