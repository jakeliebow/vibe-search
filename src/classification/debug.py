import os, cv2
import numpy as np
from typing import List, Dict
from src.models.detection import Detection

def _clamp(v, lo, hi): return max(lo, min(hi, v))

def _read_frame(video_path: str, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read frame {frame_idx}")
    return frame  # BGR

def save_cluster_crops(
    video_path: str,
    detections: List["Detection"],
    labels: np.ndarray,
    out_dir: str = "clusters_out",
    max_per_cluster: int | None = None,  # None = all
) -> Dict[int, int]:
    os.makedirs(out_dir, exist_ok=True)
    counts: Dict[int, int] = {}
    for det, lbl in zip(detections, labels):
        # cap per cluster if requested
        if max_per_cluster is not None and counts.get(lbl, 0) >= max_per_cluster:
            continue

        frame = _read_frame(video_path, det.frame_number)

        h, w = frame.shape[:2]
        x1 = _clamp(int(det.box.x1), 0, w - 1)
        y1 = _clamp(int(det.box.y1), 0, h - 1)
        x2 = _clamp(int(det.box.x2), 0, w - 1)
        y2 = _clamp(int(det.box.y2), 0, h - 1)
        if x2 <= x1 or y2 <= y1:
            continue  # degenerate box

        crop = frame[y1:y2, x1:x2]  # BGR
        if crop.size == 0:
            continue

        cdir = os.path.join(out_dir, f"cluster_{lbl}")
        os.makedirs(cdir, exist_ok=True)
        fname = f"{det.frame_number}_{det.detection_id}.jpg"
        cv2.imwrite(os.path.join(cdir, fname), crop)
        counts[lbl] = counts.get(lbl, 0) + 1
    return counts