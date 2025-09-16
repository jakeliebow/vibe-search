from src.models.frame import Frame
from typing import Dict, List, Tuple, Set

from dataclasses import dataclass


@dataclass
class Edge:
    weight: float
    vertexes: Tuple[str, str]


def calculate_entity_relationships(
    frames: List[Frame],
    *,
    min_abs: float = 0.0,
    confidence_threshold: float = 0.02,
) -> List[Edge]:
    sum_abs: Dict[Tuple[str, str], List[float]] = {}

    for frame in frames:
        if not frame.diarized_audio_segments or not frame.detections:
            continue
        for det in frame.detections:
            if det.face is None or det.face.mar_derivative is None:
                continue
            w = abs(det.face.mar_derivative)
            if w < min_abs:
                continue

            oid = det.yolo_object_id
            for seg in frame.diarized_audio_segments:
                sid = seg.speaker_label
                key = (sid, oid)

                if key not in sum_abs:
                    sum_abs[key] = [0.0, 0]

                sum_abs[key][0] += w
                sum_abs[key][1] += 1

    pairings: List[Edge] = []
    for (sid, oid), (total, n) in sum_abs.items():
        if n == 0:
            continue
        avg = total / n
        if avg >= confidence_threshold:
            pairings.append(Edge(avg, (sid, oid)))
    return pairings
