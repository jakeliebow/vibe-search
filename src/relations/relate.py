from src.models.frame import Frame
from typing import Dict, List, Tuple, Set
import numpy as np
from dataclasses import dataclass

def activation_function(value,median,std):
    return np.tanh((value - median) / std) if std > 0 else 0.0

@dataclass
class Edge:
    weight: float
    vertexes: Tuple[str, str]


def calculate_entity_relationships(
    frames: List[Frame],
    *,
    min_abs: float = 0.0,
    confidence_threshold: float = 0.01,
    minimum_for_max_mar_delta: float = 0.02,
) -> List[Edge]:
    yolo_object_speaker_occurences_with_mar_delta: Dict[Tuple[str, str], List[float]] = {}
    max_mar_delta=0.0
    mar_derivative_list=[]
    for frame in frames:
        if not frame.diarized_audio_segments or not frame.detections:
            continue
        for det in frame.detections:
            if det.face is None or det.face.mar_derivative is None:
                continue
            w = abs(det.face.mar_derivative)
            if w < min_abs:
                continue
            if w > max_mar_delta:
                max_mar_delta=w
            mar_derivative_list.append(w)
            v2 = det.yolo_uuid
            for seg in frame.diarized_audio_segments:
                v1 = seg.speaker_label
                key = (v1, v2)

                if key not in yolo_object_speaker_occurences_with_mar_delta:
                    yolo_object_speaker_occurences_with_mar_delta[key] = [0.0, 0]

                yolo_object_speaker_occurences_with_mar_delta[key][0] += w
                yolo_object_speaker_occurences_with_mar_delta[key][1] += 1

    max_mar_delta=np.maximum(max_mar_delta,minimum_for_max_mar_delta)
    mar_derivative_list.sort()
    std_dev_mar_delta = np.std(mar_derivative_list) if mar_derivative_list else 0.0

    median_mar_delta = mar_derivative_list[len(mar_derivative_list) // 2]

    pairings: List[Edge] = []
    for (v1, v2), (total, n) in yolo_object_speaker_occurences_with_mar_delta.items():
        if n == 0:
            continue
        average_mar_delta_during_cocurrence=total / n
        relation_weight = activation_function(average_mar_delta_during_cocurrence,median_mar_delta,std_dev_mar_delta)
        if relation_weight >= confidence_threshold:
            pairings.append(Edge(relation_weight, (v1, v2)))
    return pairings
