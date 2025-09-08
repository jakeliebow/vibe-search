from src.models.frame import Frame
from typing import Dict, List, NewType,Tuple

from dataclasses import dataclass

SpeakerId = NewType("SpeakerId", str)
ObjectId  = NewType("ObjectId", str)

@dataclass(frozen=True)
class Pairing:
    speaker: SpeakerId
    object_id: ObjectId
    avg_abs_mar_derivative: float
    frames: int  # number of co-present frames used in the average

def pair_speakers_and_objects_by_avg_mar(
    frames: List[Frame],
    *,
    min_abs: float = 0.0,
    confidence_threshold: float = 0.02,
) -> List[Pairing]:
    sum_abs: Dict[Tuple[SpeakerId, ObjectId], float] = {}
    cnt: Dict[Tuple[SpeakerId, ObjectId], int] = {}

    for frame in frames:
        if not frame.diarized_audio_segments or not frame.detections:
            continue
        print('?')
        for det in frame.detections:
            if det.face is None or det.face.mar_derivative is None:
                continue
            w = abs(det.face.mar_derivative)
            print(w)
            print("gay")
            if w < min_abs:
                continue

            oid = ObjectId(det.yolo_object_id)
            for seg in frame.diarized_audio_segments:
                sid = SpeakerId(seg.speaker_label)
                key = (sid, oid)

                if key not in sum_abs:
                    sum_abs[key] = 0.0
                    cnt[key] = 0

                sum_abs[key] += w
                cnt[key] += 1
                
    out: List[Pairing] = []
    for (sid, oid), total in sum_abs.items():
        n = cnt[(sid, oid)]
        if n == 0:
            continue
        avg = total / n
        if avg >= confidence_threshold:
            out.append(Pairing(speaker=sid, object_id=oid, avg_abs_mar_derivative=avg, frames=n))

    out.sort(key=lambda p: p.avg_abs_mar_derivative, reverse=True)
    return out