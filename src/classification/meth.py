from src.models.detection import BoundingBox
from typing import Optional
from src.models.detection import Detection, FaceData
from src.models.audio import FrameNormalizedAudioSegment
import numpy as np
class EffectiveWeighting:
    def __init__(self):
        self.weights = []
    def push_weight(self, similarity,name):
        self.weights.append({"weight_name":name,"similarity":similarity})

    def calculate_total_similarity(self):
        weight_map={
            "iou":50,
            "face":50
        }
        total_weight=0
        current_weight=0
        for weight in self.weights:
            current_weight=weight["similarity"]*weight_map[weight["weight_name"]]
            total_weight+=weight_map[weight["weight_name"]]
        if total_weight==0:
            return 0.0
        return current_weight/total_weight
    
def intersection_over_union(box_a: BoundingBox, box_b: BoundingBox) -> float:
    x1 = max(box_a.x1, box_b.x1); y1 = max(box_a.y1, box_b.y1)
    x2 = min(box_a.x2, box_b.x2); y2 = min(box_a.y2, box_b.y2)
    iw = max(0.0, x2 - x1); ih = max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, box_a.x2 - box_a.x1) * max(0.0, box_a.y2 - box_a.y1)
    area_b = max(0.0, box_b.x2 - box_b.x1) * max(0.0, box_b.y2 - box_b.y1)
    denom = area_a + area_b - inter
    return inter / (denom + 1e-9) if denom > 0 else 0.0

def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)

def calculate_similarity(
        node, node2
) -> float:
    EffectiveWeightingObject=EffectiveWeighting()
    if node is None or node2 is None:
        return 0.0
    
    if type(node)==Detection and type(node2)==Detection:

        if node.face is not None and node2.face is not None:
            face_similarity=cosine_similarity(
                node.face.embedding,
                node2.face.embedding,
            )
            EffectiveWeightingObject.push_weight(face_similarity,"face")
        
        iou_score = intersection_over_union(
            node.box, node2.box
        )
        EffectiveWeightingObject.push_weight(iou_score,"iou")
    return EffectiveWeightingObject.calculate_total_similarity()
