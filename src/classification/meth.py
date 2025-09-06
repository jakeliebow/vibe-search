from src.models.detection import BoundingBox
from typing import Optional
from src.models.detection import Detection
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
def calculate_similarity(
    potential_parent_detection: Optional[Detection], potential_leaf_detection: Detection
) -> float:
    if potential_parent_detection:
        iou_score = intersection_over_union(
            potential_parent_detection.box, potential_leaf_detection.box
        )
        return iou_score
    return 0.0
