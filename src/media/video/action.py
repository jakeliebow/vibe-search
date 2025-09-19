#!/usr/bin/env python3
"""
Action Classifier stuff
"""

import torch, torch.nn.functional as F
from torchvision import transforms
from torchvision.models.video import r3d_18

class ActionClassifier:
    def __init__(self, device=None, T=16, sample_every=2):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = r3d_18(weights="KINETICS400_V1").eval().to(self.device)
        self.tf = transforms.Compose([
            transforms.ToTensor(),                       # HWC -> CHW
            transforms.Resize((112, 112), antialias=True),
            transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
                                 std=[0.22803, 0.22145, 0.216989]),
        ])
        self.T = T
        self.sample_every = sample_every

    @torch.inference_mode()
    def predict(self, frames_np_rgb):  # list of len T, each HxWx3 uint8 RGB
        clip = torch.stack([self.tf(f) for f in frames_np_rgb], dim=1)  # [C,T,H,W]
        logits = self.model(clip.unsqueeze(0).to(self.device))          # [1,400]
        prob = F.softmax(logits, dim=1)
        conf, cls = prob.max(dim=1)
        return int(cls.item()), float(conf.item())
    
def process_action_detections(detection, name_map, object_counter, action_head, clip_buffers):
    if name_map.get(detection.class_type, "") == "person":
        object_counter[detection.yolo_uuid] += 1
        if object_counter[detection.yolo_uuid] % action_head.sample_every == 0 and detection.image is not None:
            # detection.image is your cropped RGB uint8 person chip
            clip_buffers[detection.yolo_uuid].append(detection.image)

            if len(clip_buffers[detection.yolo_uuid]) == action_head.T:
                cls_id, score = action_head.predict(list(clip_buffers[detection.yolo_uuid]))
                detection.action=f"{cls_id}, {score}"