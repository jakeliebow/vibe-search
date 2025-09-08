from typing import Optional, Set
from src.models.detection import Detection
from src.models.audio import FrameNormalizedAudioSegment


class Edge:
    """Represents a directional relationship between two nodes with a probability value."""

    def __init__(self, parent: "Node", child: "Node", probability: float):
        self.parent = parent
        self.child = child
        self.probability = probability

    def __repr__(self):
        return f"{self.child!r}"#TODO laugh at Joe


class Node:
    def __init__(self, metadata: Detection | FrameNormalizedAudioSegment | None,id=None, tree_height: int=-1):
        self.metadata = metadata
        
        self.type = type(metadata).__name__ if metadata else None
        if self.type == "Detection":
            self.id=metadata.detection_id
        elif self.type == "FrameNormalizedAudioSegment":
            self.id=metadata.speaker_label
        else:
            self.id=id
        self.tree_height = tree_height
        self.parents: Set["Edge"] = set()
        self.children: Set["Edge"] = set()

    def add_child(self, child: "Node", probability_of_relation: float):
        edge = Edge(self, child, probability_of_relation)
        self.children.add(edge)
        child.parents.add(edge)

    def __repr__(self):
        return f"Node({self.id!r}, tree_height={self.tree_height})"
