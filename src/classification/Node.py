from typing import Optional, Set
from src.models.detection import Detection


class Edge:
    """Represents a directional relationship between two nodes with a probability value."""

    def __init__(self, parent: "Node", child: "Node", probability: float):
        self.parent = parent
        self.child = child
        self.probability = probability

    def __repr__(self):
        return f"Edge({self.parent!r} -> {self.child!r}, p={self.probability:.3f})"


class Node:
    def __init__(self, detection: Optional[Detection], tree_height: Optional[int]):
        self.detection = detection
        self.tree_height = tree_height
        self.parents: Set["Edge"] = set()
        self.children: Set["Edge"] = set()

    def add_child(self, child: "Node", probability_of_relation: float):
        edge = Edge(self, child, probability_of_relation)
        self.children.add(edge)
        child.parents.add(edge)
        return edge

    def __repr__(self):
        return f"Node(detection={self.detection!r}, height={self.tree_height})"
