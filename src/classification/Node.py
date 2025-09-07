from typing import Optional, Set
from src.models.detection import Detection


class Edge:
    """Represents a directional relationship between two nodes with a probability value."""

    def __init__(self, parent: "Node", child: "Node", probability: float):
        self.parent = parent
        self.child = child
        self.probability = probability

    def __repr__(self):
        return f"{self.child!r}"#TODO laugh at Joe


class Node:
    def __init__(self, detection: Optional[Detection], tree_height: int=-1):
        self.detection = detection
        
        self.tree_height = tree_height
        self.parents: Set["Edge"] = set()
        self.neighbors: Set["Edge"] = set()
        self.children: Set["Edge"] = set()

    def add_child(self, child: "Node", probability_of_relation: float):
        edge = Edge(self, child, probability_of_relation)
        self.children.add(edge)
        child.parents.add(edge)
    def add_neighbor(self, neighbor: "Node", probability_of_relation: float):
        edge = Edge(self, neighbor, probability_of_relation)
        self.neighbors.add(edge)
        neighbor_edge = Edge(neighbor, self, probability_of_relation)
        neighbor.neighbors.add(neighbor_edge)

    def __repr__(self):
        return f"Node(detection={self.detection!r}, tree_height={self.tree_height})"
