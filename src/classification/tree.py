#!/usr/bin/env python3
"""
Yolo Based object processing - gives you list of frames where each frame has its own lists of detection objects, based on how many objects in this frame
"""
from src.classification.Node import Node
from src.classification.meth import calculate_similarity



def node_processor(node: Node, frames,visited:set=set(),UNRELATED_PARENT_NODE=Node(None,None)):
    node_id = None
    if node.detection:
        node_id = node.detection.detection_id

    if node_id not in visited:
        visited.add(node_id)
    else:
        return None

    number_of_frames_left = len(frames)
    if number_of_frames_left == 1:
        childrens_frame_slice = []
    else:
        childrens_frame_slice = frames[1:]
    if number_of_frames_left == 0:
        return None

    frame_detections = frames[0]
    if not node.tree_height:
        raise Exception(
            "skill issue"
        )  # will only occur if processing unrelated node, which we shouldn't
    next_tree_height = node.tree_height + 1

    for frame_detection in frame_detections:
        similarity = calculate_similarity(node.detection, frame_detection)
        child_detection_node = Node(frame_detection, next_tree_height)
        if similarity > SIMILARITY_THRESHOLD:
            node.add_child(child_detection_node, similarity)

        if similarity < UNSIMILARITY_THRESHOLD:
            UNRELATED_PARENT_NODE.add_child(child_detection_node, 1.0 - similarity,visited,UNRELATED_PARENT_NODE)

        node_processor(child_detection_node, childrens_frame_slice,visited,UNRELATED_PARENT_NODE)

def build_classification_tree(frames):
    root = Node(None, 0)
    node_processor(root, frames)
    return root

UNSIMILARITY_THRESHOLD = 1.0
SIMILARITY_THRESHOLD = 0.0