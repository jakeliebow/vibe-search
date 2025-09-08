#!/usr/bin/env python3
"""
Yolo Based object processing - gives you list of frames where each frame has its own lists of detection objects, based on how many objects in this frame
"""
from src.classification.Node import Node
from src.classification.meth import calculate_similarity



def node_processor(node: Node, frames,visited:set=set(),UNRELATED_PARENT_NODE=Node(None,None)):
    node_id = node.id
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
    if node.tree_height < 0:
        print(node)
        raise Exception(
            "skill issue"
        )  # will only occur if processing unrelated node, which we shouldn't
    next_tree_height = node.tree_height + 1

    
    nodes=[]
    
    for frame_detection in frame_detections:
        new_node=Node(frame_detection,tree_height=next_tree_height)
        nodes.append(new_node)

    for child_node in nodes:
        similarity = calculate_similarity(node, child_node)
        node.add_child(child_node, similarity)
        #UNRELATED_PARENT_NODE.add_child(child_node, 1.0 - similarity)
        node_processor(child_node, childrens_frame_slice, visited, UNRELATED_PARENT_NODE)
    
            

UNSIMILARITY_THRESHOLD = 1.0
SIMILARITY_THRESHOLD = 0.0