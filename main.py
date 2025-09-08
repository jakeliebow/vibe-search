from src.media.video_shit_boxes.yolo import extract_object_boxes_and_tag_objects_yolo, process_yolo_boxes_to_get_inferenced_detections
from src.classification.Node import Node
from src.classification.tree import node_processor
from src.media.audio.transcribe_and_diarize import transcribe_and_diarize_audio
from src.models.audio import FrameNormalizedAudioSegment
from src.classification.debug import save_cluster_crops
import sys
import datetime

video_path = "/Users/jakeliebow/milli/tests/test_data/chunks/test_chunk_010.mp4"
import sys
import numpy as np

# assume you already have:
# - Detection model
# - extract_object_boxes_and_tag_objects_yolo(video_path)
# - process_yolo_boxes_to_get_inferenced_detections(frames, video_path)
# - kmeans_spacetime_embedding(detections, K, ...)


def main():

    print("1")
    frames,fps = extract_object_boxes_and_tag_objects_yolo(video_path)

    print("2")
    inferenced_frames = process_yolo_boxes_to_get_inferenced_detections(
        frames, video_path
    )
    print("3")
    yolo_objects_root_node = Node(None, "ROOT",0)
    UNRELATED_PARENT_NODE = Node(None, "UNRELATED_PARENT_NODE",-1)
    start = datetime.datetime.now()
    print(f"starting {start}")
    node_processor(yolo_objects_root_node, inferenced_frames, UNRELATED_PARENT_NODE=UNRELATED_PARENT_NODE)
    mid = datetime.datetime.now()
    print(f"tree_built {mid - start}")
    #tracks = greedy_tracks(yolo_objects_root_node)
    end = datetime.datetime.now()
    print(f"tracks_built {end - mid}")    
    import pdb;pdb.set_trace()

    
    
    diarized_audio_segments = transcribe_and_diarize_audio(video_path)
    for segment in diarized_audio_segments:
        # Convert segment.start_time to frame number
        frame_start_number = int(segment.start_time * fps)  # Assuming fps is defined
        frame_end_number = int(segment.end_time * fps)  # Convert end time as well

        relevent_frames = inferenced_frames[frame_start_number:frame_end_number+1]
        for frame in relevent_frames:

            frame.append(
                FrameNormalizedAudioSegment(
                    normalized_start_time=frame_start_number,
                    normalized_end_time=frame_end_number,
                    start_time=segment.start_time,
                    end_time=segment.end_time,
                    speaker_label=segment.speaker_label,
                    transcription=segment.transcription,
                    asset_id=segment.asset_id, 
                    asset_type=segment.asset_id, 
                    voice_embedding=segment.voice_embedding, 
                    audio_array=segment.audio_array, 
                    sampling_rate=segment.sampling_rate 
                )
            )

if __name__ == "__main__":
    main()
