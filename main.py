from src.media.video_shit_boxes.yolo import (
    extract_object_boxes_and_tag_objects_yolo,
    process_yolo_boxes_to_get_inferenced_detections,
)
from src.classification.Node import Node
from src.classification.tree import node_processor
from src.media.audio.transcribe_and_diarize import transcribe_and_diarize_audio
from src.models.audio import DiarizedAudioSegment
from src.classification.debug import save_cluster_crops
from src.classification.voice_yolo_debug import debug_voice_yolo_pairings
from src.media.video_shit_boxes.heuristic import process_and_inject_identity_heuristics
from src.relations.relate import pair_speakers_and_objects_by_avg_mar
import datetime

video_path = "/Users/jakeliebow/milli/tests/test_data/chunks/test_chunk_010.mp4"


def frame_normalize_diarized_audio_segments(
    diarized_audio_segments, fps, inferenced_frames
):
    for segment in diarized_audio_segments:
        # Convert segment.start_time to frame number
        frame_start_number = int(segment.start_time * fps)  # Assuming fps is defined
        frame_end_number = int(segment.end_time * fps)  # Convert end time as well

        relevent_frames = inferenced_frames[frame_start_number : frame_end_number + 1]
        for frame in relevent_frames:
            frame.diarized_audio_segments.append(segment)


def main():
    print("1")

    frames, identified_objects, fps = extract_object_boxes_and_tag_objects_yolo(
        video_path
    )
    print("2")
    inferenced_frames = process_yolo_boxes_to_get_inferenced_detections(
        frames, video_path
    )
    process_and_inject_identity_heuristics(identified_objects)

    diarized_audio_segments = transcribe_and_diarize_audio(video_path)
    frame_normalize_diarized_audio_segments(
        diarized_audio_segments, fps, inferenced_frames
    )
    pairing = pair_speakers_and_objects_by_avg_mar(inferenced_frames)
    debug_dir = debug_voice_yolo_pairings(pairing, inferenced_frames, video_path)


if __name__ == "__main__":
    main()
