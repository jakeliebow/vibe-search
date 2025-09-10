from src.models.detection import YoloObjectTrack
from src.media.video_shit_boxes.yolo import (
    extract_object_boxes_and_tag_objects_yolo,
    process_yolo_boxes_to_get_inferenced_detections,
)
from src.classification.Node import Node
from src.classification.tree import node_processor
from src.media.audio.transcribe_and_diarize import transcribe_and_diarize_audio
from src.models.audio import DiarizedAudioSegment, SpeakerTrack
from src.classification.debug import save_cluster_crops
from src.relations.voice_yolo_debug import (
    debug_output,
    get_example_face_images,
    get_example_audio_segments,
)
from src.media.video_shit_boxes.heuristic import process_and_inject_identity_heuristics
from src.relations.relate import pair_speakers_and_objects_by_avg_mar, Pairing
from src.identity.id import Identity, build_individual_identities
from typing import List, Dict, Set, Tuple, Optional
from uuid import uuid4

video_path = "/Users/jakeliebow/milli/tests/test_data/chunks/test_chunk_010.mp4"


def group_diarized_audio_segments_by_speaker(
    diarized_audio_segments: List[DiarizedAudioSegment],
) -> Dict[str, SpeakerTrack]:
    """
    Group diarized audio segments by speaker label, similar to how objects are grouped by ID.

    Args:
        diarized_audio_segments: List of diarized audio segments

    Returns:
        Dictionary mapping speaker labels to SpeakerTrack objects containing grouped segments
    """
    speaker_tracks = {}

    for segment in diarized_audio_segments:
        speaker_label = segment.speaker_label

        if speaker_label not in speaker_tracks:
            speaker_tracks[speaker_label] = SpeakerTrack(
                speaker_label=speaker_label, segments=[]
            )

        speaker_tracks[speaker_label].segments.append(segment)

    return speaker_tracks


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
    identified_speakers = group_diarized_audio_segments_by_speaker(
        diarized_audio_segments
    )

    frame_normalize_diarized_audio_segments(
        diarized_audio_segments, fps, inferenced_frames
    )
    result = pair_speakers_and_objects_by_avg_mar(inferenced_frames)
    pairings: List[Pairing] = result[0]
    paired: Set[str] = result[1]

    # Build the list of individual identities
    individual_identities = build_individual_identities(
        pairings, paired, identified_objects, identified_speakers
    )

    print(f"Created {len(individual_identities)} individual identities:")
    for identity in individual_identities:
        print(f"  {identity.id}:")
        if identity.yolo_track and identity.speaker_track:
            print(
                f"    - Paired: Speaker {identity.speaker_track.speaker_label} + Object {identity.yolo_track.yolo_object_id} (confidence: {identity.pairing_confidence:.3f})"
            )
        elif identity.speaker_track:
            print(f"    - Speaker only: {identity.speaker_track.speaker_label}")
        elif identity.yolo_track:
            print(
                f"    - Object only: {identity.yolo_track.yolo_object_id} ({identity.yolo_track.type})"
            )


if __name__ == "__main__":
    main()
