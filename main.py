from src.models.detection import YoloObjectTrack
from src.media.video_shit_boxes.yolo import (
    extract_object_boxes_and_tag_objects_yolo,
    process_and_inject_yolo_boxes_frame_by_frame,
)
from src.classification.Node import Node
from src.classification.tree import node_processor
from src.media.audio.transcribe_and_diarize import transcribe_and_diarize_audio
from src.media.audio.voice_embedding import compute_voice_embeddings_per_speaker
from src.models.audio import DiarizedAudioSegment, SpeakerTrack
from src.classification.debug import save_cluster_crops
from src.relations.voice_yolo_debug import (
    debug_output,
    get_example_face_images,
    get_example_audio_segments,
)
from src.media.video_shit_boxes.heuristic import process_and_inject_identity_heuristics
from src.relations.relate import calculate_entity_relationships, Pairing
from src.identity.id import Identity, build_individual_identities
from database.psql import PostgresStorage
from typing import List, Dict, Set, Tuple, Optional
from uuid import uuid4

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
    ### VIDEO PROCESSING
    yolo_frame_by_frame_index, yolo_track_id_index, fps = extract_object_boxes_and_tag_objects_yolo(
        video_path
    )
    process_and_inject_yolo_boxes_frame_by_frame(
        yolo_frame_by_frame_index, video_path
    )
    process_and_inject_identity_heuristics(yolo_track_id_index)

    ##AUDIO PROCESSING
    diarized_audio_segments_list_index, diarized_audio_segments_by_speaker_index = transcribe_and_diarize_audio(video_path)
    compute_voice_embeddings_per_speaker(diarized_audio_segments_by_speaker_index)#512 Dimension voice embedding

    frame_normalize_diarized_audio_segments(
        diarized_audio_segments_list_index, fps, yolo_frame_by_frame_index
    )
    
    ### calculate relations
    
    edges = calculate_entity_relationships(yolo_frame_by_frame_index)
    
    tracker = {}

    with PostgresStorage() as psql:
        for speaker_label, speaker_track in diarized_audio_segments_by_speaker_index.items():
            id = uuid4()
            tracker[speaker_label] = [id]
            embedding = speaker_track.embedding
            psql.insert_row("speaker", {"id": id, "embedding": embedding})
            psql.insert_row("node", {"id": id, "type": "speaker"})

        for track_id, track in yolo_track_id_index.items():
            if track.face_embeddings == None:
                continue
            for e in track.face_embeddings:
                id = uuid4()

                if track_id not in tracker:
                    tracker[track_id] = [id]
                else:
                    tracker[track_id].append(id)

                embedding = e
                psql.insert_row("face", {"id": id, "embedding": embedding})
                psql.insert_row("node", {"id": id, "type": "face"})
        
        for edge in edges:
            tracker[edge.speaker_id]

if __name__ == "__main__":
    main()
