from src.media.video_shit_boxes.yolo import (
    extract_object_boxes_and_tag_objects_yolo,
    process_and_inject_yolo_boxes_frame_by_frame,
)
from src.media.audio.transcribe_and_diarize import transcribe_and_diarize_audio
from src.media.audio.voice_embedding import compute_voice_embeddings_per_speaker
from src.media.video_shit_boxes.heuristic import process_and_inject_identity_heuristics
from src.relations.relate import calculate_entity_relationships,Edge
from database.psql import PostgresStorage
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
    print("start")
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
    for track_id, track in yolo_track_id_index.items():#all track ids are classified as same
        edges.append(
            Edge(1, (track_id, track_id))
        )


    tracker = {}

    with PostgresStorage() as psql:
        for speaker_label, speaker_track in diarized_audio_segments_by_speaker_index.items():
            id = str(uuid4())
            tracker[speaker_label] = [id]
            embedding = speaker_track.voice_embedding
            psql.insert_row("node", {"id": id, "type": "spek"})
            psql.insert_row("speaker", {"id": id, "embedding": embedding})
            

        for track_id, track in yolo_track_id_index.items():
            if track.face_embeddings is None:
                continue
            for embedding in track.face_embeddings:
                id = str(uuid4())
                psql.query_embedding_similarity("face", embedding, top_n=10)
                if track_id not in tracker:
                    tracker[track_id] = [id]
                else:
                    tracker[track_id].append(id)
                psql.insert_row("node", {"id": id, "type": "face"})
                psql.insert_row("face", {"id": id, "embedding": embedding.tolist()})



                
        
        for edge in edges:#heuristic edges
            v1,v2 = edge.vertexes
            
            v1_uuids = tracker[v1]
            v2_uuids = tracker[v2]
            for v1_uuid in v1_uuids:
                for v2_uuid in v2_uuids:
                    if v1_uuid == v2_uuid:
                        continue
                    psql.insert_row("edge", {"v1": v1_uuid, "v2": v2_uuid,"weight": edge.weight})


if __name__ == "__main__":
    main()
