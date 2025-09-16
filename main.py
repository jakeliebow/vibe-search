from src.media.video_shit_boxes.yolo import (
    extract_object_boxes_and_tag_objects_yolo,
    process_and_inject_yolo_boxes_frame_by_frame,
)
from src.media.audio.transcribe_and_diarize import transcribe_and_diarize_audio
from src.media.audio.voice_embedding import compute_voice_embeddings_per_speaker
from src.media.video_shit_boxes.heuristic import process_and_inject_identity_heuristics
from src.relations.relate import calculate_entity_relationships, Edge
from database.psql import PostgresStorage
from uuid import uuid4
import os
import soundfile as sf  # pip install soundfile

import cv2
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
    yolo_frame_by_frame_index, yolo_track_id_index, fps = (
        extract_object_boxes_and_tag_objects_yolo(video_path)
    )
    process_and_inject_yolo_boxes_frame_by_frame(yolo_frame_by_frame_index)
    process_and_inject_identity_heuristics(yolo_track_id_index)

    ##AUDIO PROCESSING
    diarized_audio_segments_list_index, diarized_audio_segments_by_speaker_index = (
        transcribe_and_diarize_audio(video_path)
    )
    compute_voice_embeddings_per_speaker(
        diarized_audio_segments_by_speaker_index
    )  # 512 Dimension voice embedding

    frame_normalize_diarized_audio_segments(
        diarized_audio_segments_list_index, fps, yolo_frame_by_frame_index
    )

    ### calculate relations

    edges = calculate_entity_relationships(yolo_frame_by_frame_index)

    with PostgresStorage() as psql:
        #### READ SECTION
        for track_id, track in yolo_track_id_index.items():
            if track.face_embeddings is None:
                continue
            for face_embedding in track.face_embeddings:
                embedding_relations = psql.query_embedding_similarity(
                    "face", face_embedding.embedding, top_n=10
                )
                for relation in embedding_relations:
                    edges.append(
                        Edge(
                            relation["similarity"],
                            (face_embedding.uuid, relation["id"]),
                        )
                    )

        ### WRITE SECTION
        print(f"fucking here:>>> {len(diarized_audio_segments_by_speaker_index.values())}")
        for (
            speaker_label,
            speaker_track,
        ) in diarized_audio_segments_by_speaker_index.items():
            voice_embedding = speaker_track.voice_embedding
            embedding_relations = psql.query_embedding_similarity(
                "speaker", voice_embedding, top_n=10
            )
            for relation in embedding_relations:
                edges.append(
                    Edge(relation["similarity"], (speaker_label, relation["id"]))
                )
            output_dir = "./temp/debug_output/audio"
            os.makedirs(output_dir, exist_ok=True)
            audio_data_path = os.path.join(output_dir, f"{speaker_label}.wav")
            sf.write(audio_data_path, speaker_track.audio_data, 16000)

            psql.insert_row("node", {"id": speaker_label, "type": "spek"})

            psql.insert_row(
                "speaker", {"id": speaker_label, "embedding": voice_embedding.tolist(),"audio_data_path":audio_data_path}
            )

        output_dir = "./temp/debug_output/track"
        os.makedirs(output_dir, exist_ok=True)
        import pdb;
        pdb.set_trace()
        for track_id, track in yolo_track_id_index.items():
            if track.face_embeddings is None:
                continue
            image_data_path = os.path.join(output_dir, f"{track_id}.png")
            cv2.imwrite(image_data_path, track.sample.image_data)
            psql.insert_row("yolo_object", {"id": track_id,"image_data_path":image_data_path})
            psql.insert_row("node", {"id": track_id, "type": "yobj"})

            for embedding in track.face_embeddings:
                psql.insert_row("node", {"id": embedding.uuid, "type": "face"})
                psql.insert_row(
                    "face",
                    {"id": embedding.uuid, "embedding": embedding.embedding.tolist()},
                )
                edges.append(Edge(1, (track_id, embedding.uuid)))

        for edge in edges:
            v1, v2 = edge.vertexes
            if v1 != v2:
                psql.insert_row("edge", {"v1": v1, "v2": v2, "weight": edge.weight})


if __name__ == "__main__":
    main()
