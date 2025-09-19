from src.media.video import process_video
from src.media.audio.transcribe_and_diarize import transcribe_audio,diarize_audio,group_diarized_audio_segments_by_speaker,extract_audio_from_mp4
from src.media.audio.voice_embedding import compute_voice_embeddings_per_speaker
from src.media.video.face.heuristic import process_and_inject_identity_heuristics
from src.relations.relate import calculate_entity_relationships,Edge
from src.utils.yt_download import download_video
from database.psql import PostgresStorage
import os
import soundfile as sf
import cv2
from pathlib import Path


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
    video_path = Path.cwd() / "test.mp4"
    video_path_str = str(video_path)
    url = 'https://www.youtube.com/shorts/43NtLs-DgQ8?feature=share'

    if video_path.exists():
        pass
    else:
        video_path_str = download_video(url)

    print("start")
    if True:
        processed_frames, identities, fps = process_video(
                    video_path_str
                    )
        process_and_inject_identity_heuristics(identities)

    ##AUDIO PROCESSING
    audio_array, sampling_rate = extract_audio_from_mp4(video_path)

    diarized_audio_segments_list_index = diarize_audio(audio_array, sampling_rate)
    diarized_audio_segments_by_speaker_index = group_diarized_audio_segments_by_speaker(diarized_audio_segments_list_index)
    #transcription_segments = transcribe_audio(audio_array)
    
    compute_voice_embeddings_per_speaker(
            diarized_audio_segments_by_speaker_index
    )  # 512 Dimension voice embedding

    frame_normalize_diarized_audio_segments(
            diarized_audio_segments_list_index, fps, processed_frames
            )

    edges = calculate_entity_relationships(processed_frames)

    with PostgresStorage() as psql:
        psql.reset_db()
        psql.run_setup()
        #### READ SECTION
        for track_id, track in identities.items():
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
            audio_data_path =  os.path.abspath(os.path.join(output_dir, f"{speaker_label}.wav"))
            sf.write(audio_data_path, speaker_track.audio_data, 16000)

            psql.stage_insert_row("node", {"id": speaker_label, "type": "spek", "media_path": audio_data_path})

            psql.stage_insert_row(
                    "speaker", {"id": speaker_label, "embedding": voice_embedding.tolist(),"audio_data_path":audio_data_path}
                    )

        output_dir = "./temp/debug_output/track"
        os.makedirs(output_dir, exist_ok=True)
        for track_id, track in identities.items():
            if track.face_embeddings is None:
                continue
            track_output_dir = f"./temp/debug_output/track/{track_id}/"
            os.makedirs(track_output_dir, exist_ok=True)
            image_data_path = os.path.abspath(os.path.join(track_output_dir, f"track_pic.png"))
            psql.stage_insert_row("node", {"id": track_id, "type": "yobj", "media_path": image_data_path})

            cv2.imwrite(image_data_path, track.sample.image_data)

            psql.stage_insert_row("yolo_object", {"id": track_id,"image_data_path":image_data_path})
            psql.stage_insert_row("node", {"id": track_id, "type": "yobj","media_path":image_data_path})


            track_faces_output_dir = f"./temp/debug_output/track/{track_id}/faces"
            os.makedirs(track_faces_output_dir, exist_ok=True)
            for embedding in track.face_embeddings:
                face_image_data_path =  os.path.abspath(os.path.join(track_faces_output_dir, f"{embedding.uuid}.png"))
                cv2.imwrite(face_image_data_path, embedding.image_data)
                psql.stage_insert_row("node", {"id": embedding.uuid, "type": "face","media_path":face_image_data_path})
                psql.stage_insert_row(
                        "face",
                        {"id": embedding.uuid, "embedding": embedding.embedding.tolist()},
                        )
                edges.append(Edge(1, (track_id, embedding.uuid)))

        for edge in edges:
            v1, v2 = edge.vertexes
            if v1 != v2:
                psql.stage_insert_row("edge", {"v1": v1, "v2": v2, "weight": float(edge.weight)})
        psql.tx_commit()


if __name__ == "__main__":
    main()
