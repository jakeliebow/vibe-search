from src.media.video import process_video
from src.media.audio.transcribe import transcribe_audio
from src.models.audio import DiarizedAudioSegment
from src.media.audio.extract import (
    extract_audio_from_mp4,
)
from src.media.video.face.heuristic import process_and_inject_identity_heuristics
from src.relations.relate import calculate_entity_relationships, Edge
from src.utils.yt_download import download_video
from src.relations.meth import cosine_similarity
from database.psql import PostgresStorage
import os
import soundfile as sf
import cv2
from pathlib import Path
from src.media.audio.voice_embedding import generate_voice_embedding
from uuid import uuid4

from src.utils.cache import cache


def frame_normalize_diarized_audio_segments(
    transcription_audio_segments,
    fps,
    inferenced_frames,
    audio_array,
    *,
    sample_rate=16000,
):
    all_audio_segment_list = []
    max_time = float(audio_array.shape[1]) / float(sample_rate)
    min_time = 0.0

    for segment in transcription_audio_segments:
        frame_start_number = int(segment.start_time * fps)  # Assuming fps is defined
        frame_end_number = int(segment.end_time * fps)  # Convert end time as well

        relevent_frames_transcription = inferenced_frames[frame_start_number : frame_end_number + 1]
        # import pdb

        # pdb.set_trace()
        
        multi_scale_time_slices=[0.5,1.0,1.5,2.0,2.5]
        for scale in multi_scale_time_slices:
            speaker_embedding_start = float(segment.start_time)
            speaker_embedding_end = float(segment.end_time)
            speaker_segment_length = speaker_embedding_end - speaker_embedding_start
            
            if (speaker_segment_length) < scale:
                temporal_adjustment = (speaker_segment_length / scale) / 2
                if temporal_adjustment + speaker_embedding_end < max_time:
                    speaker_embedding_end += temporal_adjustment

                if speaker_embedding_start - temporal_adjustment >= min_time:
                    speaker_embedding_start -= temporal_adjustment

            speaker_embedding_audio_array = audio_array[
                0,
                int(speaker_embedding_start * 16000) : int(speaker_embedding_end * 16000),
            ]
            embedding = generate_voice_embedding(speaker_embedding_audio_array)
            diarized_audio_segment = DiarizedAudioSegment(
                start_time=speaker_embedding_start,
                end_time=speaker_embedding_end,
                speaker_label=str(uuid4()),
                audio_array=speaker_embedding_audio_array,
                embedding=embedding,
            )
            all_audio_segment_list.append(diarized_audio_segment)
            speaker_embedding_start_frame = int(speaker_embedding_start * fps)
            speaker_embedding_end_frame = int(speaker_embedding_end * fps)
            relevent_frames_diarization = inferenced_frames[speaker_embedding_start_frame : speaker_embedding_end_frame + 1]
            for frame in relevent_frames_diarization:
                frame.diarized_audio_segments.append(diarized_audio_segment)
        for frame in relevent_frames_transcription:
            frame.transcribed_audio_segments.append(segment)
    return all_audio_segment_list


def main():
    ### VIDEO PROCESSING
    video_path = Path.cwd() / "test.mp4"
    video_path_str = str(video_path)
    url = "https://www.youtube.com/shorts/43NtLs-DgQ8?feature=share"

    if video_path.exists():
        pass
    else:
        video_path_str = download_video(url)
    VIDEO_CACHE_KEY = "dickbutt"
    value = cache.get(VIDEO_CACHE_KEY)
    if value is not None:
        yolo_frame_by_frame_index, yolo_track_id_index, fps = value
    else:
        yolo_frame_by_frame_index, yolo_track_id_index, fps = process_video(
            video_path_str
        )
        process_and_inject_identity_heuristics(yolo_track_id_index)
        cache.set(
            VIDEO_CACHE_KEY, (yolo_frame_by_frame_index, yolo_track_id_index, fps)
        )

    ##AUDIO PROCESSING
    audio_array, sampling_rate = extract_audio_from_mp4(video_path)

    transcription_segments = transcribe_audio(audio_array, 16000)

    diarized_audio_segments_list_index = frame_normalize_diarized_audio_segments(
        transcription_segments, fps, yolo_frame_by_frame_index, audio_array
    )

    edges = calculate_entity_relationships(yolo_frame_by_frame_index)
    for segment_i in diarized_audio_segments_list_index:
        for segment_j in diarized_audio_segments_list_index:
            if segment_i == segment_j:
                continue
            sim = cosine_similarity(segment_i.embedding, segment_j.embedding)
            edges.append(Edge(sim, (segment_i.speaker_label, segment_j.speaker_label)))
    for track_id_i, track_i in yolo_track_id_index.items():
        for track_id_j, track_j in yolo_track_id_index.items():
            if track_id_i == track_id_j:
                continue
            for i_embedding in track_i.face_embeddings:
                for j_embedding in track_j.face_embeddings:
                    sim = cosine_similarity(
                        i_embedding.embedding, j_embedding.embedding
                    )
                    edges.append(Edge(sim, (track_id_i, track_id_j)))
    with PostgresStorage() as psql:
        psql.reset_db()
        psql.run_setup()
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
        for diarized_segment in diarized_audio_segments_list_index:
            voice_embedding = diarized_segment.embedding
            embedding_relations = psql.query_embedding_similarity(
                "speaker", voice_embedding, top_n=10
            )
            for relation in embedding_relations:
                edges.append(
                    Edge(
                        relation["similarity"],
                        (diarized_segment.speaker_label, relation["id"]),
                    )
                )
        for diarized_segment in diarized_audio_segments_list_index:
            voice_embedding = diarized_segment.embedding
            speaker_label = str(diarized_segment.speaker_label)
            output_dir = "./temp/debug_output/audio"
            os.makedirs(output_dir, exist_ok=True)
            audio_data_path = os.path.abspath(
                os.path.join(output_dir, f"{speaker_label}.wav")
            )
            sf.write(audio_data_path, diarized_segment.audio_array, 16000)
            print(f"uploading {speaker_label}")
            psql.stage_insert_row(
                "node",
                {"id": speaker_label, "type": "spek", "media_path": audio_data_path},
            )

            psql.stage_insert_row(
                "speaker",
                {
                    "id": speaker_label,
                    "embedding": voice_embedding.tolist(),
                    "audio_data_path": audio_data_path,
                },
            )

        for transcribed_audio_segment in transcription_segments:
            output_dir = "./temp/debug_output/transcription"
            os.makedirs(output_dir, exist_ok=True)
            transcription_data_path = os.path.abspath(
                os.path.join(output_dir, f"{transcribed_audio_segment.uuid}.text")
            )
            with open(transcription_data_path, "w") as f:
                f.write(transcribed_audio_segment.transcription)

            psql.stage_insert_row(
                "node",
                {
                    "id": transcribed_audio_segment.uuid,
                    "type": "tran",
                    "media_path": transcription_data_path,
                },
            )
            psql.stage_insert_row(
                "transcribed_token",
                {
                    "id": transcribed_audio_segment.uuid,
                    "transcription": transcribed_audio_segment.transcription,
                    "start_time": transcribed_audio_segment.start_time,
                    "end_time": transcribed_audio_segment.end_time,
                    "probability": transcribed_audio_segment.probability,
                },
            )

        output_dir = "./temp/debug_output/track"
        os.makedirs(output_dir, exist_ok=True)
        for track_id, track in yolo_track_id_index.items():
            if track.face_embeddings is None:
                continue
            track_output_dir = f"./temp/debug_output/track/{track_id}/"
            os.makedirs(track_output_dir, exist_ok=True)
            image_data_path = os.path.abspath(
                os.path.join(track_output_dir, f"track_pic.png")
            )
            psql.stage_insert_row(
                "node", {"id": track_id, "type": "yobj", "media_path": image_data_path}
            )

            cv2.imwrite(image_data_path, track.sample.image_data)

            psql.stage_insert_row(
                "yolo_object", {"id": track_id, "image_data_path": image_data_path}
            )
            psql.stage_insert_row(
                "node", {"id": track_id, "type": "yobj", "media_path": image_data_path}
            )

            track_faces_output_dir = f"./temp/debug_output/track/{track_id}/faces"
            os.makedirs(track_faces_output_dir, exist_ok=True)
            for embedding in track.face_embeddings:
                face_image_data_path = os.path.abspath(
                    os.path.join(track_faces_output_dir, f"{embedding.uuid}.png")
                )
                cv2.imwrite(face_image_data_path, embedding.image_data)
                psql.stage_insert_row(
                    "node",
                    {
                        "id": embedding.uuid,
                        "type": "face",
                        "media_path": face_image_data_path,
                    },
                )
                psql.stage_insert_row(
                    "face",
                    {"id": embedding.uuid, "embedding": embedding.embedding.tolist()},
                )
                edges.append(Edge(1, (track_id, embedding.uuid)))

        edge_rows = []
        for edge in edges:
            v1, v2 = edge.vertexes
            if v1 != v2:
                edge_rows.append({"v1": v1, "v2": v2, "weight": float(edge.weight)})
        psql.stage_insert_many("edge", edge_rows)
        psql.tx_commit()


if __name__ == "__main__":
    main()
