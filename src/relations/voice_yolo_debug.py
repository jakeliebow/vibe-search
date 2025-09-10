import os
import cv2
import numpy as np
import random
import soundfile as sf
from uuid import uuid4
from typing import List, Dict, Tuple
from src.models.detection import Detection
from src.models.audio import DiarizedAudioSegment
from src.relations.relate import Pairing
from src.media.video_shit_boxes.misc.image_helpers import get_frame_image


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _read_frame(video_path: str, frame_idx: int) -> np.ndarray:
    """Read a specific frame from video"""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read frame {frame_idx}")
    return frame  # BGR


def get_example_face_images(
    pairing: Pairing,
    inferenced_frames: List,
    video_path: str,
    max_samples: int = 5,
) -> List[Tuple[np.ndarray, int]]:
    """
    Extract example face images for a voice-yolo pairing.

    Args:
        pairing: Voice-yolo pairing from relate.py
        inferenced_frames: List of frames with detections and audio segments
        video_path: Path to the source video file
        max_samples: Maximum number of samples to extract

    Returns:
        List of tuples (face_crop, frame_number)
    """
    # Find frames where this pairing occurs
    relevant_frames = []
    for frame in inferenced_frames:
        # Check if frame has both the speaker and object
        has_speaker = any(
            seg.speaker_label == pairing.speaker
            for seg in frame.diarized_audio_segments
        )
        has_object = any(
            det.yolo_object_id == int(pairing.object_id)
            for det in frame.detections
            if det.yolo_object_id is not None
        )

        if has_speaker and has_object:
            relevant_frames.append(frame)

    if not relevant_frames:
        return []

    # Extract face shots (up to max_samples random frames)
    face_frames = [
        f
        for f in relevant_frames
        if any(
            det.face is not None and det.yolo_object_id == int(pairing.object_id)
            for det in f.detections
        )
    ]

    if not face_frames:
        return []

    sample_face_frames = random.sample(face_frames, min(max_samples, len(face_frames)))

    face_crops = []
    for frame in sample_face_frames:
        # Find the detection for this object in this frame
        target_detection = None
        for det in frame.detections:
            if det.yolo_object_id == int(pairing.object_id) and det.face is not None:
                target_detection = det
                break

        if target_detection is None:
            continue

        try:
            # Read frame and crop face
            frame_image = _read_frame(video_path, frame.frame_number)

            # Use face bounding box for more precise cropping
            face_box = target_detection.face.face_box
            x1 = int(face_box.x1)
            y1 = int(face_box.y1)
            x2 = int(face_box.x2)
            y2 = int(face_box.y2)

            if x2 <= x1 or y2 <= y1:
                continue  # degenerate box

            face_crop = frame_image[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue

            face_crops.append((face_crop, frame.frame_number))

        except Exception as e:
            print(f"    Error extracting face: {e}")
            continue

    return face_crops


def get_example_audio_segments(
    pairing: Pairing,
    inferenced_frames: List,
    max_samples: int = 5,
) -> List[DiarizedAudioSegment]:
    """
    Extract example audio segments for a voice-yolo pairing.

    Args:
        pairing: Voice-yolo pairing from relate.py
        inferenced_frames: List of frames with detections and audio segments
        max_samples: Maximum number of samples to extract

    Returns:
        List of DiarizedAudioSegment objects
    """
    # Find frames where this pairing occurs
    relevant_frames = []
    for frame in inferenced_frames:
        # Check if frame has both the speaker and object
        has_speaker = any(
            seg.speaker_label == pairing.speaker
            for seg in frame.diarized_audio_segments
        )
        has_object = any(
            det.yolo_object_id == int(pairing.object_id)
            for det in frame.detections
            if det.yolo_object_id is not None
        )

        if has_speaker and has_object:
            relevant_frames.append(frame)

    if not relevant_frames:
        return []

    # Extract audio clips (up to max_samples random segments)
    audio_segments = []
    for frame in relevant_frames:
        for seg in frame.diarized_audio_segments:
            if seg.speaker_label == pairing.speaker and seg.audio_array is not None:
                audio_segments.append(seg)

    # Remove duplicates based on start_time and end_time
    unique_segments = []
    seen_times = set()
    for seg in audio_segments:
        time_key = (seg.start_time, seg.end_time)
        if time_key not in seen_times:
            unique_segments.append(seg)
            seen_times.add(time_key)

    if not unique_segments:
        return []

    sample_audio_segments = random.sample(
        unique_segments, min(max_samples, len(unique_segments))
    )

    return sample_audio_segments


def debug_output(
    pairings: List[Pairing],
    face_images_dict: Dict[int, List[Tuple[np.ndarray, int]]],
    audio_segments_dict: Dict[int, List[DiarizedAudioSegment]],
    video_path: str,
    output_base_dir: str = "debug_output",
) -> str:
    """
    Save debug output for voice-yolo pairings.

    Args:
        pairings: List of voice-yolo pairings from relate.py
        face_images_dict: Dictionary mapping pairing index to list of (face_crop, frame_number) tuples
        audio_segments_dict: Dictionary mapping pairing index to list of audio segments
        video_path: Path to the source video file
        output_base_dir: Base directory for output

    Returns:
        Path to the generated debug folder
    """
    # Create unique output directory
    debug_id = str(uuid4())
    debug_dir = os.path.join(output_base_dir, debug_id)
    face_shots_dir = os.path.join(debug_dir, "face_shots")
    audio_clips_dir = os.path.join(debug_dir, "audio_clips")

    os.makedirs(face_shots_dir, exist_ok=True)
    os.makedirs(audio_clips_dir, exist_ok=True)

    print(f"Creating debug output in: {debug_dir}")
    print(f"Found {len(pairings)} voice-yolo pairings to debug")

    for pairing_idx, pairing in enumerate(pairings):
        print(
            f"Processing pairing {pairing_idx + 1}/{len(pairings)}: Speaker {pairing.speaker} <-> Object {pairing.object_id}"
        )

        # Save face images
        face_images = face_images_dict.get(pairing_idx, [])
        for face_idx, (face_crop, frame_number) in enumerate(face_images):
            try:
                face_filename = f"pairing_{pairing_idx}_speaker_{pairing.speaker}_object_{pairing.object_id}_face_{face_idx}_frame_{frame_number}.jpg"
                face_path = os.path.join(face_shots_dir, face_filename)
                cv2.imwrite(face_path, face_crop)
            except Exception as e:
                print(f"    Error saving face shot {face_idx}: {e}")
                continue

        # Save audio segments
        audio_segments = audio_segments_dict.get(pairing_idx, [])
        for audio_idx, segment in enumerate(audio_segments):
            try:
                if (
                    segment.audio_array is not None
                    and segment.sampling_rate is not None
                ):
                    audio_filename = f"pairing_{pairing_idx}_speaker_{pairing.speaker}_object_{pairing.object_id}_audio_{audio_idx}_start_{segment.start_time:.2f}s.wav"
                    audio_path = os.path.join(audio_clips_dir, audio_filename)
                    sf.write(
                        audio_path, segment.audio_array, int(segment.sampling_rate)
                    )
            except Exception as e:
                print(f"    Error saving audio clip {audio_idx}: {e}")
                continue

    # Create summary file
    summary_path = os.path.join(debug_dir, "pairings_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Voice-YOLO Object Pairings Debug Output\n")
        f.write(f"Generated: {debug_id}\n")
        f.write(f"Video source: {video_path}\n")
        f.write(f"Total pairings: {len(pairings)}\n\n")

        for idx, pairing in enumerate(pairings):
            f.write(
                f"Pairing {idx}: Speaker '{pairing.speaker}' <-> Object ID {pairing.object_id}\n"
            )
            f.write(f"  Avg MAR derivative: {pairing.avg_abs_mar_derivative:.4f}\n")
            f.write(f"  Co-present frames: {pairing.frames}\n\n")

    print(f"Debug output completed: {debug_dir}")
    return debug_dir


def debug_voice_yolo_pairings(
    pairings: List[Pairing],
    inferenced_frames: List,
    video_path: str,
    max_samples: int = 5,
    output_base_dir: str = "debug_output",
) -> str:
    """
    Debug function to output voice-yolo object pairings with face shots and audio clips.

    Args:
        pairings: List of voice-yolo pairings from relate.py
        inferenced_frames: List of frames with detections and audio segments
        video_path: Path to the source video file
        max_samples: Maximum number of samples per pairing (default 5)
        output_base_dir: Base directory for output (default "debug_output")

    Returns:
        Path to the generated debug folder
    """
    # Extract face images and audio segments for each pairing
    face_images_dict = {}
    audio_segments_dict = {}

    for pairing_idx, pairing in enumerate(pairings):
        face_images_dict[pairing_idx] = get_example_face_images(
            pairing, inferenced_frames, video_path, max_samples
        )
        audio_segments_dict[pairing_idx] = get_example_audio_segments(
            pairing, inferenced_frames, max_samples
        )

    # Save debug output
    return debug_output(
        pairings, face_images_dict, audio_segments_dict, video_path, output_base_dir
    )
