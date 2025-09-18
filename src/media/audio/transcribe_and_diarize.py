import numpy as np
import torch
from uuid import uuid4
from typing import List, Dict, Any,Tuple
from src.state.AI.audio_models import diarizer_model, whisper_model
from src.models.audio import DiarizedAudioSegment,SpeakerTrack,TranscribedAudioSegment
import subprocess
import os
import time

def extract_audio_from_mp4(mp4_path: str, sampling_rate: int = 16000) -> tuple[np.ndarray, int]:
    """
    Extract audio from MP4 as NumPy array suitable for diarization.
    
    Returns (channels, time), float32 in [-1, 1].
    Handles mono or multi-channel automatically.
    """
    id=str(uuid4())
    
    os.makedirs("temp/working_audio",exist_ok=True)
    command = f"ffmpeg -i {mp4_path} -ab 160k -ac 2 -ar {sampling_rate} -vn temp/working_audio/{id}.wav"

    subprocess.call(command, shell=True)
    wav_file_path = f"temp/working_audio/{id}.wav"
    # Poll until the file exists
    while not os.path.exists(wav_file_path):
        print(f"Waiting for {wav_file_path} to be created...")
        time.sleep(1)  # Wait for 1 second before checking again

    audio_array, sampling_rate = torchaudio.load(f"temp/working_audio/{id}.wav")    
    audio_array = audio_array.numpy()

    return audio_array, sampling_rate



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
    label_uuid_map={}
    for segment in diarized_audio_segments:
        speaker_label = str(segment.speaker_label)
        
        if speaker_label not in label_uuid_map:
            uuid=str(uuid4())
            label_uuid_map[speaker_label]=uuid
            speaker_tracks[uuid] = SpeakerTrack(
                speaker_label=speaker_label, segments=[], voice_embedding=None, audio_data=None
            )
        segment.speaker_label=label_uuid_map[speaker_label]
        speaker_tracks[label_uuid_map[speaker_label]].segments.append(segment)
    return speaker_tracks

def diarize_audio(audio_array: np.ndarray, sampling_rate: int | float) -> List[DiarizedAudioSegment]:
    """
    Perform speaker diarization on audio array.
    """

    
    diarization = diarizer_model(
        {"waveform": torch.from_numpy(audio_array), "sample_rate": sampling_rate}
    )
    
    diarization_result = [
        DiarizedAudioSegment(
            start_time=speech_turn.start,
            end_time=speech_turn.end,
            speaker_label=speaker_label,
            audio_array=audio_array[
                0,
                int(speech_turn.start * sampling_rate): int(speech_turn.end * sampling_rate)
            ]
        )
        for speech_turn, _, speaker_label in diarization.itertracks(yield_label=True)
    ]
    
    return diarization_result

def transcribe_audio(audio_array: np.ndarray) -> List[TranscribedAudioSegment]:
    """
    Transcribe audio using Whisper model.
    """
    # Use the whisper model to transcribe the entire audio
    transcription: Dict[str, Any] = whisper_model.transcribe(audio_array)

    transcription_segments = [
        TranscribedAudioSegment(
            start_time=float(segment["start"]),
            end_time=float(segment["end"]),
            transcription=str(segment["text"]).strip(),
        ) for segment in transcription["segments"]
    ]
    
    return transcription_segments


def add_audio_arrays_to_segments(
    segments: List[DiarizedAudioSegment], 
    full_audio_array: np.ndarray, 
    sampling_rate: int | float
) -> List[DiarizedAudioSegment]:
    """
    Extract audio arrays for each segment from the full audio array and add them to the segments in memory.
    
    Args:
        segments: List of DiarizedAudioSegment objects
        full_audio_array: Complete audio array from the source file
        sampling_rate: Sampling rate of the audio
    
    Returns:
        List of segments with audio_array and sampling_rate populated
    """
    
    for index,segment in enumerate(segments):
        # Convert time to sample indices
        start_sample = int(segment.start_time * sampling_rate)
        end_sample = int(segment.end_time * sampling_rate)
        
        # Ensure indices are within bounds
        start_sample = max(0, start_sample)
        end_sample = min(len(full_audio_array), end_sample)
        segment_audio = full_audio_array[start_sample:end_sample]
        

        segments[index].sampling_rate=sampling_rate
        segments[index].audio_array=segment_audio
    
    return segments
