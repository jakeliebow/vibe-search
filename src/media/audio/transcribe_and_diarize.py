import librosa
import numpy as np
import torch
from typing import List, Dict, Any
from src.state.AI.audio_models import diarizer_model, whisper_model
from src.models.audio import DiarizedAudioSegment

def transcribe_and_diarize_audio(video_path: str) -> List[DiarizedAudioSegment]:
    """
    Transcribe and diarize audio, merging transcription segments into diarization segments.
    """
    audio_array, sampling_rate = librosa.load(video_path,sr=16000)

    diarization_segments = diarize_audio(audio_array, sampling_rate)
    transcription_segments = transcribe_audio(audio_array)

    merged_segments = merge_transcription_with_diarization(diarization_segments, transcription_segments)
    
    segments_with_audio = add_audio_arrays_to_segments(merged_segments, audio_array, sampling_rate)
    return segments_with_audio


def merge_transcription_with_diarization(
    diarization_segments: List[DiarizedAudioSegment], 
    transcription_segments: List[DiarizedAudioSegment]
) -> List[DiarizedAudioSegment]:
    """
    Merge transcription text into diarization segments based on time overlap.
    O(N+M) implementation using two-pointer technique.
    """
    if not diarization_segments or not transcription_segments:
        return diarization_segments
    
    # Sort both lists by start time (should already be sorted, but ensure it)
    diar_sorted = sorted(diarization_segments, key=lambda x: x.start_time)
    trans_sorted = sorted(transcription_segments, key=lambda x: x.start_time)
    
    merged_segments = []
    trans_idx = 0
    
    for diar_segment in diar_sorted:
        overlapping_texts = []
        
        # Move transcription pointer to first potentially overlapping segment
        while (trans_idx < len(trans_sorted) and 
               trans_sorted[trans_idx].end_time <= diar_segment.start_time):
            trans_idx += 1
        
        # Check all transcription segments that could overlap with current diarization segment
        temp_idx = trans_idx
        while (temp_idx < len(trans_sorted) and 
               trans_sorted[temp_idx].start_time < diar_segment.end_time):
            
            trans_segment = trans_sorted[temp_idx]
            
            # Check for time overlap
            overlap_start = max(diar_segment.start_time, trans_segment.start_time)
            overlap_end = min(diar_segment.end_time, trans_segment.end_time)
            
            if overlap_start < overlap_end:  # There is overlap
                overlap_duration = overlap_end - overlap_start
                trans_duration = trans_segment.end_time - trans_segment.start_time
                
                # Only include if significant overlap (>50% of transcription segment)
                if overlap_duration / trans_duration > 0.5:
                    overlapping_texts.append(trans_segment.transcription)
            
            temp_idx += 1
        
        # Combine overlapping transcriptions
        combined_text = " ".join(overlapping_texts).strip()
        
        # Create merged segment
        merged_segment = DiarizedAudioSegment(
            speaker_label=diar_segment.speaker_label,
            start_time=diar_segment.start_time,
            end_time=diar_segment.end_time,
            transcription=combined_text,
            asset_id=diar_segment.asset_id,
            asset_type=diar_segment.asset_type
        )
        
        merged_segments.append(merged_segment)
    
    return merged_segments


def diarize_audio(audio_array: np.ndarray, sampling_rate: int | float) -> List[DiarizedAudioSegment]:
    """
    Perform speaker diarization on audio array.
    """
    # Reshape audio array for pyannote (needs to be 2D)
    if len(audio_array.shape) == 1:
        audio_array = audio_array.reshape(1, -1)
    
    diarization = diarizer_model(#network request
        {"waveform": torch.from_numpy(audio_array), "sample_rate": sampling_rate}
    )
    
    diarization_result = [
        DiarizedAudioSegment(
            start_time=speech_turn.start,
            end_time=speech_turn.end,
            speaker_label=speaker_label,
            transcription="",  # Will be filled by transcription merge
            asset_id=None,
            asset_type=None
        )
        for speech_turn, _, speaker_label in diarization.itertracks(yield_label=True)
    ]

    return diarization_result

def transcribe_audio(audio_array: np.ndarray) -> List[DiarizedAudioSegment]:
    """
    Transcribe audio using Whisper model.
    """
    # Use the whisper model to transcribe the entire audio
    transcription: Dict[str, Any] = whisper_model.transcribe(audio_array)

    transcription_segments = [
        DiarizedAudioSegment(
            speaker_label="",
            start_time=float(segment["start"]),
            end_time=float(segment["end"]),
            transcription=str(segment["text"]).strip(),
            asset_id=None,
            asset_type=None
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
