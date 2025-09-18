#!/usr/bin/env python3
"""
Voice/Speaker Schemas
Pydantic models for voice recognition and speaker data structures.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
import numpy as np
from pydantic import BaseModel, Field



class TranscribedAudioSegment(BaseModel):
    """
    Model for a transcribed audio segment.

    Attributes:
        transcription (str): The transcribed text of the segment.
        start_time (float): Start time of the transcribed segment in seconds.
        end_time (float): End time of the transcribed segment in seconds.
    """

    transcription: str = Field(..., description="The transcribed text of the segment.")
    start_time: float = Field(
        ..., description="Start time of the transcribed segment in seconds."
    )
    end_time: float = Field(
        ..., description="End time of the transcribed segment in seconds."
    )
    class Config:
        arbitrary_types_allowed = True


class DiarizedAudioSegment(BaseModel):
    """
    Model for a diarized audio segment with speaker identification and transcription.

    Attributes:
        speaker_label (str): Local reference to the speaker from the diarizer model (e.g., SPEAKER_00)
        start_time (float): Start time of diarized segment in seconds
        end_time (float): End time of diarized segment in seconds
    """

    speaker_label: str = Field(
        ...,
        description="local reference to the speaker from the diarizer model, e.g., SPEAKER_00",
    )
    start_time: float = Field(..., description="start time of diarized segment")
    end_time: float = Field(..., description="end time of diarized segment")
    audio_array: np.ndarray=Field(...,description="fuck")
    class Config:
        arbitrary_types_allowed = True


class SpeakerTrack(BaseModel):
    """
    Represents a tracked speaker across multiple audio segments.

    This model aggregates individual diarized audio segments of the same speaker,
    allowing for analysis of their speech patterns and content throughout the audio.

    Attributes:
        speaker_label (str): Speaker uuid from the diarizer model
        segments (List[DiarizedAudioSegment]): List of audio segments for this speaker
        voice_embedding: Optional[List[float]] = Field(
        default=None, description="voice embedding"
    )
    audio_data: Optional[np.ndarray] = Field(..., description="audio data")
    """

    speaker_label: str = Field(
        ..., description="Speaker identifier from the diarizer model"
    )
    segments: List[DiarizedAudioSegment] = Field(
        ..., description="List of audio segments for this speaker"
    )
    voice_embedding: Optional[List[float]] = Field(
        default=None, description="voice embedding"
    )
    audio_data: Optional[np.ndarray] = Field(default=None, description="audio data")

    class Config:
        arbitrary_types_allowed = True
