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


class AssetType(str, Enum):
    """Enum for asset types"""
    AUDIO = "audio"
    VIDEO = "video"


class DiarizedAudioSegment(BaseModel):
    """
    Model for a diarized audio segment with speaker identification and transcription.
    
    Attributes:
        speaker_label (str): Local reference to the speaker from the diarizer model (e.g., SPEAKER_00)
        start_time (float): Start time of diarized segment in seconds
        end_time (float): End time of diarized segment in seconds
        transcription (str): Transcription of the diarized segment
        asset_id (str | None): Asset ID of the source of the diarized segment
        asset_type (AssetType | None): Asset type of the source (audio/video)
        voice_embedding (Optional[List[float]]): Voice embedding vector for speaker identification
        audio_array (Optional[np.ndarray]): Raw audio data for this segment
        sampling_rate (Optional[int | float]): Sampling rate of the audio data in Hz
    """
    start_time: float = Field(..., description="start time of diarized segment")
    end_time: float = Field(..., description="end time of diarized segment")
    transcription: str = Field(..., description="transcription of the diarized segment")
    asset_id: str | None = Field(..., description="asset id of the source of the diarized segment")
    asset_type: AssetType | None = Field(..., description="asset type of the source of the diarized segment")


    voice_embedding: Optional[List[float]]= Field(default=None,description="voice embedding")
    audio_array: Optional[np.ndarray] = Field(default=None, description="Audio data for this segment")
    sampling_rate: Optional[int | float] = Field(default=None, description="Sampling rate of the audio data")
    
    class Config:
        arbitrary_types_allowed = True