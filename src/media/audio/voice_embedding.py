import torch
from typing import List,Dict
from src.state.AI.audio_models import voice_embedding_model
from src.models.audio import DiarizedAudioSegment,SpeakerTrack
import numpy as np 

def generate_voice_embedding(audio_array:np.ndarray) -> List[float]:
    audio_tensor = torch.from_numpy(audio_array).float()
    # Ensure correct shape (add batch dimension if needed)
    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    
    # Check minimum length requirement (at least 0.5 seconds at 16kHz = 8000 samples)
    min_samples = 8000
    if audio_tensor.shape[-1] < min_samples:
        # Pad the audio to minimum length
        padding_needed = min_samples - audio_tensor.shape[-1]
        audio_tensor = torch.nn.functional.pad(audio_tensor, (0, padding_needed), mode='constant', value=0)
    
    embeddings = voice_embedding_model.encode_batch(audio_tensor)
    
    # Convert to list
    embedding = embeddings.squeeze().detach().cpu().numpy().tolist()#cpu bound, make dynamic

    return embedding


def compute_voice_embeddings_per_speaker(diarized_audio_segments_by_speaker_track_index:Dict[str,SpeakerTrack]):
    for speaker_label,speaker_track in diarized_audio_segments_by_speaker_track_index.items():
        # Concatenate all audio arrays for the speaker into a single numpy array
        # Assuming all segments have the same sampling rate and are compatible for concatenation
        if speaker_track.segments and all(s.audio_array is not None for s in speaker_track.segments):
            concatenated_audio_array = np.concatenate([segment.audio_array for segment in speaker_track.segments])
            speaker_track.voice_embedding = generate_voice_embedding(concatenated_audio_array)
        else:
            raise Exception("speaker track has no audio segments or bad audio segments")
    return diarized_audio_segments_by_speaker_track_index
    
