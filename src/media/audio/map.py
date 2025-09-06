from typing import List,Dict
from src.models.audio import DiarizedAudioSegment

def get_speaker_embedding_map(diarized_audio_segments:List[DiarizedAudioSegment])->Dict[str,List[float]]:
    """
    Create a mapping from speaker labels to their voice embeddings.
    
    Args:
        diarized_audio_segments: List of diarized audio segments with speaker labels and embeddings
        
    Returns:
        Dictionary mapping speaker labels to their voice embeddings
    """
    speaker_embedding_map = {}
    
    for segment in diarized_audio_segments:
        if segment.voice_embedding is not None:
            speaker_embedding_map[segment.speaker_label] = segment.voice_embedding
    
    return speaker_embedding_map