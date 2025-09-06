import torch
from typing import List
from src.utils.AI_models import voice_embedding_model
import numpy as np 

assert voice_embedding_model is not None

def generate_voice_embedding(audio_array:List[np.ndarray]) -> List[float]:
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


