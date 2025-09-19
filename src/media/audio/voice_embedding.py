import torch
from src.state.AI.audio_models import voice_embedding as voice_embedding_model
import numpy as np
from src.media.audio.fake_audio_file import NumpyAudioFile


def generate_voice_embedding(audio: np.ndarray) -> np.ndarray:
    """
    Extract Titanet speaker embedding from audio in numpy format.

    Args:
        audio: np.ndarray of shape (time,) or (channels, time), float32 in [-1,1].
        sr: sample rate of audio (Titanet expects 16kHz).

    Returns:
        np.ndarray of shape (embedding_dim,)
    """

    psuedo_file = NumpyAudioFile(audio, 16000)
    with torch.no_grad():
        emb = voice_embedding_model.get_embedding(psuedo_file)
    return emb.squeeze(0).cpu().numpy()
