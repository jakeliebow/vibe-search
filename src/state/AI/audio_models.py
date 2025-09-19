import os
import torch
from .lazy_loader import LazyLoader
import whisperx
import numpy as np
from nemo.collections.asr.models import EncDecSpeakerLabelModel

huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
device: str = "cuda" if torch.cuda.is_available() else "cpu"

voice_embedding = EncDecSpeakerLabelModel.from_pretrained("titanet_large")
voice_embedding.eval()

whisper_model_size = "base"
whisper_model = LazyLoader(
    whisperx.load_model, whisper_model_size, device=device, compute_type="float32"
)
