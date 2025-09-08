import os
import torch
from pyannote.audio import Pipeline
from speechbrain.inference import SpeakerRecognition
from .lazy_loader import LazyLoader
import whisper

huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
if not huggingface_token:
    raise ValueError(
        "HUGGINGFACE_TOKEN environment variable is required for speaker diarization"
    )
device: str = "cuda" if torch.cuda.is_available() else "cpu"

# Audio models
diarization_hugging_face_model_name = "pyannote/speaker-diarization-3.1"
def setup_diarizer(diarizer_model_tmp):
    if torch.cuda.is_available():
        diarizer_model_tmp.to(torch.device(device))
    return diarizer_model_tmp
diarizer_model = LazyLoader(Pipeline.from_pretrained, diarization_hugging_face_model_name, use_auth_token=huggingface_token,
                            _setup=setup_diarizer)



whisper_model_size = "base"
whisper_model = LazyLoader(whisper.load_model, whisper_model_size, device=device)
voice_embedding_model = LazyLoader(SpeakerRecognition.from_hparams, source="speechbrain/spkrec-xvect-voxceleb", savedir="temp/pretrained_models/spkrec-xvect-voxceleb")
