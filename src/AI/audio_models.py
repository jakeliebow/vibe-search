import os
import torch
import whisper
from pyannote.audio import Pipeline
from speechbrain.inference import SpeakerRecognition

huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
if not huggingface_token:
    raise ValueError(
        "HUGGINGFACE_TOKEN environment variable is required for speaker diarization"
    )
device: str = "cuda" if torch.cuda.is_available() else "cpu"

voice_embedding_model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-xvect-voxceleb",
    savedir="temp/pretrained_models/spkrec-xvect-voxceleb",
)

# Audio models
diarization_hugging_face_model_name = "pyannote/speaker-diarization-3.1"
diarizer_model = Pipeline.from_pretrained(
    diarization_hugging_face_model_name, use_auth_token=huggingface_token
)
if torch.cuda.is_available():
    diarizer_model.to(torch.device(device))

whisper_model_size = "base"
whisper_model = whisper.load_model(whisper_model_size, device=device)
