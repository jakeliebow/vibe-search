import numpy as np
from uuid import uuid4
import subprocess
import os
import time
import torchaudio

def extract_audio_from_mp4(mp4_path: str, sampling_rate: int = 16000) -> tuple[np.ndarray, int]:
    """
    Extract audio from MP4 as NumPy array suitable for diarization.
    
    Returns (channels, time), float32 in [-1, 1].
    Handles mono or multi-channel automatically.
    """
    id=str(uuid4())
    
    os.makedirs("temp/working_audio",exist_ok=True)
    command = f"ffmpeg -i {mp4_path} -ab 160k -ac 2 -ar {sampling_rate} -vn temp/working_audio/{id}.wav"

    subprocess.call(command, shell=True)
    wav_file_path = f"temp/working_audio/{id}.wav"
    # Poll until the file exists
    while not os.path.exists(wav_file_path):
        print(f"Waiting for {wav_file_path} to be created...")
        time.sleep(1)  # Wait for 1 second before checking again

    audio_array, sampling_rate = torchaudio.load(f"temp/working_audio/{id}.wav")    
    audio_array = audio_array.numpy()

    return audio_array, sampling_rate