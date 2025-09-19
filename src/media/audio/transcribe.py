import numpy as np
from typing import List, Dict, Any
from src.state.AI.audio_models import whisper_model
from src.models.audio import TranscribedAudioSegment
from src.media.audio.extract import extract_audio_from_mp4
import numpy as np
from typing import List, Dict, Any
from scipy.signal import resample_poly
from uuid import uuid4

# WhisperX for improved performance and word-level alignment
import whisperx

TARGET_SR = 16000


def _to_mono_1d(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 1:
        return arr
    # pick the smaller dim as channels
    t0, t1 = arr.shape
    if t0 < t1:  # (channels, time)
        return arr.mean(axis=0)
    else:  # (time, channels)
        return arr.mean(axis=1)


def _normalize_f32(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    m = np.max(np.abs(x)) if x.size else 0.0
    return x / m if m > 1.0 else x


def _resample_to_16k(x: np.ndarray, sr: int) -> np.ndarray:
    if sr == TARGET_SR:
        return x
    # polyphase resample
    from math import gcd

    g = gcd(sr, TARGET_SR)
    up, down = TARGET_SR // g, sr // g
    return resample_poly(x, up, down).astype(np.float32, copy=False)


def transcribe_audio(
    audio_array: np.ndarray,
    sampling_rate: int,
) -> List[TranscribedAudioSegment]:
    # shape -> mono
    x = _to_mono_1d(audio_array)
    # resample -> 16k
    x = _resample_to_16k(x, sampling_rate)
    # normalize (keep within [-1,1])
    x = _normalize_f32(x)
    # quick silence/NaN guard
    if not x.size or np.isnan(x).any() or np.max(np.abs(x)) < 1e-6:
        return []

    # WhisperX transcription
    result = whisper_model.transcribe(x, batch_size=16)

    model_a, metadata = whisperx.load_align_model(
        language_code="en", device=whisper_model.device
    )
    result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        x,
        whisper_model.device,
        return_char_alignments=False,
    )

    out: List[TranscribedAudioSegment] = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            out.append(
                TranscribedAudioSegment(
                    start_time=float(w["start"]),
                    end_time=float(w["end"]),
                    transcription=str(w["word"]).strip(),
                    probability=float(w.get("score", 1.0)),
                    uuid=str(uuid4()),
                )
            )
    return out


if __name__ == "__main__":
    arr, sr = extract_audio_from_mp4("/Users/jakeliebow/vibe-search/test.mp4")
    out = transcribe_audio(arr, sr)
    print(out)
