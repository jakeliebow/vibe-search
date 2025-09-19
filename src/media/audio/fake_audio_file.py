import io
import numpy as np
import struct
from typing import Iterable, Tuple, Optional


def _to_int16_pcm(x: np.ndarray) -> np.ndarray:
    """Float [-1,1] or any dtype -> int16 PCM safely."""
    if not np.issubdtype(x.dtype, np.floating):
        x = x.astype(np.float32, copy=False)
    # normalize if out of range
    m = np.max(np.abs(x)) if x.size else 1.0
    if m > 1.0:
        x = x / m
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).round().astype(np.int16)


def _ensure_channels_first(x: np.ndarray) -> np.ndarray:
    """
    Accepts (T,), (T, C), (C, T) and returns (C, T).
    Mono becomes (1, T).
    """
    if x.ndim == 1:
        x = x[np.newaxis, :]
    elif x.ndim == 2:
        # pick the orientation with smaller first dim as channels
        C, T = x.shape
        if T < C:  # probably (T, C)
            x = x.T
    else:
        raise ValueError("audio array must be 1D or 2D")
    return x


def _wav_header(
    num_channels: int, sample_rate: int, num_frames: int, bits_per_sample: int = 16
) -> bytes:
    """Minimal RIFF/WAVE header (PCM)."""
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = num_frames * block_align
    riff_size = 36 + data_size

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return tag + struct.pack("<I", len(payload)) + payload

    fmt_payload = struct.pack(
        "<HHIIHH",
        1,  # PCM format
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
    )
    header = b"RIFF" + struct.pack("<I", riff_size) + b"WAVE"
    header += chunk(b"fmt ", fmt_payload)
    header += b"data" + struct.pack("<I", data_size)
    return header


def numpy_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Convert arbitrary float/np array to 16-bit PCM WAV bytes."""
    a = _ensure_channels_first(audio)
    a_i16 = _to_int16_pcm(a)
    # interleave channels to (T, C) then to bytes
    interleaved = a_i16.T.reshape(-1)  # (T, C) -> flat
    header = _wav_header(
        num_channels=a_i16.shape[0],
        sample_rate=int(sample_rate),
        num_frames=a_i16.shape[1],
    )
    return header + interleaved.tobytes()


class NumpyAudioFile(io.BytesIO):
    """
    A BytesIO that looks/acts like a file containing a WAV-encoded
    version of the provided NumPy audio. Useful when a library expects
    a file object (not a path) but you want to keep everything in memory.
    """

    def __init__(self, audio: np.ndarray, sample_rate: int, name: Optional[str] = None):
        wav_bytes = numpy_to_wav_bytes(audio, sample_rate)
        super().__init__(wav_bytes)
        # Many libs read .name to pick decoder/format; give it a WAV-ish name.
        self.name = name or "inmemory.wav"

    # Optional niceties: ensure context manager returns self and rewinds on enter
    def __enter__(self):
        self.seek(0)
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


# -------------------------
# Example usage:
# -------------------------
# audio_np: np.ndarray (T,), (T,C), or (C,T); sr: int
# from your pipeline:
# fobj = NumpyAudioFile(audio_np, sr)
# Pass `fobj` anywhere a file-like object is accepted:
#
#   import soundfile as sf
#   data, sr2 = sf.read(fobj)         # works
#
#   import librosa
#   fobj.seek(0)
#   y, sr3 = librosa.load(fobj, sr=None, mono=False)
#
#   import torchaudio
#   fobj.seek(0)
#   waveform, sr4 = torchaudio.load(fobj)
#
# If a lib insists on sniffing by extension, it will see .name="inmemory.wav".
