"""Minimal MLX decoder bridge for WhisperX's original VAD pipeline.

Keep upstream WhisperX VAD implementations and merge_chunks exactly as-is:
  WhisperX Pyannote VAD or Silero -> WhisperX VAD merge_chunks -> MLX Whisper
  -> WhisperX Wav2Vec2 forced alignment -> Pyannote Community-1 diaryzacja
  -> WhisperX assign_word_speakers.

ONLY the ASR decoder is replaced by mlx-whisper. All segment boundaries
before forced alignment belong to upstream WhisperX (not model output).
"""
from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from scipy.io import wavfile
from whisperx.audio import SAMPLE_RATE
from whisperx.log_utils import get_logger
from whisperx.vads import Pyannote, Silero

logger = get_logger(__name__)
# WhisperX merge_chunks/Binarize is unchanged. Only its public chunk_size
# parameter is tuned for MLX Whisper Turbo greedy decoding. The upstream 30s
# choice caused extreme repetition hallucinations on the verified recording;
# the previous 12s MLX path did not. Test WER, do not judge by E2E alone.
WHISPERX_VAD_PARAMS = {"chunk_size": 12, "vad_onset": .5, "vad_offset": .363}


def load_whisperx_vad(vad_backend: str, device: str = "mps"):
    """Instantiate upstream WhisperX VAD, with honest CPU fallback on MPS."""
    if vad_backend == "torch-silero":
        return Silero(**WHISPERX_VAD_PARAMS)
    if vad_backend != "whisperx-pyannote":
        raise ValueError("Unknown upstream WhisperX VAD backend: " + vad_backend)
    try:
        return Pyannote(torch.device(device), token=None, **WHISPERX_VAD_PARAMS)
    except Exception as exc:
        if device != "mps":
            raise
        logger.warning("WhisperX VAD failed to initialize on MPS (%s); retrying CPU", exc)
        return Pyannote(torch.device("cpu"), token=None, **WHISPERX_VAD_PARAMS)


def whisperx_speech_chunks(waveform: np.ndarray, vad_backend: str,
                           device: str = "mps"):
    """Use *unchanged* WhisperX preprocessing and merging of voice turns."""
    vad = load_whisperx_vad(vad_backend, device=device)
    try:
        preprocessed = vad.preprocess_audio(waveform)
        try:
            vad_output = vad({"waveform": preprocessed, "sample_rate": SAMPLE_RATE})
        except Exception as exc:
            if vad_backend != "whisperx-pyannote" or device != "mps":
                raise
            logger.warning("WhisperX VAD inference failed on MPS (%s); retrying CPU", exc)
            del vad
            vad = load_whisperx_vad(vad_backend, device="cpu")
            preprocessed = vad.preprocess_audio(waveform)
            vad_output = vad({"waveform": preprocessed, "sample_rate": SAMPLE_RATE})
        # Exact upstream WhisperX merge strategy, including pyannote's
        # Binarize/max_duration and the original segment boundary convention.
        chunks = vad.merge_chunks(
            vad_output, WHISPERX_VAD_PARAMS["chunk_size"],
            onset=WHISPERX_VAD_PARAMS["vad_onset"],
            offset=WHISPERX_VAD_PARAMS["vad_offset"],
        )
        return chunks
    finally:
        del vad


def transcribe_whisperx_mlx(audio_path: str, transcribe_clip, *, vad_backend: str,
                           vad_device: str = "mps", on_segment=None):
    """WhisperX VAD clips -> replace ONLY Whisper decoder with mlx-whisper.

    Each WhisperX chunk produces one candidate utterance; word timings
    are NEVER inferred from Whisper's own segment boundaries. They are
    subsequently assigned exclusively by WhisperX's Wav2Vec2 align().
    """
    from whisperx.audio import load_audio
    waveform = load_audio(audio_path)
    chunks = whisperx_speech_chunks(waveform, vad_backend, device=vad_device)
    segments = []
    if not chunks:
        return {"language": "pl", "segments": []}
    with TemporaryDirectory(prefix="whisperx-mlx-") as tmp:
        for index, chunk in enumerate(chunks):
            start, end = float(chunk["start"]), float(chunk["end"])
            a, b = int(start * SAMPLE_RATE), int(end * SAMPLE_RATE)
            chunk_audio = waveform[a:b]
            if len(chunk_audio) == 0:
                continue
            target = os.path.join(tmp, f"whisperx_chunk_{index:05d}.wav")
            pcm = np.clip(chunk_audio * 32767, -32768, 32767).astype(np.int16)
            wavfile.write(target, SAMPLE_RATE, pcm)
            decoded = transcribe_clip(target)
            text = " ".join(
                str(item.get("text", "")).strip()
                for item in decoded.get("segments", [])
                if item.get("text", "").strip()
            ).strip()
            if not text:
                continue
            # Match upstream WhisperX/FasterWhisperPipeline.transcribe() output.
            row = {
                "start": round(start, 3), "end": round(end, 3),
                "text": text,
            }
            segments.append(row)
            if on_segment is not None:
                on_segment(dict(row))
    return {"language": "pl", "segments": segments}
