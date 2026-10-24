"""
Backend abstractions for WhisperX to support multiple inference engines.

This module provides pluggable backends for:
- ASR (Automatic Speech Recognition): MLX (Apple Silicon) or CTranslate2 (x86/GPU)
- VAD (Voice Activity Detection): MLX Silero or PyTorch pyannote/silero
- Diarization: PyTorch pyannote (CPU or MPS)

The goal is to optimize for Apple Silicon (MLX) while maintaining compatibility
with other platforms.
"""

from .asr_backend import ASRBackend, MLXASRBackend, CTranslate2ASRBackend
from .vad_backend import VADBackend, SileroMLXBackend, PyannoteVADBackend
from .diarization_backend import BaseDiarizationBackend, PyannoteDiarizationBackend

__all__ = [
    'ASRBackend',
    'MLXASRBackend',
    'CTranslate2ASRBackend',
    'VADBackend',
    'SileroMLXBackend',
    'PyannoteVADBackend',
    'DiarizationBackend',
    'PyannoteDiarizationBackend',
]
