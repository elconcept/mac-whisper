"""
Transcription pipeline with MLX ASR for Apple Silicon.

Uses MLX (mlx-whisper) for ASR on Apple Silicon GPU for maximum speed,
with automatic MPS fallback for diarization.
"""

import os
import platform
import torch
import gc
import psutil
from typing import Optional, Dict, Any
from pathlib import Path

from whisperx.audio import load_audio, SAMPLE_RATE
from whisperx.alignment import load_align_model, align
from whisperx.diarize import assign_word_speakers, DiarizationPipeline
from whisperx.log_utils import get_logger
from whisperx.utils import exact_div
from whisperx.memory_guard import cleanup_unified_memory, check_memory_safety

logger = get_logger(__name__)

logger = get_logger(__name__)


def cleanup_unified_memory(force: bool = True):
    """Aggressive memory cleanup for Apple Silicon Unified Memory."""
    gc.collect()
    try:
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except:
        pass
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass
    try:
        import mlx.core as mx
        mx.clear_cache()
    except:
        pass
    if force:
        gc.collect()


def check_memory_safety(threshold_gb: float = 2.0) -> bool:
    """Check if system has enough free RAM."""
    free_gb = psutil.virtual_memory().available / (1024 ** 3)
    if free_gb < threshold_gb:
        logger.warning(f"Low memory ({free_gb:.2f}GB), forcing cleanup...")
        cleanup_unified_memory(force=True)
    return free_gb >= threshold_gb


class TranscriptionPipeline:
    """
    Transcription pipeline with MLX ASR for Apple Silicon.

    Uses mlx-whisper for GPU-accelerated ASR on Apple Silicon (MPS),
    with automatic MPS-to-CPU fallback for diarization.
    """

    def __init__(
        self,
        asr_model: str = "large-v3-turbo",
        device: str = "auto",
        hf_token: Optional[str] = None,
        language: str = "pl",
        batch_size: int = 16,
        use_mps_for_diarization: bool = True,
        diarization_backend: str = "pyannote",
    ):
        """
        Initialize transcription pipeline.

        Args:
            asr_model: Whisper model name
            device: Device for ASR (auto/cuda/cpu)
            hf_token: HuggingFace token for pyannote
            language: Default language
            batch_size: Batch size for CTranslate2
            use_mps_for_diarization: Try MPS for diarization (with CPU fallback)
            diarization_backend: Diarization backend type ("pyannote" or "mlx_nemotron")
        """
        self.asr_model = asr_model
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.language = language
        self.batch_size = batch_size
        self.use_mps_for_diarization = use_mps_for_diarization
        self.diarization_backend = diarization_backend

        # Detect platform and auto-select ASR backend
        is_apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"

        if is_apple_silicon:
            # Apple Silicon: Use MLX (mlx-whisper) on MPS
            self.asr_backend = "mlx"
            self.device = "mps"
            logger.info("Apple Silicon detected, using MLX for ASR on MPS")
        else:
            # x86/Linux/Windows: Use CTranslate2
            self.asr_backend = "ctranslate2"
            self.device = device if device != "auto" else "cuda"
            logger.info(f"x86 detected, using CTranslate2 on {self.device}")

        # Diarization device
        if is_apple_silicon and use_mps_for_diarization:
            self.diarization_device = "mps"
        else:
            self.diarization_device = "cpu" if is_apple_silicon else self.device

        # Models
        self.model = None
        self.align_model = None
        self.align_metadata = None
        self.diarize_model = None

        # MPS crash tracking
        self.mps_crashed = False

        logger.info(f"Pipeline initialized:")
        logger.info(f"  ASR: {self.asr_backend} on {self.device}")
        logger.info(f"  Diarization: Pyannote on {self.diarization_device} (with CPU fallback)")

    def transcribe_mlx(
        self,
        audio: str,
        language: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Transcribe using MLX (mlx-whisper) on Apple Silicon.

        Args:
            audio: Path to audio file
            language: Language code
            **kwargs: Additional parameters

        Returns:
            Dictionary with transcription results
        """
        try:
            import mlx_whisper
        except ImportError:
            raise ImportError(
                "mlx-whisper required for MLX ASR. "
                "Install with: pip install mlx-whisper"
            )

        language = language or self.language

        # Map model names to MLX Community repo IDs
        model_mapping = {
            "tiny": "mlx-community/whisper-tiny-mlx",
            "base": "mlx-community/whisper-base-mlx",
            "small": "mlx-community/whisper-small-mlx",
            "medium": "mlx-community/whisper-medium-mlx",
            "large-v3": "mlx-community/whisper-large-v3-mlx",
            "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
            "turbo": "mlx-community/whisper-large-v3-turbo",
            "large-v2": "mlx-community/whisper-large-v2-mlx",
        }

        model_name = model_mapping.get(self.asr_model, self.asr_model)

        logger.info(f"Loading MLX Whisper model: {model_name}")
        
        # Load model using correct mlx-whisper API
        # mlx-whisper supports initial_prompt directly, not via its decoder prompt token list.
        initial_prompt = kwargs.get("initial_prompt")
        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=model_name,
            language=language,
            verbose=False,
            initial_prompt=initial_prompt,
        )

        # Normalize to WhisperX format
        segments = []
        for seg in result.get("segments", []):
            segments.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
            })

        return {
            "segments": segments,
            "language": result.get("language", language),
        }

    def transcribe_ctranslate2(
        self,
        audio: str,
        language: Optional[str] = None,
        batch_size: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Transcribe using CTranslate2 (faster-whisper).

        Args:
            audio: Path to audio file
            language: Language code
            batch_size: Batch size
            **kwargs: Additional parameters

        Returns:
            Dictionary with transcription results
        """
        from whisperx.asr import load_model

        language = language or self.language
        batch_size = batch_size or self.batch_size

        logger.info("Loading CTranslate2 model...")
        model = load_model(
            self.asr_model,
            device=self.device,
            compute_type="float16" if self.device == "cuda" else "float32",
            language=language,
        )

        audio_data = load_audio(audio)

        logger.info("Running CTranslate2 transcription...")
        result = model.transcribe(
            audio_data,
            batch_size=batch_size,
        )

        return result

    def transcribe(
        self,
        audio: str,
        language: Optional[str] = None,
        batch_size: Optional[int] = None,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        enable_diarization: bool = True,
        enable_alignment: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run full transcription pipeline with memory safety.

        Args:
            audio: Path to audio file
            language: Language code
            batch_size: Batch size override
            num_speakers: Exact number of speakers
            min_speakers: Minimum number of speakers
            max_speakers: Maximum number of speakers
            enable_diarization: Enable speaker diarization
            enable_alignment: Enable word-level alignment

        Returns:
            Dictionary with transcription results
        """
        language = language or self.language
        batch_size = batch_size or self.batch_size

        # Memory safety check before starting
        check_memory_safety(threshold_gb=2.0)

        # Part 1: ASR (MLX on Apple Silicon, CTranslate2 on x86)
        if self.asr_backend == "mlx":
            logger.info("Using MLX ASR backend...")
            result = self.transcribe_mlx(audio, language=language, **kwargs)
        else:
            logger.info("Using CTranslate2 ASR backend...")
            result = self.transcribe_ctranslate2(
                audio, language=language, batch_size=batch_size, **kwargs
            )

        # CRITICAL: Cleanup ASR model before alignment
        if self.model is not None:
            del self.model
            self.model = None
        cleanup_unified_memory(force=True)

        # Part 2: Alignment (always PyTorch CPU)
        if enable_alignment:
            logger.info("Loading alignment model...")
            self.align_model, self.align_metadata = load_align_model(
                language_code=result["language"],
                device="cpu",  # Alignment always on CPU
            )

            logger.info("Running alignment...")
            result = align(
                result["segments"],
                self.align_model,
                self.align_metadata,
                audio,
                "cpu",
            )

            # CRITICAL: Unload alignment before diarization
            del self.align_model
            self.align_model = None
            cleanup_unified_memory(force=True)

        # Part 3: Diarization with backend selection
        if enable_diarization:
            logger.info(f"Loading diarization model (backend: {self.diarization_backend})...")

            # Use MLX Nemotron backend if selected
            if self.diarization_backend == "mlx_nemotron":
                try:
                    from whisperx.backends.diarization_backend import DiarizationBackendFactory
                    
                    diarize_backend = DiarizationBackendFactory.create(
                        backend_type="mlx_nemotron",
                        hf_token=self.hf_token,
                    )
                    
                    diarize_df = diarize_backend.diarize(
                        audio,
                        num_speakers=num_speakers,
                        min_speakers=min_speakers,
                        max_speakers=max_speakers,
                    )
                    
                    logger.info("Assigning speakers to words...")
                    result = assign_word_speakers(diarize_df, result, fill_nearest=True)
                    
                    diarize_backend.unload()
                    cleanup_unified_memory(force=True)
                    
                except Exception as e:
                    logger.error(f"MLX Nemotron diarization failed: {e}")
                    raise
            else:
                # Use Pyannote (default)
                target_device = self.diarization_device
                if self.mps_crashed and self.diarization_device == "mps":
                    target_device = "cpu"
                    logger.info("MPS crashed previously, using CPU for diarization")

                try:
                    self.diarize_model = DiarizationPipeline(
                        model_name="pyannote/speaker-diarization-community-1",
                        token=self.hf_token,
                        device=target_device,
                    )
                except Exception as e:
                    if self.diarization_device == "mps":
                        logger.warning(f"Failed to load diarization on MPS ({e}), falling back to CPU")
                        self.mps_crashed = True
                        target_device = "cpu"
                        self.diarize_model = DiarizationPipeline(
                            model_name="pyannote/speaker-diarization-community-1",
                            token=self.hf_token,
                            device=target_device,
                        )
                    else:
                        raise

                logger.info(f"Running diarization on {target_device}...")

                try:
                    diarize_segments = self.diarize_model(
                        audio,
                        num_speakers=num_speakers,
                        min_speakers=min_speakers,
                        max_speakers=max_speakers,
                    )
                except Exception as e:
                    if self.diarization_device == "mps" and not self.mps_crashed:
                        logger.warning(f"MPS crashed during diarization ({e}), retrying on CPU")
                        self.mps_crashed = True
                        del self.diarize_model
                        self.diarize_model = DiarizationPipeline(
                            model_name="pyannote/speaker-diarization-community-1",
                            token=self.hf_token,
                            device="cpu",
                        )
                        diarize_segments = self.diarize_model(
                            audio,
                            num_speakers=num_speakers,
                            min_speakers=min_speakers,
                            max_speakers=max_speakers,
                        )
                    else:
                        raise

                logger.info("Assigning speakers to words...")
                logger.info(f"Diarization segments type: {type(diarize_segments)}")
                if hasattr(diarize_segments, 'itertracks'):
                    tracks = list(diarize_segments.itertracks(yield_label=True))
                    logger.info(f"Number of diarization tracks: {len(tracks)}")
                result = assign_word_speakers(diarize_segments, result, fill_nearest=True)

                # CRITICAL: Unload diarization immediately
                del self.diarize_model
                self.diarize_model = None
                cleanup_unified_memory(force=True)

        return result

    def cleanup(self):
        """Clean up loaded models."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.align_model is not None:
            del self.align_model
            self.align_model = None
        if self.diarize_model is not None:
            del self.diarize_model
            self.diarize_model = None
        cleanup_unified_memory(force=True)
