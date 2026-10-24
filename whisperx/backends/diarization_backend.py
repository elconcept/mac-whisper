"""
Diarization backend with two engines for Apple Silicon.

Supports:
- PyannoteDiarizationBackend: High-precision hierarchical clustering (AHC)
- MlxNemotronDiarizationBackend: Ultra-fast native MLX GPU diarization

Both implement BaseDiarizationBackend interface for seamless switching.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple
import pandas as pd
import torch
from whisperx.audio import load_audio, SAMPLE_RATE
from whisperx.log_utils import get_logger
from whisperx.memory_guard import cleanup_unified_memory

logger = get_logger(__name__)


class BaseDiarizationBackend(ABC):
    """Abstract base class for diarization backends."""

    def __init__(self):
        self._model = None

    @abstractmethod
    def diarize(
        self,
        audio_path: str,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        num_speakers: Optional[int] = None,
        progress_callback=None
    ) -> pd.DataFrame:
        """
        Perform speaker diarization.

        Args:
            audio_path: Path to audio file
            min_speakers: Minimum number of speakers
            max_speakers: Maximum number of speakers
            num_speakers: Exact number of speakers (if known)
            progress_callback: Optional callback for progress updates

        Returns:
            DataFrame with columns: ['segment', 'label', 'speaker', 'start', 'end']
        """
        pass

    @abstractmethod
    def unload(self):
        """Unload model and free memory."""
        pass


class PyannoteDiarizationBackend(BaseDiarizationBackend):
    """
    PyTorch-based diarization using pyannote.audio.

    High-precision hierarchical clustering (AHC) with automatic
    MPS-to-CPU fallback for shader bugs on Apple Silicon.
    """

    def __init__(self, hf_token: str, device: str = "mps", use_fallback: bool = True):
        """
        Initialize Pyannote diarization backend.

        Args:
            hf_token: HuggingFace token for model access
            device: Device to run on ("mps", "cuda", "cpu")
            use_fallback: Enable automatic MPS-to-CPU fallback on errors
        """
        super().__init__()
        self.device_str = device
        self.hf_token = hf_token
        self.use_fallback = use_fallback
        self._mps_crashed = False
        self._pipeline = None

    def _load_model(self):
        """Load pyannote diarization model with fallback handling."""
        from pyannote.audio import Pipeline

        target_device = torch.device(self.device_str)

        # Use CPU if MPS crashed previously
        if self._mps_crashed and self.device_str == "mps":
            target_device = torch.device("cpu")
            logger.info("MPS crashed previously, using CPU for diarization")

        try:
            self._pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-community-1",
                token=self.hf_token
            ).to(target_device)
            logger.info(f"Pyannote diarization loaded on {target_device}")

        except Exception as e:
            if self.device_str == "mps" and self.use_fallback:
                logger.warning(f"Failed to load on MPS ({e}), falling back to CPU")
                self._mps_crashed = True
                self.device_str = "cpu"
                self._pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-community-1",
                    token=self.hf_token
                ).to(torch.device("cpu"))
            else:
                raise

    def diarize(
        self,
        audio_path: str,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        num_speakers: Optional[int] = None,
        progress_callback=None
    ) -> pd.DataFrame:
        """
        Perform diarization using pyannote.

        Args:
            audio_path: Path to audio file
            min_speakers: Minimum number of speakers
            max_speakers: Maximum number of speakers
            num_speakers: Exact number of speakers
            progress_callback: Optional progress callback

        Returns:
            DataFrame with diarization results
        """
        if self._pipeline is None:
            self._load_model()

        # Load audio
        audio = load_audio(audio_path)
        audio_data = {
            'waveform': torch.from_numpy(audio[None, :]),
            'sample_rate': SAMPLE_RATE
        }

        # Run diarization with fallback handling
        try:
            output = self._pipeline(
                audio_data,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                num_speakers=num_speakers,
            )
        except Exception as e:
            if self.device_str == "mps" and self.use_fallback and not self._mps_crashed:
                logger.warning(f"MPS crashed during diarization ({e}), retrying on CPU")
                self._mps_crashed = True
                self.device_str = "cpu"
                self._pipeline.to(torch.device("cpu"))
                output = self._pipeline(
                    audio_data,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                    num_speakers=num_speakers,
                )
            else:
                raise

        # Convert to DataFrame
        diarization = output.speaker_diarization
        df = pd.DataFrame(
            diarization.itertracks(yield_label=True),
            columns=['segment', 'label', 'speaker']
        )
        df['start'] = df['segment'].apply(lambda x: x.start)
        df['end'] = df['segment'].apply(lambda x: x.end)

        return df

    def unload(self):
        """Unload model and free memory."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
        cleanup_unified_memory(force=True)


class MlxNemotronDiarizationBackend(BaseDiarizationBackend):
    """
    MLX-based diarization using mlx-audio Sortformer.

    Ultra-fast native Apple Silicon GPU diarization that bypasses
    PyTorch overhead. Uses mlx-audio library with Sortformer model.

    Reference: https://huggingface.co/mlx-community/diar_sortformer_4spk-v1-fp16
    """

    def __init__(self, model_name: str = "mlx-community/diar_sortformer_4spk-v1-fp16"):
        """
        Initialize MLX Sortformer diarization backend.

        Args:
            model_name: Model identifier (defaults to Sortformer v1)
        """
        super().__init__()
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        """Load MLX Sortformer diarization model."""
        try:
            from mlx_audio.vad import load
            self._model = load(self.model_name, strict=True)
            logger.info(f"MLX Sortformer diarization loaded: {self.model_name}")
        except ImportError:
            raise ImportError(
                "mlx-audio not available for MLX diarization. "
                "Install with: pip install mlx-audio"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load MLX Sortformer model: {e}")

    def diarize(
        self,
        audio_path: str,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        num_speakers: Optional[int] = None,
        progress_callback=None
    ) -> pd.DataFrame:
        """
        Perform diarization using MLX Sortformer.

        Args:
            audio_path: Path to audio file
            min_speakers: Minimum number of speakers
            max_speakers: Maximum number of speakers
            num_speakers: Exact number of speakers
            progress_callback: Optional progress callback

        Returns:
            DataFrame with diarization results
        """
        if self._model is None:
            self._load_model()

        # Run MLX Sortformer diarization
        logger.info(f"Running MLX Sortformer diarization on {audio_path}...")
        
        result = self._model.generate(audio_path, threshold=0.5, verbose=False)
        
        # Convert MLX result to DataFrame format
        records = []
        for segment in result.segments:
            records.append({
                'segment': None,
                'label': segment.speaker,
                'speaker': segment.speaker,
                'start': segment.start,
                'end': segment.end
            })
        
        df = pd.DataFrame(records)
        logger.info(f"MLX Sortformer diarization completed: {len(df)} segments")
        
        return df

    def unload(self):
        """Unload model and free memory."""
        if self._model is not None:
            del self._model
            self._model = None
        cleanup_unified_memory(force=True)


class DiarizationBackendFactory:
    """Factory for creating appropriate diarization backend."""

    @staticmethod
    def create(
        backend_type: str,
        hf_token: Optional[str] = None,
        device: str = "mps",
        **kwargs
    ) -> BaseDiarizationBackend:
        """
        Create diarization backend based on type.

        Args:
            backend_type: "pyannote" or "mlx_nemotron"
            hf_token: HuggingFace token (required for pyannote)
            device: Device for pyannote backend
            **kwargs: Additional backend-specific parameters

        Returns:
            Configured diarization backend instance
        """
        if backend_type == "pyannote":
            if not hf_token:
                raise ValueError("HF_TOKEN required for pyannote backend")
            return PyannoteDiarizationBackend(
                hf_token=hf_token,
                device=device,
                use_fallback=kwargs.get("use_fallback", True)
            )
        elif backend_type == "mlx_nemotron":
            return MlxNemotronDiarizationBackend(
                model_name=kwargs.get("model_name", "mlx-community/diar_sortformer_4spk-v1-fp16")
            )
        else:
            raise ValueError(f"Unknown backend type: {backend_type}")
