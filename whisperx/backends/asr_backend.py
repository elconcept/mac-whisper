"""
ASR (Automatic Speech Recognition) backend abstraction.

Supports multiple inference engines:
- MLX backend: mlx-whisper for Apple Silicon GPU acceleration
- CTranslate2 backend: faster-whisper for x86/GPU platforms
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import platform


class ASRBackend(ABC):
    """Abstract base class for ASR backends."""

    def __init__(self, model_name: str, device: str, **kwargs):
        """
        Initialize ASR backend.

        Args:
            model_name: Model identifier (e.g., "large-v3-turbo", "mlx-community/whisper-large-v3-turbo")
            device: Device to run on ("mps", "cuda", "cpu")
            **kwargs: Additional backend-specific parameters
        """
        self.model_name = model_name
        self.device = device
        self.kwargs = kwargs
        self._model = None

    @abstractmethod
    def load_model(self) -> None:
        """Load the ASR model."""
        pass

    @abstractmethod
    def transcribe(
        self,
        audio: str,
        language: str = "pl",
        beam_size: int = 5,
        initial_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Transcribe audio file.

        Args:
            audio: Path to audio file or audio array
            language: Language code (default: "pl" for Polish)
            beam_size: Beam size for decoding
            initial_prompt: Optional initial prompt for context
            **kwargs: Additional transcription parameters

        Returns:
            Dictionary with "segments" key containing transcription segments
        """
        pass

    @abstractmethod
    def unload(self) -> None:
        """Unload model and free memory."""
        pass

    @classmethod
    def get_default_backend(cls) -> "ASRBackend":
        """
        Get the default backend for the current platform.

        Returns:
            Appropriate backend class instance
        """
        if platform.system() == "Darwin":
            # Apple Silicon - use MLX
            return MLXASRBackend
        else:
            # x86/Linux/Windows - use CTranslate2
            return CTranslate2ASRBackend


class MLXASRBackend(ASRBackend):
    """MLX-based ASR backend using mlx-whisper for Apple Silicon."""

    def __init__(self, model_name: str, device: str = "mps", **kwargs):
        super().__init__(model_name, device, **kwargs)
        # Map short names to MLX Community repo IDs
        self._model_mapping = {
            "tiny": "mlx-community/whisper-tiny-mlx",
            "base": "mlx-community/whisper-base-mlx",
            "small": "mlx-community/whisper-small-mlx",
            "medium": "mlx-community/whisper-medium-mlx",
            "large-v3": "mlx-community/whisper-large-v3-mlx",
            "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
            "turbo": "mlx-community/whisper-large-v3-turbo",
        }

    def _resolve_model_name(self) -> str:
        """Resolve short model name to full HF repo ID."""
        if self.model_name in self._model_mapping:
            return self._model_mapping[self.model_name]
        return self.model_name

    def load_model(self) -> None:
        """Load MLX Whisper model."""
        try:
            import mlx_whisper
        except ImportError:
            raise ImportError(
                "mlx-whisper is required for MLXASRBackend. "
                "Install with: pip install mlx-whisper"
            )

        model_name = self._resolve_model_name()
        self._model = mlx_whisper.load_model(model_name, device=self.device)

    def transcribe(
        self,
        audio: str,
        language: str = "pl",
        beam_size: int = 5,
        initial_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Transcribe audio using MLX Whisper.

        Note: mlx-whisper doesn't support all WhisperX features like
        batched inference or interleaved context. This is a limitation
        of the mlx-whisper library.
        """
        if self._model is None:
            self.load_model()

        import mlx_whisper

        # Prepare transcription parameters
        transcribe_params = {
            "path_or_file": audio,
            "language": language,
            "beam_size": beam_size,
            "verbose": kwargs.get("verbose", False),
        }

        if initial_prompt:
            transcribe_params["prompt"] = initial_prompt

        # Transcribe
        result = mlx_whisper.transcribe(**transcribe_params)

        # Normalize result to WhisperX format
        return {
            "segments": result.get("segments", []),
            "language": result.get("language", language),
        }

    def unload(self) -> None:
        """Unload MLX model (MLX doesn't have explicit unload, so just clear reference)."""
        self._model = None


class CTranslate2ASRBackend(ASRBackend):
    """CTranslate2-based ASR backend using faster-whisper for x86/GPU."""

    def __init__(self, model_name: str, device: str = "cuda", **kwargs):
        super().__init__(model_name, device, **kwargs)
        self._compute_type = kwargs.get("compute_type", "default")

    def load_model(self) -> None:
        """Load faster-whisper model."""
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError(
                "faster-whisper is required for CTranslate2ASRBackend. "
                "Install with: pip install faster-whisper"
            )

        self._model = WhisperModel(
            self.model_name,
            device=self.device,
            compute_type=self._compute_type,
        )

    def transcribe(
        self,
        audio: str,
        language: str = "pl",
        beam_size: int = 5,
        initial_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Transcribe audio using faster-whisper.

        Returns WhisperX-compatible format.
        """
        if self._model is None:
            self.load_model()

        segments, info = self._model.transcribe(
            audio,
            language=language,
            beam_size=beam_size,
            initial_prompt=initial_prompt,
            word_timestamps=True,
        )

        # Convert to WhisperX format
        result_segments = []
        for segment in segments:
            result_segments.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "words": [
                    {
                        "start": word.start,
                        "end": word.end,
                        "word": word.word,
                    }
                    for word in segment.words
                ],
            })

        return {
            "segments": result_segments,
            "language": info.language,
        }

    def unload(self) -> None:
        """Unload faster-whisper model."""
        if self._model is not None:
            del self._model
            self._model = None
