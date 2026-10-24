"""
VAD (Voice Activity Detection) backend abstraction.

Supports multiple VAD implementations:
- MLX Silero: Silero VAD v6 converted to MLX for Apple Silicon
- PyTorch Pyannote: pyannote.audio VAD for x86/GPU
- PyTorch Silero: Original Silero VAD from PyTorch Hub
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import platform


class VADBackend(ABC):
    """Abstract base class for VAD backends."""

    def __init__(self, device: str = "cpu", **kwargs):
        """
        Initialize VAD backend.

        Args:
            device: Device to run on ("mps", "cuda", "cpu")
            **kwargs: Additional backend-specific parameters
        """
        self.device = device
        self.kwargs = kwargs
        self._model = None

    @abstractmethod
    def load_model(self) -> None:
        """Load the VAD model."""
        pass

    @abstractmethod
    def preprocess_audio(self, audio):
        """
        Preprocess audio for VAD.

        Args:
            audio: Audio array or path

        Returns:
            Preprocessed audio in format expected by VAD model
        """
        pass

    @abstractmethod
    def detect_speech(self, audio) -> List[Dict[str, float]]:
        """
        Detect speech segments in audio.

        Args:
            audio: Preprocessed audio

        Returns:
            List of speech segments with "start" and "end" times
        """
        pass

    @abstractmethod
    def merge_chunks(
        self,
        segments: List[Dict[str, float]],
        chunk_size: float = 30.0,
        onset: float = 0.5,
        offset: float = 0.3
    ) -> List[Dict[str, float]]:
        """
        Merge nearby speech segments into chunks.

        Args:
            segments: Raw speech segments
            chunk_size: Maximum chunk duration in seconds
            onset: Minimum silence duration to split speech
            offset: Minimum speech duration to start segment

        Returns:
            Merged chunks
        """
        pass

    @classmethod
    def get_default_backend(cls) -> "VADBackend":
        """
        Get the default backend for the current platform.

        Returns:
            Appropriate backend class instance
        """
        if platform.system() == "Darwin":
            # Apple Silicon - use MLX Silero
            return SileroMLXBackend
        else:
            # x86/Linux/Windows - use Pyannote
            return PyannoteVADBackend


class SileroMLXBackend(VADBackend):
    """MLX-based VAD backend using Silero VAD v6 for Apple Silicon."""

    def __init__(self, device: str = "mps", **kwargs):
        super().__init__(device, **kwargs)
        self._model_repo = kwargs.get(
            "model_repo",
            "mlx-community/silero-vad-v6"
        )
        self._sample_rate = 16000
        self._chunk_samples = 512  # 32ms at 16kHz
        self._context_samples = 64
        self._stft_hop = 128
        self._stft_pad_right = 64
        self._lstm_hidden = 128

        self._weights = None
        self._h = None
        self._c = None

    def load_model(self) -> None:
        """Load MLX Silero VAD model from HuggingFace."""
        try:
            import mlx.core as mx
            from huggingface_hub import hf_hub_download
            from safetensors.numpy import load_file
        except ImportError as e:
            raise ImportError(
                f"MLX dependencies required for SileroMLXBackend: {e}. "
                "Install with: pip install mlx safetensors huggingface_hub"
            )

        print(f"Loading Silero VAD MLX model from {self._model_repo}...")

        # Download weights from HuggingFace
        weights_path = hf_hub_download(
            repo_id=self._model_repo,
            filename="model.safetensors"
        )

        # Load weights
        raw = load_file(weights_path)

        # Required keys for Silero VAD v6
        keys = [
            "vad_16k.stft_conv.weight",
            "vad_16k.conv1.weight", "vad_16k.conv1.bias",
            "vad_16k.conv2.weight", "vad_16k.conv2.bias",
            "vad_16k.conv3.weight", "vad_16k.conv3.bias",
            "vad_16k.conv4.weight", "vad_16k.conv4.bias",
            "vad_16k.lstm.Wx", "vad_16k.lstm.Wh", "vad_16k.lstm.bias",
            "vad_16k.final_conv.weight", "vad_16k.final_conv.bias",
        ]

        # Convert to MLX arrays
        import mlx.core as mx
        self._weights = {k: mx.array(raw[k]) for k in keys}

        # Initialize LSTM state
        self._h = mx.zeros((1, self._lstm_hidden))
        self._c = mx.zeros((1, self._lstm_hidden))

        print("Silero VAD MLX model loaded successfully")

    def preprocess_audio(self, audio):
        """Preprocess audio for Silero VAD (ensure 16kHz mono)."""
        import numpy as np
        from whisperx.audio import load_audio, SAMPLE_RATE

        if isinstance(audio, str):
            audio = load_audio(audio)

        # WhisperX loads at 16kHz, but ensure mono
        if len(audio.shape) > 1:
            audio = audio.mean(axis=0)

        # Ensure float32
        audio = audio.astype(np.float32)

        return audio

    def _reflect_pad_right_576(self, samples_576):
        """PyTorch ReflectionPad1d(right=64): out[L+i] = in[L-2-i] for i in 0..63."""
        import numpy as np
        L = samples_576.shape[-1]
        pad = np.empty(self._stft_pad_right, dtype=samples_576.dtype)
        for i in range(self._stft_pad_right):
            pad[i] = samples_576[L - 2 - i]
        return np.concatenate([samples_576, pad], axis=-1)

    def _predict_chunk(self, audio_640, h, c):
        """Run VAD prediction on a single chunk."""
        import mlx.core as mx

        w = self._weights

        # STFT
        z = mx.conv1d(audio_640, w["vad_16k.stft_conv.weight"], stride=self._stft_hop, padding=0)
        real, imag = z[:, :, :129], z[:, :, 129:]
        x = mx.sqrt(real * real + imag * imag + 1e-12)

        # Encoder (4 Conv1d layers)
        cfgs = [(1, 1), (2, 1), (2, 1), (1, 1)]
        for i, (stride, padding) in enumerate(cfgs, start=1):
            x = mx.conv1d(x, w[f"vad_16k.conv{i}.weight"], stride=stride, padding=padding)
            x = x + w[f"vad_16k.conv{i}.bias"]
            x = mx.maximum(x, 0)

        feat = x[:, 0, :]

        # LSTM
        gates = feat @ w["vad_16k.lstm.Wx"].T + h @ w["vad_16k.lstm.Wh"].T + w["vad_16k.lstm.bias"]
        i_g, f_g, g_g, o_g = mx.split(gates, 4, axis=-1)
        c_new = mx.sigmoid(f_g) * c + mx.sigmoid(i_g) * mx.tanh(g_g)
        h_new = mx.sigmoid(o_g) * mx.tanh(c_new)

        # Decoder
        dec = mx.conv1d(
            mx.maximum(h_new, 0)[:, None, :],
            w["vad_16k.final_conv.weight"], stride=1, padding=0,
        )
        dec = dec + w["vad_16k.final_conv.bias"]

        return mx.sigmoid(dec), h_new, c_new

    def detect_speech(self, audio, threshold: float = 0.5) -> List[Dict[str, float]]:
        """
        Detect speech segments using MLX Silero VAD.

        Args:
            audio: Preprocessed audio array (16kHz mono)
            threshold: Speech probability threshold (default: 0.5)

        Returns:
            List of speech segments with "start" and "end" times
        """
        if self._weights is None:
            self.load_model()

        import mlx.core as mx
        import numpy as np

        # Reset LSTM state
        self._h = mx.zeros((1, self._lstm_hidden))
        self._c = mx.zeros((1, self._lstm_hidden))

        # Process audio in chunks
        n_chunks = audio.shape[0] // self._chunk_samples
        context = np.zeros(self._context_samples, dtype=np.float32)

        speech_chunks = []
        in_speech = False
        speech_start = None

        for ci in range(n_chunks):
            chunk = audio[ci * self._chunk_samples : (ci + 1) * self._chunk_samples]
            merged = np.concatenate([context, chunk])
            padded = self._reflect_pad_right_576(merged)
            audio_input = mx.array(padded[None, :, None])

            prob, self._h, self._c = self._predict_chunk(audio_input, self._h, self._c)
            mx.eval(prob, self._h, self._c)

            p = float(prob[0, 0, 0].item())

            # Track speech segments
            if p >= threshold:
                if not in_speech:
                    in_speech = True
                    speech_start = ci * self._chunk_samples / self._sample_rate
            else:
                if in_speech:
                    in_speech = False
                    speech_end = ci * self._chunk_samples / self._sample_rate
                    speech_chunks.append({
                        "start": speech_start,
                        "end": speech_end,
                    })

            # Update context for next chunk
            context = chunk[-self._context_samples:].copy()

        # Close any open speech segment
        if in_speech:
            speech_end = n_chunks * self._chunk_samples / self._sample_rate
            speech_chunks.append({
                "start": speech_start,
                "end": speech_end,
            })

        return speech_chunks

    def merge_chunks(
        self,
        segments: List[Dict[str, float]],
        chunk_size: float = 30.0,
        onset: float = 0.5,
        offset: float = 0.3
    ) -> List[Dict[str, float]]:
        """Merge nearby speech segments."""
        if not segments:
            return []

        # Sort by start time
        segments = sorted(segments, key=lambda x: x["start"])

        merged = []
        current = segments[0].copy()

        for seg in segments[1:]:
            # If gap is smaller than onset, merge
            if seg["start"] - current["end"] < onset:
                current["end"] = seg["end"]
            else:
                merged.append(current)
                current = seg.copy()

        merged.append(current)

        # Split chunks that are too large
        final_chunks = []
        for chunk in merged:
            duration = chunk["end"] - chunk["start"]
            if duration > chunk_size:
                # Split into smaller chunks
                num_splits = int(duration / chunk_size) + 1
                split_duration = duration / num_splits
                for i in range(num_splits):
                    final_chunks.append({
                        "start": chunk["start"] + i * split_duration,
                        "end": chunk["start"] + (i + 1) * split_duration,
                    })
            else:
                final_chunks.append(chunk)

        return final_chunks


class PyannoteVADBackend(VADBackend):
    """PyTorch-based VAD backend using pyannote.audio."""

    def __init__(self, device: str = "cpu", **kwargs):
        super().__init__(device, **kwargs)
        self._model_name = kwargs.get(
            "model_name",
            "pyannote/segmentation-3.0"
        )
        self._use_fallback = kwargs.get("use_fallback", True)
        self._crashed = False

    def load_model(self) -> None:
        """Load pyannote VAD model with fallback handling."""
        try:
            from pyannote.audio import Pipeline
            import torch
        except ImportError:
            raise ImportError(
                "pyannote.audio is required for PyannoteVADBackend. "
                "Install with: pip install pyannote.audio"
            )

        try:
            self._model = Pipeline.from_pretrained(
                self._model_name,
                use_auth_token=self.kwargs.get("hf_token")
            )

            target_device = torch.device(self.device)
            if self._crashed and self.device == "mps":
                target_device = torch.device("cpu")
                print("Warning: MPS crashed previously, using CPU for VAD")

            self._model.to(target_device)
            print(f"Pyannote VAD loaded on {target_device}")

        except Exception as e:
            if self.device == "mps" and self._use_fallback:
                print(f"Warning: Failed to load Pyannote VAD on MPS ({e}), falling back to CPU")
                self._crashed = True
                self.device = "cpu"
                self._model = Pipeline.from_pretrained(
                    self._model_name,
                    use_auth_token=self.kwargs.get("hf_token")
                )
                self._model.to(torch.device("cpu"))
            else:
                raise

    def preprocess_audio(self, audio):
        """Preprocess audio for pyannote VAD."""
        import torch
        from whisperx.audio import load_audio, SAMPLE_RATE

        if isinstance(audio, str):
            audio = load_audio(audio)

        return {
            "waveform": torch.from_numpy(audio[None, :]),
            "sample_rate": SAMPLE_RATE
        }

    def detect_speech(self, audio) -> List[Dict[str, float]]:
        """Detect speech segments using pyannote VAD with fallback."""
        if self._model is None:
            self.load_model()

        try:
            output = self._model(audio)
        except Exception as e:
            if self.device == "mps" and self._use_fallback:
                print(f"Warning: Pyannote VAD crashed on MPS ({e}), retrying on CPU")
                self._crashed = True
                self.device = "cpu"
                self._model.to(torch.device("cpu"))
                output = self._model(audio)
            else:
                raise

        # Convert to segment list
        segments = []
        for turn, _ in output.itertracks(yield_label=True):
            segments.append({
                "start": turn.start,
                "end": turn.end,
            })

        return segments

    def merge_chunks(
        self,
        segments: List[Dict[str, float]],
        chunk_size: float = 30.0,
        onset: float = 0.5,
        offset: float = 0.3
    ) -> List[Dict[str, float]]:
        """Merge nearby speech segments (same logic as Silero)."""
        return SileroMLXBackend.merge_chunks(
            self, segments, chunk_size, onset, offset
        )
