"""
Transcription pipeline with MLX ASR for Apple Silicon.

Uses MLX (mlx-whisper) for ASR on Apple Silicon GPU for maximum speed,
with automatic MPS fallback for diarization.
"""

import os
import platform
import torch
from typing import Optional, Dict, Any
from pathlib import Path

from whisperx.audio import load_audio, SAMPLE_RATE
from whisperx.alignment import load_align_model, align
from whisperx.diarize import assign_word_speakers, DiarizationPipeline
from whisperx.log_utils import get_logger
from whisperx.utils import exact_div
from whisperx.memory_guard import cleanup_unified_memory, check_memory_safety, release_mlx_model, memory_snapshot

logger = get_logger(__name__)


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
        asr_backend: str = "mlx-whisper",
        vad_backend: str = "mlx-silero",
        gcp_project: str | None = None,
        gcp_location: str = "eu",
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
        self.vad_backend = vad_backend
        self.selected_asr_backend = asr_backend
        self.audio_model = None
        self.gcp_project = gcp_project
        self.gcp_location = gcp_location
        self.gcp_credentials = None
        self.gcp_client = None

        # Detect platform and auto-select ASR backend
        is_apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"

        if asr_backend == "gcp":
            # Cloud recognition is platform-neutral, alignment and diarization local.
            self.asr_backend = "gcp"
            self.device = "cloud"
            logger.info("Google Cloud Speech V2 selected (audio uploaded only when user runs transcription)")
        elif is_apple_silicon:
            # Apple Silicon: Use MLX (mlx-whisper) on MPS
            self.asr_backend = "mlx" if asr_backend == "mlx-whisper" else asr_backend
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
        self.last_memory = {}

        logger.info(f"Pipeline initialized:")
        logger.info(f"  ASR: {self.asr_backend} on {self.device}")
        logger.info(f"  Diarization: {self.diarization_backend} on {self.diarization_device}")

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
        decode_kwargs = {
            name: kwargs[name]
            for name in ("clip_timestamps", "condition_on_previous_text",
                         "without_timestamps", "beam_size", "temperature")
            if kwargs.get(name) is not None
        }
        try:
            result = mlx_whisper.transcribe(
                audio, path_or_hf_repo=model_name, language=language,
                verbose=False, initial_prompt=initial_prompt,
                **decode_kwargs,
            )
        except Exception as exc:
            # HF rejects public downloads when a globally configured token has
            # expired. Retry only hub-authentication errors anonymously, never
            # suppress inference/decoder errors or silently change the model.
            reason = str(exc)
            if not any(fragment in reason for fragment in (
                "User Access Token", "401 Client Error", "Invalid credentials",
            )):
                raise
            from mac_whisper_engines import public_repo_path
            public_weights = public_repo_path(model_name)
            result = mlx_whisper.transcribe(
                audio, path_or_hf_repo=public_weights, language=language,
                verbose=False, initial_prompt=initial_prompt,
                **decode_kwargs,
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
        use_vad: bool = False,
        on_segment=None,
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
        check_memory_safety(threshold_gb=1.2)
        self.last_memory["before_asr"] = memory_snapshot()

        # The default core is original WhisperX VAD/merge -> MLX Whisper decoder.
        # Crucially we DO NOT use homemade speech spans or MLX-generated
        # segment boundaries for forced alignment. Keep alternative backends
        # separate so they cannot modify this core pipeline.
        if (self.asr_backend == "mlx" and use_vad
                and self.vad_backend in ("whisperx-pyannote", "torch-silero")):
            from whisperx.mlx_adapter import transcribe_whisperx_mlx
            # Mirror upstream WhisperX's timestamp-free, independent
            # chunk decoding. mlx-whisper does NOT implement beam search:
            # use deterministic temperature-0 greedy decoding instead.
            # Precise word times come from Polish Wav2Vec2 below.
            whisperx_decode = dict(kwargs)
            whisperx_decode.update(
                without_timestamps=True,
                condition_on_previous_text=False,
                temperature=0.0,
            )
            result = transcribe_whisperx_mlx(
                audio,
                lambda clip: self.transcribe_mlx(clip, language=language,
                                                **whisperx_decode),
                vad_backend=self.vad_backend,
                vad_device=self.diarization_device,
                on_segment=on_segment,
            )
            self.last_memory["vad_source"] = "upstream-whisperx"
        elif use_vad or on_segment is not None or self.asr_backend in ("mlx-audio", "gcp"):

            import numpy as np
            from scipy.io import wavfile
            from tempfile import TemporaryDirectory
            from mac_whisper_engines import (
                get_vad_spans, merge_spans, full_chunks,
                mlx_audio_transcribe, model_id,
            )
            waveform = load_audio(audio)
            duration = len(waveform) / SAMPLE_RATE
            if use_vad:
                spans = get_vad_spans(audio, self.vad_backend)
                clips = merge_spans(spans, duration)
                self.last_memory["vad_intervals"] = len(spans)
            else:
                clips = full_chunks(duration)
            segments = []
            if clips:
                with TemporaryDirectory(prefix="macwhisper-clips-") as temp:
                    for index, (start, end) in enumerate(clips):
                        filename = os.path.join(temp, f"clip_{index:05d}.wav")
                        a, b = round(start * SAMPLE_RATE), round(end * SAMPLE_RATE)
                        pcm = np.clip(waveform[a:b] * 32767, -32768, 32767).astype(np.int16)
                        if not len(pcm):
                            continue
                        wavfile.write(filename, SAMPLE_RATE, pcm)
                        if self.asr_backend == "gcp":
                            from mac_whisper_gcp import application_default_credentials, gcp_transcribe_clip
                            if self.gcp_credentials is None:
                                self.gcp_credentials, self.gcp_project = application_default_credentials(self.gcp_project)
                            self.gcp_client, excerpt = gcp_transcribe_clip(
                                filename, model=self.asr_model, project_id=self.gcp_project,
                                location=self.gcp_location, client=self.gcp_client,
                                credentials=self.gcp_credentials, duration=end-start,
                                hints=kwargs.get("initial_prompt"))
                        elif self.asr_backend == "mlx-audio":
                            self.audio_model, excerpt = mlx_audio_transcribe(
                                filename, model_id("mlx-audio", self.asr_model),
                                model=self.audio_model,
                                initial_prompt=kwargs.get("initial_prompt"), duration=end-start)
                        elif self.asr_backend == "mlx":
                            excerpt = self.transcribe_mlx(filename, language=language, **kwargs)
                        else:
                            excerpt = self.transcribe_ctranslate2(
                                filename, language=language, batch_size=batch_size, **kwargs)
                        for row in excerpt.get("segments", []):
                            row["start"] = float(row["start"]) + start
                            row["end"] = float(row["end"]) + start
                            for word in row.get("words", []):
                                if word.get("start") is not None:
                                    word["start"] = float(word["start"]) + start
                                if word.get("end") is not None:
                                    word["end"] = float(word["end"]) + start
                            segments.append(row)
                            if on_segment is not None:
                                on_segment(dict(row))
            del waveform
            result = {"language": "pl", "segments": segments}
        elif self.asr_backend == "mlx":
            result = self.transcribe_mlx(audio, language=language, **kwargs)
        else:
            result = self.transcribe_ctranslate2(
                audio, language=language, batch_size=batch_size, **kwargs
            )

        # CRITICAL: Cleanup ASR model before alignment
        if self.model is not None:
            del self.model
            self.model = None
        release_mlx_model()
        self.audio_model = None
        cleanup_unified_memory(force=True)
        self.last_memory["after_asr"] = memory_snapshot()

        if not result.get("segments"):
            return result

        # Original WhisperX forced alignment, using Polish Wav2Vec2 CTC.
        # Prefer MPS on Apple Silicon, with CPU fallback for unsupported
        # operators. NEVER manufacture word times from Whisper chunks.
        if enable_alignment:
            align_device = "mps" if self.diarization_device == "mps" else "cpu"
            try:
                self.align_model, self.align_metadata = load_align_model(
                    language_code=result["language"], device=align_device
                )
                result = align(
                    result["segments"], self.align_model, self.align_metadata,
                    audio, align_device,
                )
            except (RuntimeError, NotImplementedError, TypeError) as exc:
                if align_device != "mps":
                    raise
                logger.warning("WhisperX Wav2Vec2 MPS unsupported (%s); retrying CPU", exc)
                if self.align_model is not None:
                    del self.align_model
                    self.align_model = None
                cleanup_unified_memory(force=True)
                self.align_model, self.align_metadata = load_align_model(
                    language_code=result["language"], device="cpu"
                )
                result = align(
                    result["segments"], self.align_model, self.align_metadata,
                    audio, "cpu",
                )

            # CRITICAL: Unload alignment before diarization
            del self.align_model
            self.align_model = None
            cleanup_unified_memory(force=True)
            self.last_memory["after_alignment"] = memory_snapshot()

        # Part 3: Diarization with backend selection
        if enable_diarization:
            logger.info(f"Loading diarization model (backend: {self.diarization_backend})...")

            if self.diarization_backend != "pyannote":
                from mac_whisper_engines import mlx_audio_diarization, overlap_intervals
                if any(n is not None for n in (num_speakers, min_speakers, max_speakers)):
                    logger.warning("MLX diarization ignores requested speaker count; native models detect automatically")
                diarize_df = mlx_audio_diarization(audio, self.diarization_backend)
                overlaps = overlap_intervals(diarize_df)
                result["diarization_intervals"] = diarize_df.to_dict("records")
                result["overlap_intervals"] = overlaps
                if not diarize_df.empty:
                    result = assign_word_speakers(diarize_df, result, fill_nearest=False)
                    for seg in result.get("segments", []):
                        if any(r["start"] < seg["end"] and seg["start"] < r["end"] for r in overlaps):
                            seg["speaker_ambiguous"] = True
                        for word in seg.get("words", []):
                            if word.get("start") is not None and word.get("end") is not None:
                                if any(r["start"] < word["end"] and word["start"] < r["end"] for r in overlaps):
                                    word["speaker_ambiguous"] = True
                cleanup_unified_memory(force=True)
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
                result = assign_word_speakers(diarize_segments, result, fill_nearest=False)

                # CRITICAL: Unload diarization immediately
                del self.diarize_model
                self.diarize_model = None
                cleanup_unified_memory(force=True)

        self.last_memory["after_diarization"] = memory_snapshot()
        return result

    def cleanup(self):
        """Clean up loaded models and mlx-whisper cache."""
        release_mlx_model()
        self.audio_model = None
        self.gcp_client = None
        self.gcp_credentials = None
        if self.model is not None:
            del self.model
            self.model = None
        if self.align_model is not None:
            del self.align_model
            self.align_model = None
            self.align_metadata = None
        if self.diarize_model is not None:
            del self.diarize_model
            self.diarize_model = None
        cleanup_unified_memory(force=True)
