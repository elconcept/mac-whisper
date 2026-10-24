#!/usr/bin/env python3
"""Real, model-driven E2E gate for signed/tagged macOS releases.

Synthesizes Polish speech on the runner with espeak-ng; no private recordings
or external test corpora. Uses real MLX ASR, Polish Wav2Vec2 alignment, and
pyannote Community-1 diarization.
"""
import json
import os
import platform
import subprocess
import tempfile
from pathlib import Path


def main():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("Real E2E must execute on macOS Apple Silicon (arm64)")
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN secret is required for real Pyannote E2E; release BLOCKED.")

    with tempfile.TemporaryDirectory(prefix="mac-whisper-e2e-") as temp:
        audio = Path(temp) / "polish-synthetic.wav"
        text = ("Dzień dobry. To jest automatyczny test rozpoznawania polskiej mowy. "
                "Dzisiaj sprawdzamy nagranie, znaczniki czasu oraz rozpoznawanie rozmówcy. ") * 9
        subprocess.run(["espeak-ng", "-v", "pl", "-s", "138", "-w", str(audio), text],
                       check=True, timeout=90)
        assert audio.exists() and audio.stat().st_size > 50_000

        from whisperx.pipeline import TranscriptionPipeline
        engine = TranscriptionPipeline(
            asr_model="small",
            hf_token=token,
            language="pl",
            use_mps_for_diarization=False,
            diarization_backend="pyannote",
        )
        try:
            result = engine.transcribe(
                str(audio),
                language="pl",
                num_speakers=1,
                enable_alignment=True,
                enable_diarization=True,
                initial_prompt="Automatyczny test rozpoznawania polskiej mowy.",
            )
        finally:
            engine.cleanup()

        segments = result.get("segments", [])
        assert segments, "No speech detected by the real MLX engine"
        assert any(s.get("text", "").strip() for s in segments), "No transcribed text"
        assert any(s.get("words") for s in segments), "Missing Polish word-level alignment"
        assert any(s.get("speaker") for s in segments), "Missing real Pyannote speaker labels"
        assert all(0 <= s["start"] <= s["end"] for s in segments), "Invalid segment timestamps"
        assert result.get("language") == "pl", "Non-Polish output language"
        out = Path(temp) / "e2e.json"
        out.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        print("REAL E2E PASSED: Polish MLX ASR + Wav2Vec2 alignment + Pyannote diarization")
        print("Segments:", len(segments), "Words:", sum(len(s.get("words", [])) for s in segments))


if __name__ == "__main__":
    main()
