#!/usr/bin/env python3
"""Real Polish model-driven E2E quality smoke gate for tagged releases."""
from pathlib import Path
import json
import os
import platform

from quality_metrics import wer, cer


def main():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("Real model E2E requires macOS Apple Silicon")
    from dotenv import dotenv_values
    token = os.getenv("HF_TOKEN") or dotenv_values().get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN unavailable in Actions or .env. Release blocked.")
    audio = Path("input/sample_pl_30s.ogg")
    ground_truth = Path("input/sample_pl_30s.reference.txt")
    if not audio.is_file() or not ground_truth.is_file():
        raise FileNotFoundError("Run python scripts/generate_polish_sample.py")
    from whisperx.pipeline import TranscriptionPipeline
    from whisperx.memory_guard import memory_snapshot

    before = memory_snapshot()
    engine = TranscriptionPipeline(
        asr_model="small", hf_token=token, language="pl",
        use_mps_for_diarization=False, diarization_backend="pyannote",
    )
    try:
        result = engine.transcribe(
            str(audio), language="pl", num_speakers=1,
            enable_alignment=True, enable_diarization=True,
            initial_prompt="Polski test mowy; Kraków, Warszawa, Szczecin",
            use_vad=False,
        )
        peaks = dict(engine.last_memory)
    finally:
        engine.cleanup()

    segments = result.get("segments", [])
    assert segments, "No segments from actual MLX Whisper"
    assert any(s.get("words") for s in segments), "Real alignment missing word timestamps"
    assert any(s.get("speaker") for s in segments), "Real Pyannote missing speaker IDs"
    assert result.get("language") == "pl"
    assert all(0 <= s["start"] <= s["end"] for s in segments)
    transcription = " ".join(s.get("text", "") for s in segments)
    truth = ground_truth.read_text(encoding="utf-8")
    score = {"wer": wer(truth, transcription), "cer": cer(truth, transcription),
             "segments": len(segments), "memory": {"before": before, "stages": peaks,
                                                    "after": memory_snapshot()}}
    Path("output").mkdir(exist_ok=True)
    Path("output/sample_pl_30s.e2e.json").write_text(
        json.dumps(score, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(score, ensure_ascii=False, indent=2))
    # Basic sanity: we do not treat synthetic WER as a real-dialogue benchmark.
    assert score["wer"] < 1.0, "ASR failed to recognize any meaningful reference words"
    print("Real MLX + Polish alignment + Pyannote + WER/CER gate passed")


if __name__ == "__main__":
    main()
