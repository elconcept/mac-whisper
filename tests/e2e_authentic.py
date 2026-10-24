#!/usr/bin/env python3
"""Run *real models* on the public Polish 3-voice recording in input/.

This is NOT a mocked pytest or a WER/DER benchmark. No verified human
word transcript or speaker activity annotation exists in the repo yet.
Save diagnostic reports for each ASR/diarization combination.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, default=Path("input/sample-pl.m4a"))
    parser.add_argument("--asr-backend", choices=("mlx-whisper", "mlx-audio"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--diarize", required=True)
    parser.add_argument("--vad-backend", default="mlx-silero")
    parser.add_argument("--no-alignment", action="store_true")
    parser.add_argument("--reference-text", type=Path)
    parser.add_argument("--output", type=Path, default=Path("output/authentic-e2e"))
    args = parser.parse_args(argv)

    load_dotenv(override=not bool(os.getenv("HF_TOKEN")))
    if not args.audio.exists():
        raise FileNotFoundError(f"Missing authentic Polish recording: {args.audio}")
    if args.diarize == "pyannote" and not os.getenv("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN missing. Do not mark Pyannote E2E passed.")

    from mac_whisper_app import transcribe_one
    from quality_metrics import wer, cer

    label = "-".join((args.asr_backend, args.model, args.diarize)).replace("/", "_")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    events = []
    try:
        result = transcribe_one(
            args.audio, args.output / label,
            overrides={
                "asr_backend": args.asr_backend,
                "model": args.model,
                "diarization_backend": args.diarize,
                "vad": True,
                "vad_backend": args.vad_backend,
                "alignment": not args.no_alignment,
                "diarization": True,
                "turbo": args.model == "large-v3-turbo",
                "format": "json",
                "min_speakers": 2 if args.diarize == "pyannote" else None,
                "max_speakers": 4 if args.diarize == "pyannote" else None,
            },
            on_segment=lambda segment: events.append({
                "start": segment.get("start"),
                "end": segment.get("end"),
                "chars": len(segment.get("text", "")),
            }),
            progress=lambda message: print("stage:", message, flush=True),
        )
    except Exception as error:
        report = {"status": "error", "mode": label, "error": str(error),
                  "elapsed_seconds": round(time.monotonic() - started, 1)}
        (args.output / f"{label}.report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        raise

    segments = result.get("segments", [])
    words = [word for seg in segments for word in seg.get("words", [])]
    diar = result.get("diarization_intervals", [])
    distinct = sorted({str(row["speaker"]) for row in diar} |
                      {str(seg["speaker"]) for seg in segments if seg.get("speaker")})
    uncertain = sum(1 for seg in segments if seg.get("speaker_ambiguous"))
    missed = sum(1 for seg in segments if not seg.get("speaker"))
    hyp = " ".join(seg.get("text", "") for seg in segments)
    score = {"wer": None, "cer": None}
    if args.reference_text and args.reference_text.is_file():
        truth = args.reference_text.read_text(encoding="utf-8")
        score = {"wer": wer(truth, hyp), "cer": cer(truth, hyp)}

    report = {
        "status": "passed" if segments and words and distinct else "failed",
        "mode": label,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_filename": args.audio.name,
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "streaming_callback_events": len(events),
        "segments": len(segments),
        "words": len(words),
        "observed_speaker_count": len(distinct),
        "observed_speaker_labels": distinct,
        "expected_speakers_described_by_user": 3,
        "detected_all_three": len(distinct) >= 3,
        "segments_without_speaker": missed,
        "segments_in_overlap": uncertain,
        "overlap_intervals": len(result.get("overlap_intervals", [])),
        "wer": score["wer"],
        "cer": score["cer"],
        "no_manual_reference": not bool(args.reference_text and args.reference_text.is_file()),
        "note": "Third speaker is distant/quiet; auto speaker labels do not verify identity.",
    }
    report_file = args.output / f"{label}.report.json"
    report_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if len(distinct) < 3:
        print("::warning::Third distant speaker not independently confirmed by this model.")
    assert segments, "ASR produced no segments"
    assert words, "Polish alignment produced no words"
    assert distinct, "Diarization produced no speaker attribution"
    assert events, "Real transcription failed to stream any segments"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
