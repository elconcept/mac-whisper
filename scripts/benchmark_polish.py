#!/usr/bin/env python3
"""Reference-aware WER/CER, DER and word-alignment benchmark.

Usage: python scripts/benchmark_polish.py cases.json --model large-v3-turbo --vad off
cases.json:
[
 {"audio":"path/to/sample.wav",
  "reference_text":"weryfikowany ręcznie tekst wypowiedzi",
  "speaker_intervals":[{"start":0.0,"end":2.0,"speaker":"A"}],
  "word_intervals":[{"word":"weryfikowany","start":0.0,"end":0.5}]}
]
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from mac_whisper_app import transcribe_one
from quality_metrics import wer, cer, der, word_alignment_mae


def run(manifest: Path, *, model: str, vad: bool, output: Path):
    cases = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("Manifest must be a nonempty list of reference cases")
    report = []
    for row in cases:
        path = (manifest.parent / row["audio"]).resolve()
        try:
            result = transcribe_one(
                path, output / "transcripts",
                overrides={"model": model, "vad": vad, "format": "json",
                           "diarization": bool(row.get("speaker_intervals")), "alignment": True}
            )
            segments = result.get("segments", [])
            pred_text = " ".join(s.get("text", "") for s in segments)
            pred_speakers = [
                {"start": s["start"], "end": s["end"], "speaker": s["speaker"]}
                for s in segments if s.get("speaker")
            ]
            words = [w for s in segments for w in s.get("words", [])]
            metrics = {
                "file": row["audio"], "wer": wer(row["reference_text"], pred_text),
                "cer": cer(row["reference_text"], pred_text),
                "der_approx": der(row.get("speaker_intervals", []), pred_speakers)
                    if row.get("speaker_intervals") else None,
                "alignment_mae_s": word_alignment_mae(row.get("word_intervals", []), words)
                    if row.get("word_intervals") else None,
            }
        except Exception as exc:
            metrics = {"file": row["audio"], "error": str(exc)}
        report.append(metrics)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "quality_report.json"
    path.write_text(json.dumps({
        "model": model, "vad": vad, "cases": report
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(path)
    if any("error" in x for x in report):
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--model", default="large-v3-turbo")
    parser.add_argument("--vad", choices=["off", "silero"], default="off")
    parser.add_argument("--output", type=Path, default=Path("output/benchmark"))
    args = parser.parse_args()
    run(args.manifest, model=args.model, vad=args.vad == "silero", output=args.output)
