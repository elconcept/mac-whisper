#!/usr/bin/env python3
"""Create an editable speaker timeline *template* from model hypothesis.

Never pretend model predictions are human annotations. Do not overwrite
previous corrections. Manually copy the template to
output/sample-pl.speakers.reference.json, verify boundaries and set
verified=true before diarization scoring.
"""
from __future__ import annotations
import json
from pathlib import Path

SOURCE = Path("output/hypotheses/mlx-audio-large-v3-turbo-mlx-nemotron-bf16.json")
TARGET = Path("output/sample-pl.speakers.template.json")

def main():
    if TARGET.exists():
        print("Speaker template exists; preserved")
        return
    result = json.loads(SOURCE.read_text(encoding="utf-8"))
    diar = result.get("diarization_intervals") or [
        {"start": s["start"], "end": s["end"], "speaker": s["speaker"]}
        for s in result.get("segments", [])
        if s.get("speaker")
    ]
    if not diar:
        raise RuntimeError("No predicted speaker timeline available")
    target = {
        "source_audio": "input/sample-pl.m4a",
        "status": "MODEL_PREDICTION_NOT_VERIFIED",
        "verified": False,
        "instruction": "Copy to sample-pl.speakers.reference.json, correct starts/ends/overlaps and set verified=true",
        "intervals": [
            {"start": round(float(x["start"]), 3),
             "end": round(float(x["end"]), 3),
             "speaker": str(x["speaker"])}
            for x in diar
        ],
    }
    TARGET.write_text(json.dumps(target, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Created", TARGET, "with", len(diar), "candidate speaker intervals")

if __name__ == "__main__":
    main()
