#!/usr/bin/env python3
"""Score archived Polish ASR models against the *human-verified* reference.

Reference source: output/sample-pl.reference.md, edited manually in GitHub.
A DRAFT reference never yields WER/CER numbers. Model hypotheses are never
mutated. Speaker DER needs hand-timed turn endpoints and is not estimated here.
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quality_metrics import wer, cer

ROW = re.compile(
    r"^\[(\d{2}):(\d{2})\]\s+\*\*(NIEUSTALONY|SPEAKER_\d{2})\*\*:\s*(.*?)\s*$",
    flags=re.UNICODE,
)
OVERLAP = re.compile(r"\s*\*\*\[NAKŁADAJĄCE SIĘ GŁOSY[^\]]*\]\*\*")


def parse_reference(path: Path):
    raw = path.read_text(encoding="utf-8")
    if "REFERENCE_STATUS: VERIFIED" not in raw:
        return None
    turns = []
    for line in raw.splitlines():
        match = ROW.fullmatch(line)
        if not match:
            continue
        mm, ss, speaker, utterance = match.groups()
        utterance = OVERLAP.sub("", utterance).strip()
        if utterance:
            turns.append({"start_s_rounded": 60 * int(mm) + int(ss),
                          "speaker": speaker, "text": utterance})
    if len(turns) < 3:
        raise ValueError("Verified transcript contains fewer than three timestamped turns")
    return turns


def evaluate(reference: Path, hypotheses: Path, output: Path):
    turns = parse_reference(reference)
    if turns is None:
        print("Reference remains DRAFT — skipping metrics until manual verification")
        return False
    ref_text = " ".join(x["text"] for x in turns)
    rows = []
    for file in sorted(hypotheses.glob("*.json")):
        if file.name.endswith(".report.json"):
            continue
        obj = json.loads(file.read_text(encoding="utf-8"))
        predicted = " ".join(x.get("text", "") for x in obj.get("segments", []))
        if not predicted:
            continue
        rows.append((file.stem, wer(ref_text, predicted), cer(ref_text, predicted)))
    if not rows:
        raise FileNotFoundError("No archived transcription hypotheses in output/hypotheses")
    rows.sort(key=lambda row: (row[1], row[2]))
    result = [
        "# Benchmark polskiej mowy — odniesienie po ręcznej korekcie", "",
        f"Reference file: {reference.as_posix()}",
        "Source audio: input/sample-pl.m4a",
        "WER/CER use the complete reference transcript, not duration-normalized excerpts.",
        "",
        "| ASR and diarization | WER | CER |",
        "|---|---:|---:|",
    ]
    for model, w, c in rows:
        result.append(f"| {model} | {w:.2%} | {c:.2%} |")
    result += [
        "",
        "Speaker DER: **not calculated**. The editable Markdown has only",
        "rounded turn *start* times. Correct diarization scoring requires",
        "manually annotated start/end and overlap intervals. Do not infer",
        "that a model predicting three speaker IDs identified all three voices.",
        "",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(result), encoding="utf-8")
    print(f"Saved {len(rows)} evaluated model hypotheses in {output}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path,
                        default=Path("output/sample-pl.reference.md"))
    parser.add_argument("--hypotheses", type=Path,
                        default=Path("output/hypotheses"))
    parser.add_argument("--output", type=Path,
                        default=Path("output/reference-quality.md"))
    args = parser.parse_args()
    evaluate(args.reference, args.hypotheses, args.output)
