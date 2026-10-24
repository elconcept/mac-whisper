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
from quality_metrics import wer, cer, der

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
    speaker_file = reference.with_name("sample-pl.speakers.reference.json")
    reference_speakers = None
    if speaker_file.exists():
        labels = json.loads(speaker_file.read_text(encoding="utf-8"))
        if labels.get("verified") is True:
            reference_speakers = labels["intervals"]
            if not reference_speakers or any(
                not isinstance(x.get("speaker"), str)
                or not isinstance(x.get("start"), (int, float))
                or not isinstance(x.get("end"), (int, float))
                or x["end"] <= x["start"]
                for x in reference_speakers
            ):
                raise ValueError("Human speaker reference requires verified valid intervals")
    rows = []
    for file in sorted(hypotheses.glob("*.json")):
        if file.name.endswith(".report.json"):
            continue
        obj = json.loads(file.read_text(encoding="utf-8"))
        predicted = " ".join(x.get("text", "") for x in obj.get("segments", []))
        if not predicted:
            continue
        model_speakers = obj.get("diarization_intervals") or [
            {"start": x["start"], "end": x["end"], "speaker": x["speaker"]}
            for x in obj.get("segments", []) if x.get("speaker")
        ]
        speaker_error = der(reference_speakers, model_speakers) if reference_speakers else None
        rows.append((file.stem, wer(ref_text, predicted), cer(ref_text, predicted), speaker_error))
    if not rows:
        raise FileNotFoundError("No archived transcription hypotheses in output/hypotheses")
    rows.sort(key=lambda row: (row[1], row[2]))
    result = [
        "# Benchmark polskiej mowy — odniesienie po ręcznej korekcie", "",
        f"Reference file: {reference.as_posix()}",
        "Source audio: input/sample-pl.m4a",
        "WER/CER use the complete reference transcript, not duration-normalized excerpts.",
        "",
        "| ASR and diarization | WER | CER | DER approximation |",
        "|---|---:|---:|---:|",
    ]
    for model, w, c, d in rows:
        der_cell = f"{d:.2%}" if d is not None else "—"
        result.append(f"| {model} | {w:.2%} | {c:.2%} | {der_cell} |")
    result += [
        "",
        "Speaker DER is **not calculated** until an independently manually",
        "verified output/sample-pl.speakers.reference.json provides start/end/overlap",
        "intervals. If supplied, the table contains a 100-ms-frame approximate",
        "speaker-time error after speaker-label permutation, NOT pyannote's",
        "official collar- and overlap-adjusted DER. This approximation is sensitive",
        "to exact boundaries and should not be treated as legal evidence.",
        "Model predictions alone do not establish individual identities.",
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
