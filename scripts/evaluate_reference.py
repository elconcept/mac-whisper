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
from scripts.primary_quality import PRIMARY, primary_view, score_primary, focus_windows

ROW = re.compile(
    r"^\[(\d{2}):(\d{2})\]\s+\*\*(NIEUSTALONY|SPEAKER_\d{2})\*\*\s*:?\s*(.*?)\s*$",
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
    reference_view = reference.with_name("sample-pl.primary.reference.md")
    reference_view.write_text(primary_view(turns), encoding="utf-8")
    windows, window_metadata = focus_windows(turns)
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
        primary = score_primary(turns, obj)
        rows.append({
            "model": file.stem,
            "full_wer": wer(ref_text, predicted),
            "full_cer": cer(ref_text, predicted),
            "der": speaker_error,
            "primary": primary,
        })
    if not rows:
        raise FileNotFoundError("No archived transcription hypotheses in output/hypotheses")
    # Main dialogue determines ranking; full audio is auxiliary only.
    rows.sort(key=lambda row: (
        row["primary"]["macro_wer"] if row["primary"] and row["primary"]["macro_wer"] is not None else float("inf"),
        row["primary"]["focus_wer"] if row["primary"] else float("inf"),
        row["full_wer"],
    ))
    result = [
        "# Benchmark polskiej mowy — główny dialog 00/01", "",
        f"Reference: {reference.as_posix()} (VERIFIED, niezmieniony)",
        "Source audio: input/sample-pl.m4a",
        "Priorytet: SPEAKER_00 i SPEAKER_01. SPEAKER_02 jest poboczny.",
        "Trzeci głos nadal jest zachowany w oryginalnej transkrypcji.",
        "",
        "UWAGA: oznaczenia w referencji mają dokładność tylko 1 sekundy",
        "i brak czasów końca. Wyniki 00/01 są **przybliżonym WER/CER",
        "w oknach czasowych**, a nie prawdziwym diarized WER.",
        "Modelowe speaker ID nie służą do wybierania słów w oknach.",
        "Zachowane są więc pomyłki w rozpoznaniu mowy i zakłócenia wewnątrz",
        "głównych wypowiedzi; omijane są odrębne okna SPEAKER_02.",
        f"Pokrycie głównej referencji: {window_metadata['coverage']:.1%}",
        f"({window_metadata['included_reference_words']} / {window_metadata['primary_reference_words']} słów).",
        f"Pominięte sekundy ze zbieżnymi głosami: {window_metadata['ambiguous_timestamp_seconds']}.",
        "",
        "| Model | WER 00/01 (macro) | WER 00 | WER 01 | WER 00/01 łącznie | CER 00/01 | WER całe audio | DER approx. |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    pct = lambda value: f"{value:.2%}" if value is not None else "—"
    for row in rows:
        primary = row["primary"]
        r0 = primary["per_speaker"]["SPEAKER_00"]["wer"] if primary else None
        r1 = primary["per_speaker"]["SPEAKER_01"]["wer"] if primary else None
        result.append(
            f"| {row['model']} | {pct(primary['macro_wer'] if primary else None)} "
            f"| {pct(r0)} | {pct(r1)} | "
            f"{pct(primary['focus_wer'] if primary else None)} | "
            f"{pct(primary['focus_cer'] if primary else None)} | "
            f"{pct(row['full_wer'])} | {pct(row['der'])} |"
        )
    result += ["", "### Metryka dodatkowa: wszystkie wypowiedzi", "",
               "| Model | WER całego nagrania | CER całego nagrania |",
               "|---|---:|---:|"]
    for row in rows:
        result.append(f"| {row['model']} | {pct(row['full_wer'])} | {pct(row['full_cer'])} |")
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
