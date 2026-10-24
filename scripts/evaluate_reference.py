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
    r"^\[(\d{2}):(\d{2})\]\s+\*\*(NIEUSTALONY|SPEAKER_\d{2}|TK|URZĘDNICZKA SĄDOWA)\*\*\s*:?\s*(.*?)\s*$",
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
        speaker = {"TK": "SPEAKER_00", "URZĘDNICZKA SĄDOWA": "SPEAKER_01"}.get(speaker, speaker)
        utterance = OVERLAP.sub("", utterance).strip()
        if utterance:
            turns.append({"start_s_rounded": 60 * int(mm) + int(ss),
                          "speaker": speaker, "text": utterance})
    if not turns:
        raise ValueError("Verified transcript has no timestamped turns")
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
    explicitly_curated = reference_view.exists()
    if explicitly_curated:
        primary_turns = parse_reference(reference_view)
        if primary_turns is None:
            print("Authoritative primary reference is not VERIFIED; skipping metrics")
            return False
        if not primary_turns or any(t["speaker"] not in PRIMARY for t in primary_turns):
            raise ValueError("Primary reference may contain only TK and URZĘDNICZKA SĄDOWA")
        primary_context = turns
    else:
        # Backward compatible mode for independently supplied reference files.
        primary_turns = turns
        primary_context = None
        reference_view.write_text(primary_view(turns), encoding="utf-8")
    windows, window_metadata = focus_windows(primary_turns, context_turns=primary_context)
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
        primary = score_primary(primary_turns, obj, context_turns=primary_context)
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
        "# Benchmark polskiej mowy — główny dialog TK ↔ URZĘDNICZKA SĄDOWA", "",
        f"Główna referencja: {reference_view.as_posix()} (VERIFIED, ręczna korekta)",
        f"Pomocnicza wcześniejsza transkrypcja pełna: {reference.as_posix()}",
        "Source audio: input/sample-pl.m4a",
        "Priorytet: TK = SPEAKER_00, URZĘDNICZKA SĄDOWA = SPEAKER_01.",
        "SPEAKER_02 i kwestie poza ręcznie wskazanym głównym dialogiem są poboczne.",
        "Pełne hipotezy modeli i rozmowy z trzecią osobą są zachowane.",
        "",
        "**Ważne:** pomocnicza pełna referencja pochodzi z wcześniejszej korekty",
        "i nie została ponownie zsynchronizowana z najnowszymi poprawkami.",
        "Jej WER jest jedynie historycznym punktem odniesienia, NIE główną oceną.",
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
        "| Model | WER TK/urzędniczka (macro) | WER TK | WER urzędniczka | WER dialogu łącznie | CER dialogu | WER dawnej pełnej ref. | DER approx. |",
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
    result += ["", "### Pomocnicza metryka wobec starszej pełnej referencji", "",
               "| Model | WER dawnej pełnej ref. | CER dawnej pełnej ref. |",
               "|---|---:|---:|"]
    for row in rows:
        result.append(f"| {row['model']} | {pct(row['full_wer'])} | {pct(row['full_cer'])} |")
    result += [
        "",
        "Wyniki dialogu oceniają wyłącznie ręcznie wybrane kwestie.",
        "Czasy początku zaokrąglono do sekund; słowa przypisano do",
        "rozłącznych okien, a nie według automatycznych speaker ID.",
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
