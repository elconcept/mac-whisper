"""Targeted ASR scoring: SPEAKER_00 and SPEAKER_01 are the interaction.

The third voice (SPEAKER_02) is a bystander / interference: retain it in
the recording and full hypotheses, but exclude its reference turns from the
primary dialogue metric.

REFERENCE timestamps are rounded to whole seconds and have no end times.
Consequently this is a TIME-WINDOW PROXY for WER/CER, *not* true
speaker-attributed WER or a diarization metric. Predicted words are included
based on their actual timestamp, ignoring the model's anonymous speaker IDs.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from quality_metrics import wer, cer, normalize

PRIMARY = ("SPEAKER_00", "SPEAKER_01")
WINDOW_START_TOLERANCE = 0.20
WINDOW_MAX_SECONDS = 20.0
LAST_WINDOW_SECONDS = 8.0
MIN_WINDOW_SECONDS = 0.35


def focus_windows(turns: list[dict], *, context_turns: list[dict] | None = None) -> tuple[list[dict], dict]:
    """Select only curated dialog, use full audio events as boundaries.

    The user's explicit corrected dialog is authoritative; historical
    full-transcript turns can delimit (but never *create*) a scored window.
    """
    grouped: dict[int, list[dict]] = defaultdict(list)
    for turn in turns:
        grouped[int(turn["start_s_rounded"])].append(turn)
    context: dict[int, list[dict]] = defaultdict(list)
    for turn in context_turns or []:
        context[int(turn["start_s_rounded"])].append(turn)
    timestamps = sorted(set(grouped) | set(context))
    selected = []
    skipped_ambiguous = []
    for index, stamp in enumerate(timestamps):
        entries = grouped.get(stamp, [])
        if not entries:
            continue
        speakers = {row["speaker"] for row in entries}
        bystander_overlap = any(row["speaker"] not in PRIMARY for row in context.get(stamp, []))
        if len(speakers) != 1 or bystander_overlap:
            skipped_ambiguous.append(stamp)
            continue
        speaker = next(iter(speakers))
        if speaker not in PRIMARY:
            continue
        following = (timestamps[index + 1]
                     if index + 1 < len(timestamps)
                     else stamp + LAST_WINDOW_SECONDS)
        start = max(0.0, stamp - WINDOW_START_TOLERANCE)
        end = min(stamp + WINDOW_MAX_SECONDS,
                  following - WINDOW_START_TOLERANCE)
        if end - start < MIN_WINDOW_SECONDS:
            skipped_ambiguous.append(stamp)
            continue
        selected.append({
            "start": start, "end": end, "speaker": speaker,
            "text": " ".join(row["text"] for row in entries),
            "source_second": stamp,
        })
    total_primary = sum(len(normalize(x["text"]).split()) for x in turns
                        if x["speaker"] in PRIMARY)
    measured = sum(len(normalize(x["text"]).split()) for x in selected)
    return selected, {
        "primary_reference_words": total_primary,
        "included_reference_words": measured,
        "coverage": measured / total_primary if total_primary else 0,
        "ambiguous_timestamp_seconds": skipped_ambiguous,
    }


def _timed_model_words(result: dict) -> list[dict]:
    words = []
    for segment in result.get("segments", []):
        for entry in segment.get("words") or []:
            a, b = entry.get("start"), entry.get("end")
            if isinstance(a, (int, float)) and isinstance(b, (int, float)) and b >= a:
                words.append({"time": (a + b) / 2.0, "word": str(entry.get("word", ""))})
    return sorted(words, key=lambda item: item["time"])


def score_primary(turns: list[dict], model: dict, *,
                  context_turns: list[dict] | None = None) -> dict | None:
    """Return role-focused proxy metrics; None if no timed words exist."""
    windows, details = focus_windows(turns, context_turns=context_turns)
    words = _timed_model_words(model)
    if not words or not windows:
        return None
    predicted_by_speaker: dict[str, list[str]] = {who: [] for who in PRIMARY}
    selected_predicted = []
    for word in words:
        # Focus windows are disjoint by construction. Ignore utterances that
        # occur only in the bystander's reference time windows.
        window = next((w for w in windows
                       if w["start"] <= word["time"] < w["end"]), None)
        if window is not None:
            predicted_by_speaker[window["speaker"]].append(word["word"])
            selected_predicted.append(word["word"])

    reference = " ".join(w["text"] for w in windows)
    hypothesis = " ".join(selected_predicted)
    per_speaker = {}
    for who in PRIMARY:
        ref = " ".join(w["text"] for w in windows if w["speaker"] == who)
        if ref and normalize(ref):
            hyp = " ".join(predicted_by_speaker[who])
            per_speaker[who] = {"wer": wer(ref, hyp), "cer": cer(ref, hyp)}
        else:
            per_speaker[who] = {"wer": None, "cer": None}
    present = [d["wer"] for d in per_speaker.values() if d["wer"] is not None]
    return {
        "focus_wer": wer(reference, hypothesis),
        "focus_cer": cer(reference, hypothesis),
        "macro_wer": sum(present) / len(present) if present else None,
        "per_speaker": per_speaker,
        "timed_hypothesis_words": len(words),
        "scored_hypothesis_words": len(selected_predicted),
        "windows": len(windows),
        **details,
    }


def primary_view(turns: list[dict]) -> str:
    """Derive an independent view, never modify the human-corrected original."""
    windows, metadata = focus_windows(turns)
    scored_seconds = {w["source_second"] for w in windows}
    lines = [
        "# Główny dialog — ręcznie zweryfikowany tekst (widok pochodny)",
        "",
        "Źródło: [sample-pl.reference.md](sample-pl.reference.md).",
        "SPEAKER_00 i SPEAKER_01 są osobami kluczowymi. SPEAKER_02 pozostaje",
        "w oryginalnym pliku jako rozmówca poboczny / zakłócenie.",
        "",
        "Wskaźniki jakości używają tylko niejednoznacznych czasowo rozdzielonych",
        "okien. Wypowiedzi z tego samego znacznika sekundy, gdy mówią różne",
        "osoby, są zachowane poniżej, lecz NIE oceniane czasowo.",
        "",
    ]
    for row in turns:
        if row["speaker"] not in PRIMARY:
            continue
        t = row["start_s_rounded"]
        suffix = " [POMINIĘTE W METRYCE: kolizja czasu]" if t not in scored_seconds else ""
        lines.append(f"[{t//60:02d}:{t%60:02d}] **{row['speaker']}**: {row['text']}{suffix}")
    lines.extend([
        "",
        f"Udział referencyjnych słów w ocenie: {metadata['coverage']:.1%}",
        f"Niejednoznaczne sekundy: {metadata['ambiguous_timestamp_seconds']}",
        "",
    ])
    return "\n".join(lines)
