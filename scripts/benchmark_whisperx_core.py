#!/usr/bin/env python3
"""Real upstream WhisperX VAD benchmark against historic MLX-Silero VAD.

Only compares genuine model artifacts to manually VERIFIED primary-dialogue
reference. Times in reference are rounded to seconds, so WER is a proxy.
No automatic DER or "word timestamp accuracy" is claimed without human labels.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_reference import parse_reference
from scripts.primary_quality import PRIMARY, focus_windows, score_primary

BASELINE = "mlx-whisper-large-v3-turbo-pyannote"
CORE = "mlx-whisper-large-v3-turbo-whisperx-vad-pyannote"
SOURCE_RUNS = (40109950600, 40110317780)
ORIGINAL_NAME = "mlx-whisper-large-v3-turbo-pyannote"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def import_run(run_id: int, folder: Path, hypotheses: Path) -> str:
    if run_id not in SOURCE_RUNS:
        raise ValueError(f"Unknown real experiment run: {run_id}")
    transcript = folder / ORIGINAL_NAME / "sample-pl.json"
    manifest = folder / ORIGINAL_NAME / "sample-pl.manifest.json"
    report = folder / (ORIGINAL_NAME + ".report.json")
    if not all(path.is_file() for path in (transcript, manifest, report)):
        raise FileNotFoundError(f"Missing full real-model artifact: {folder}")
    metadata = read(manifest)
    diagnostics = read(report)
    if metadata.get("memory", {}).get("stages", {}).get("vad_source") != "upstream-whisperx":
        raise ValueError("VAD source is not original WhisperX")
    if diagnostics.get("status") != "passed" or diagnostics.get("input_filename") != "sample-pl.m4a":
        raise ValueError("Original audio E2E was not successful")
    if not any(seg.get("words") for seg in read(transcript).get("segments", [])):
        raise ValueError("No word-aligned model output")
    stem = f"{CORE}-r{SOURCE_RUNS.index(run_id)+1}"
    hypotheses.mkdir(parents=True, exist_ok=True)
    for source, target in [
        (transcript, hypotheses / (stem + ".json")),
        (report, hypotheses / (stem + ".report.json")),
    ]:
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise FileExistsError(f"Refusing to overwrite immutable hypothesis {target}")
        if not target.exists():
            target.write_bytes(source.read_bytes())
    return stem


def diagnostics(obj):
    segments = obj.get("segments", [])
    words = [w for s in segments for w in s.get("words", [])]
    timed = [w for w in words if isinstance(w.get("start"), (int, float))
             and isinstance(w.get("end"), (int, float)) and w["end"] >= w["start"]]
    scores = [w["score"] for w in timed if isinstance(w.get("score"), (int, float))]
    return {
        "segments": len(segments), "words": len(words),
        "timed_percent": len(timed)/len(words) if words else 0,
        "first": round(min(w["start"] for w in timed), 3) if timed else None,
        "last": round(max(w["end"] for w in timed), 3) if timed else None,
        "missing_speaker": sum(not w.get("speaker") for w in words),
        "speakers": len({s["speaker"] for s in segments if s.get("speaker")}),
        "mean_alignment_score": sum(scores)/len(scores) if scores else None,
        "low_score_words": sum(x < .5 for x in scores),
    }


def compare(reference: Path, full: Path, hypotheses: Path):
    turns = parse_reference(reference)
    context = parse_reference(full)
    if turns is None or context is None or not turns or any(t["speaker"] not in PRIMARY for t in turns):
        raise ValueError("Manually VERIFIED TK and clerk transcript is required")
    _, coverage = focus_windows(turns, context_turns=context)
    stems = [BASELINE, CORE + "-r1", CORE + "-r2"]
    results = []
    for stem in stems:
        path = hypotheses / (stem + ".json")
        if not path.exists():
            raise FileNotFoundError(f"Missing E2E archive {path}")
        obj = read(path)
        score = score_primary(turns, obj, context_turns=context)
        if score is None or score["macro_wer"] is None:
            raise ValueError(f"No real timed words to compare in {stem}")
        report = hypotheses / (stem + ".report.json")
        runtime = read(report).get("elapsed_seconds") if report.exists() else None
        results.append({"variant": stem, "score": score,
                        "word_diagnostics": diagnostics(obj), "runtime": runtime})
    return results, coverage


def fmt_pct(value):
    return f"{value:.2%}" if value is not None else "—"


def markdown(rows, coverage, reference):
    output = [
        "# Porównanie: WhisperX Pyannote VAD vs wcześniejsze MLX Silero VAD",
        "",
        "Referencja (ręcznie VERIFIED): " + str(reference),
        "Audio: input/sample-pl.m4a; ASR: mlx-whisper large-v3-turbo;",
        "alignment: Polish Wav2Vec2 CTC; diarization: Pyannote Community-1.",
        "",
        "Podstawowa zmienna: sposób VAD i podziału na fragmenty.",
        "Dwa przebiegi oryginalnego WhisperX pokazują zmienność inferencji.",
        "WER/CER są przybliżone: ludzkie początki kwestii mają tylko sekundy",
        "i nie ma zweryfikowanych czasów końca. Nie wyciągać wniosków o DER.",
        "",
        "| Wariant | WER TK | WER urzędniczki | WER macro | CER dialogu | Pokrycie czasów słów | Słowa / segmenty | Słowa bez mówcy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        m, d = row["score"], row["word_diagnostics"]
        roles = m["per_speaker"]
        output.append(
            f"| {row['variant']} | {fmt_pct(roles['SPEAKER_00']['wer'])}"
            f" | {fmt_pct(roles['SPEAKER_01']['wer'])}"
            f" | {fmt_pct(m['macro_wer'])}"
            f" | {fmt_pct(m['focus_cer'])}"
            f" | {fmt_pct(d['timed_percent'])}"
            f" | {d['words']} / {d['segments']}"
            f" | {d['missing_speaker']} |"
        )
    output.extend([
        "",
        "## Diagnostyka alignmentu",
        "",
        "| Wariant | Pierwsze i ostatnie słowo (sekundy) | Śr. score Wav2Vec2* | Score poniżej 0.5 | Liczba mówców | Czas E2E |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for row in rows:
        d = row["word_diagnostics"]
        mean = f"{d['mean_alignment_score']:.3f}" if d["mean_alignment_score"] is not None else "—"
        runtime = f"{row['runtime']:.1f} s" if isinstance(row["runtime"], (int, float)) else "—"
        output.append(f"| {row['variant']} | {d['first']} / {d['last']}"
                      f" | {mean} | {d['low_score_words']}"
                      f" | {d['speakers']} | {runtime} |")
    output.extend([
        "",
        f"Referencja: {coverage['included_reference_words']}/{coverage['primary_reference_words']}"
        f" słów w rozłącznych oknach ({fmt_pct(coverage['coverage'])}).",
        "*Wynik alignment-score NIE jest miarą dokładności czasów słów.",
        "Kompletne timestampy nie dowodzą trafności ich położenia.",
        "MAE start/end wymaga ludzkiej referencji czasowej, DER wymaga",
        "ręcznie oznaczonego diarization timeline. Oba pozostają niewyznaczone.",
        "E5 i BERTScore liczą osobne workflows na tych samych JSON.",
        "",
    ])
    return "\n".join(output)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference", type=Path, default=Path("output/sample-pl.primary.reference.md"))
    p.add_argument("--full", type=Path, default=Path("output/sample-pl.reference.md"))
    p.add_argument("--hypotheses", type=Path, default=Path("output/hypotheses"))
    p.add_argument("--output", type=Path, default=Path("output/whisperx-core-benchmark.md"))
    p.add_argument("--source", action="append", default=[],
                   help="RUN_ID:directory of complete immutable downloaded artifacts")
    args = p.parse_args()
    seen = set()
    for entry in args.source:
        raw, folder = entry.split(":", 1)
        run_id = int(raw)
        if run_id in seen:
            raise ValueError("Duplicate run")
        import_run(run_id, Path(folder), args.hypotheses)
        seen.add(run_id)
    if seen and seen != set(SOURCE_RUNS):
        raise ValueError(f"Need both runs {SOURCE_RUNS}")
    rows, coverage = compare(args.reference, args.full, args.hypotheses)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown(rows, coverage, args.reference), encoding="utf-8")
    print(f"Saved 3-way controlled ASR regression report: {args.output}")
    for row in rows:
        print(f"{row['variant']}: WER macro {fmt_pct(row['score']['macro_wer'])}")


if __name__ == "__main__":
    main()
