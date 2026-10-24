#!/usr/bin/env python3
"""Continuous vector-based ASR benchmark, with separate critical-fact flags.

Compares human-curated TK/clerk utterances to transcript words falling in
their original-clock intervals. Cosines are uncalibrated similarities, NOT
ASR accuracy percentages. Does not use model-assigned speaker labels.
"""
from __future__ import annotations
import argparse
import json
import math
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quality_metrics import normalize, wer
from scripts.evaluate_reference import parse_reference
from scripts.primary_quality import PRIMARY, _timed_model_words, focus_windows

EMBEDDING_MODELS = ("intfloat/multilingual-e5-small", "sdadas/mmlw-e5-small")
NAMES = {"SPEAKER_00": "TK", "SPEAKER_01": "URZĘDNICZKA SĄDOWA"}
NEGATIONS = {"nie", "nigdy", "bez", "brak", "żaden", "żadnego", "żadna", "żadnej"}
NUMBER_PATTERN = re.compile(r"(?<!\w)(?:[lL]\s*[-–]?\s*\d+|\d+(?:[-–]\d+)*)(?!\w)")


def key_facts(text: str) -> dict:
    numbers = [re.sub(r"[\s–-]+", "", m).upper()
               for m in NUMBER_PATTERN.findall(text)]
    neg = [w for w in normalize(text).split() if w in NEGATIONS]
    return {"numbers": numbers, "negations": neg}


def critical_warnings(reference: str, hypothesis: str) -> list[str]:
    a, b = key_facts(reference), key_facts(hypothesis)
    flags = []
    if a["numbers"] != b["numbers"]:
        flags.append("NUMERY")
    if sorted(a["negations"]) != sorted(b["negations"]):
        flags.append("NEGACJA")
    return flags


def cosine(a, b) -> float:
    numerator = sum(float(x) * float(y) for x, y in zip(a, b))
    aa = math.sqrt(sum(float(x)**2 for x in a))
    bb = math.sqrt(sum(float(x)**2 for x in b))
    if aa <= 0 or bb <= 0:
        return 0.0
    return max(-1.0, min(1.0, numerator/(aa*bb)))


def reference_pairs(turns, hypothesis, context=None):
    windows, details = focus_windows(turns, context_turns=context)
    words = _timed_model_words(hypothesis)
    return [{
        "time": window["source_second"],
        "speaker": window["speaker"],
        "reference": window["text"],
        "hypothesis": " ".join(
            w["word"] for w in words
            if window["start"] <= w["time"] < window["end"]
        ).strip(),
    } for window in windows]


def compute_scores(pairs, encoder, model_id):
    if not pairs:
        raise ValueError("No primary-dialogue utterances to evaluate")
    refs = [p["reference"] for p in pairs]
    hyps = [p["hypothesis"] for p in pairs]
    n = len(pairs)
    if model_id == EMBEDDING_MODELS[0]:
        payload = ["query: " + s for s in refs + hyps]
        vectors = encoder.encode(payload, normalize_embeddings=True,
                                 batch_size=32, show_progress_bar=False)
        similarities = [cosine(vectors[i], vectors[n+i]) for i in range(n)]
    elif model_id == EMBEDDING_MODELS[1]:
        # Polish E5 retrieval model expects different prefixes.
        # Use both directions to avoid order bias.
        payload = (["query: " + s for s in refs] +
                   ["passage: " + s for s in hyps] +
                   ["query: " + s for s in hyps] +
                   ["passage: " + s for s in refs])
        vectors = encoder.encode(payload, normalize_embeddings=True,
                                 batch_size=32, show_progress_bar=False)
        similarities = [
            .5 * (cosine(vectors[i], vectors[n+i]) +
                  cosine(vectors[2*n+i], vectors[3*n+i]))
            for i in range(n)
        ]
    else:
        raise ValueError("Unrecognized encoder: " + model_id)
    rows = []
    for pair, similarity in zip(pairs, similarities):
        # The embedding of an empty sentence can still have high cosine with
        # actual sentences (E5 anisotropy). A missing ASR prediction must not
        # earn semantic credit. Zero is an explicit MISS sentinel, not cosine.
        missing = not pair["hypothesis"].strip()
        if missing:
            similarity = 0.0
        rows.append({**pair, "similarity": similarity,
                     "missing": missing,
                     "warnings": (["BRAK TRANSKRYPCJI"] if missing else []) +
                         critical_warnings(pair["reference"], pair["hypothesis"]),
                     "wer": wer(pair["reference"], pair["hypothesis"])})
    roles = {}
    for role in PRIMARY:
        values = [row["similarity"] for row in rows
                  if row["speaker"] == role]
        if not values:
            raise ValueError("No reference turns for " + role)
        roles[role] = sum(values)/len(values)
    return {
        "cosine_macro": sum(roles.values())/len(roles),
        "roles": roles,
        "warnings_count": sum(bool(row["warnings"]) for row in rows),
        "missing_turns": sum(row["missing"] for row in rows),
        "rows": rows,
    }


def report_markdown(results, reference):
    lines = [
        "# Semantyczna ocena polskiej transkrypcji",
        "",
        "Ręczna referencja: " + str(reference),
        "Priorytet: TK i URZĘDNICZKA SĄDOWA, trzeci głos pomijany",
        "jako osobna kwestia referencyjna.",
        "",
        "**Cosine embeddingów jest podobieństwem, a nie procentem",
        "poprawności tekstu ani skalibrowanym prawdopodobieństwem.**",
        "Nie można porównywać wartości bezpośrednio między encoderami.",
        "Nie ustalamy arbitralnego progu zaliczenia.",
        "Pusta hipoteza ASR dostaje specjalną karę 0.0 (BRAK TRANSKRYPCJI),",
        "a nie cosine wektora pustego tekstu, który byłby myląco wysoki.",
        "",
        "Każda ręcznie zweryfikowana kwestia została sparowana z tekstem",
        "rozpoznanym w jej oknie czasowym. Są to tylko czasy początku",
        "zaokrąglone do sekundy, co ogranicza precyzję pomiaru.",
        "Wypowiedzi poboczne wewnątrz głównych okien mogą nadal wpływać",
        "na wynik; modelowe ID mówców nie są traktowane jako prawda.",
        "",
    ]
    for embedder, scores in results.items():
        lines.extend([
            "## Encoder: " + embedder,
            "",
            "| Model ASR / diaryzacja | TK cosine | Urzędniczka cosine | Średnia ról | Puste kwestie | Alarmy |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        rank = sorted(scores.items(), key=lambda item: item[1]["cosine_macro"], reverse=True)
        for name, result in rank:
            lines.append(
                f"| {name} | {result['roles']['SPEAKER_00']:.4f}"
                f" | {result['roles']['SPEAKER_01']:.4f}"
                f" | {result['cosine_macro']:.4f}"
                f" | {result['missing_turns']}"
                f" | {result['warnings_count']} |"
            )
        lines.extend([
            "",
            "### Dziesięć najsłabiej dopasowanych kwestii najlepszego wariantu",
            "",
        ])
        best_name, best_result = rank[0]
        lines.append("Wariant: " + best_name)
        for row in sorted(best_result["rows"], key=lambda x: x["similarity"])[:10]:
            t = row["time"]
            alarms = ", ".join(row["warnings"]) if row["warnings"] else "brak"
            lines.extend([
                "",
                f"- [{t//60:02d}:{t%60:02d}] {NAMES[row['speaker']]} | cosine {row['similarity']:.4f} | WER {row['wer']:.2%} | flagi: {alarms}",
                "  - Referencja: " + row["reference"],
                "  - ASR: " + (row["hypothesis"] or "(brak)"),
            ])
        lines.extend(["", "---", ""])
    lines.extend([
        "## Ważne ograniczenia",
        "",
        "Negacja i numery (np. 73-39 lub L-419) są dodatkowo flagowane,",
        "ponieważ duży cosine nie gwarantuje niezmienionego znaczenia.",
        "Flagi są sygnałami do sprawdzenia, a nie rozstrzygnięciem.",
        "WER/CER z dokładnym dopasowaniem leksykalnym pozostają w",
        "output/reference-quality.md; DER wymaga odrębnej referencji",
        "z granicami czasowymi każdego głosu.",
        "",
    ])
    return "\n".join(lines)


def load_encoder(model_id):
    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer
    snapshot = snapshot_download(repo_id=model_id, token=False)
    return SentenceTransformer(snapshot, device="cpu", trust_remote_code=False)


def evaluate(reference, legacy, hypotheses, output,
             models=EMBEDDING_MODELS, encoders=None):
    turns = parse_reference(reference)
    if turns is None:
        print("Reference is not VERIFIED; no semantic scores")
        return False
    if not turns or any(turn["speaker"] not in PRIMARY for turn in turns):
        raise ValueError("Curated reference must contain only TK and clerk")
    context = parse_reference(legacy) if legacy.exists() else None
    predictions = {}
    for path in sorted(hypotheses.glob("*.json")):
        if path.name.endswith(".report.json"):
            continue
        obj = json.loads(path.read_text(encoding="utf-8"))
        pairs = reference_pairs(turns, obj, context)
        if pairs:
            predictions[path.stem] = pairs
    if not predictions:
        raise FileNotFoundError("No archived ASR word-timestamp hypotheses")
    scores = {}
    for model in models:
        encoder = load_encoder(model) if encoders is None else encoders[model]
        scores[model] = {
            name: compute_scores(pairs, encoder, model)
            for name, pairs in predictions.items()
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report_markdown(scores, reference), encoding="utf-8")
    print(f"Saved {len(predictions)} variants with {len(models)} vector encoders to {output}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path,
                        default=Path("output/sample-pl.primary.reference.md"))
    parser.add_argument("--legacy", type=Path,
                        default=Path("output/sample-pl.reference.md"))
    parser.add_argument("--hypotheses", type=Path,
                        default=Path("output/hypotheses"))
    parser.add_argument("--output", type=Path,
                        default=Path("output/semantic-quality.md"))
    parser.add_argument("--models", nargs="+", choices=EMBEDDING_MODELS,
                        default=list(EMBEDDING_MODELS))
    args = parser.parse_args()
    evaluate(args.reference, args.legacy, args.hypotheses,
             args.output, models=args.models)
