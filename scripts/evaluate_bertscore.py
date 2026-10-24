#!/usr/bin/env python3
"""BERTScore token embeddings evaluated on the verified Polish main dialogue.

Separate precision/recall/F1; role-balanced macro F1; zero for missing
utterances. Source timestamps are rounded: results are approximate.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_reference import parse_reference
from scripts.evaluate_semantic import critical_warnings, reference_pairs
from scripts.primary_quality import PRIMARY

MODEL_TYPE = "bert-base-multilingual-cased"


def collect_pairs(reference, legacy, hypotheses):
    turns = parse_reference(reference)
    if turns is None:
        raise ValueError("Primary reference must be VERIFIED")
    if not turns or any(t["speaker"] not in PRIMARY for t in turns):
        raise ValueError("Primary reference must contain TK and the clerk only")
    context = parse_reference(legacy) if legacy.exists() else None
    pairs = {}
    for path in sorted(hypotheses.glob("*.json")):
        if path.name.endswith(".report.json"):
            continue
        obj = json.loads(path.read_text(encoding="utf-8"))
        records = reference_pairs(turns, obj, context)
        if records:
            pairs[path.stem] = records
    if not pairs:
        raise FileNotFoundError("No word timestamp ASR hypotheses")
    return pairs


def compute(paired, scorer=None):
    flat = [(name, row) for name, rows in paired.items() for row in rows]
    nonempty = [(index, row) for index, (name, row) in enumerate(flat)
                if row["hypothesis"].strip()]
    scores = [(0., 0., 0.) for _ in flat]
    if nonempty:
        if scorer is None:
            from bert_score import score as scorer
        p, r, f = scorer(
            [row["hypothesis"] for _, row in nonempty],
            [row["reference"] for _, row in nonempty],
            model_type=MODEL_TYPE, lang="pl", device="cpu",
            batch_size=16, idf=False, verbose=False,
            rescale_with_baseline=False,
        )
        p, r, f = [x.tolist() for x in (p, r, f)]
        if len(p) != len(nonempty) or len(r) != len(p) or len(f) != len(p):
            raise ValueError("BERTScore returned unexpected batch length")
        for i, (index, _) in enumerate(nonempty):
            scores[index] = float(p[i]), float(r[i]), float(f[i])
    by_model = defaultdict(list)
    for (name, record), (precision, recall, f1) in zip(flat, scores):
        by_model[name].append({
            **record, "precision": precision, "recall": recall, "f1": f1,
            "missing": not bool(record["hypothesis"].strip()),
            "warnings": critical_warnings(record["reference"], record["hypothesis"]),
        })
    outputs = {}
    for name, rows in by_model.items():
        by_role = {}
        for speaker in PRIMARY:
            entries = [row for row in rows if row["speaker"] == speaker]
            if not entries:
                raise ValueError("Missing role in verified reference: " + speaker)
            by_role[speaker] = {
                metric: sum(row[metric] for row in entries)/len(entries)
                for metric in ("precision", "recall", "f1")
            }
        outputs[name] = {
            "roles": by_role,
            "f1": sum(by_role[x]["f1"] for x in PRIMARY)/len(PRIMARY),
            "precision": sum(by_role[x]["precision"] for x in PRIMARY)/len(PRIMARY),
            "recall": sum(by_role[x]["recall"] for x in PRIMARY)/len(PRIMARY),
            "missing": sum(row["missing"] for row in rows),
            "warnings": sum(bool(row["warnings"]) for row in rows),
            "rows": rows,
        }
    return outputs


def markdown(result, reference):
    lines = [
        "# BERTScore — polski główny dialog",
        "",
        "Ręczna referencja: " + str(reference),
        "Model tokenowy: " + MODEL_TYPE,
        "Zakres: TK i URZĘDNICZKA SĄDOWA. Trzeci rozmówca nie jest oceniany.",
        "",
        "BERTScore P/R/F1 jest nieliniowym podobieństwem kontekstowych",
        "wektorów tokenów, NIE procentem dosłownie poprawnej transkrypcji.",
        "Brak rozpoznanej kwestii otrzymuje P=R=F1=0.",
        "Ocena uwzględnia osobno alarmy numerów i negacji.",
        "Czasy źródłowe są zaokrąglone do sekund i okna są przybliżone.",
        "",
        "| Model ASR + diaryzacja | F1 TK | F1 urzędniczki | F1 macro | Precision macro | Recall macro | Braki | Alarmy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    ordered = sorted(result.items(), key=lambda x: x[1]["f1"], reverse=True)
    for name, data in ordered:
        lines.append(
            f"| {name} | {data['roles']['SPEAKER_00']['f1']:.4f}"
            f" | {data['roles']['SPEAKER_01']['f1']:.4f}"
            f" | {data['f1']:.4f} | {data['precision']:.4f}"
            f" | {data['recall']:.4f} | {data['missing']} | {data['warnings']} |"
        )
    name, data = ordered[0]
    lines.extend(["", "## Dziesięć najsłabszych fragmentów najlepszego wariantu", "", "Model: " + name])
    for row in sorted(data["rows"], key=lambda x: x["f1"])[:10]:
        t = row["time"]
        speaker = "TK" if row["speaker"] == "SPEAKER_00" else "URZĘDNICZKA SĄDOWA"
        lines.extend([
            "",
            f"- [{t//60:02d}:{t%60:02d}] {speaker}: F1 {row['f1']:.4f}",
            "  - Referencja: " + row["reference"],
            "  - ASR: " + (row["hypothesis"] or "(brak)"),
            "  - Alerty: " + (", ".join(row["warnings"]) if row["warnings"] else "brak"),
        ])
    lines.extend(["", "WER/CER oraz sentence embeddings pozostają osobnymi",
                  "metrykami. Wysoki BERTScore nie dowodzi poprawności liczb i zaprzeczeń.", ""])
    return "\n".join(lines)


def evaluate(reference, legacy, hypotheses, output, scorer=None):
    results = compute(collect_pairs(reference, legacy, hypotheses), scorer=scorer)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown(results, reference), encoding="utf-8")
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference", type=Path, default=Path("output/sample-pl.primary.reference.md"))
    p.add_argument("--legacy", type=Path, default=Path("output/sample-pl.reference.md"))
    p.add_argument("--hypotheses", type=Path, default=Path("output/hypotheses"))
    p.add_argument("--output", type=Path, default=Path("output/bertscore-quality.md"))
    args = p.parse_args()
    evaluate(args.reference, args.legacy, args.hypotheses, args.output)


if __name__ == "__main__":
    main()
