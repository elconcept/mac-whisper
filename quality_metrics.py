"""Simple, reproducible Polish ASR and diarization benchmark metrics.

WER/CER are edit-distance-based. DER samples time at 10 Hz and optimally
maps anonymous predicted speakers onto reference speakers (up to 6 labels).
"""
from __future__ import annotations
from itertools import permutations
import re
import unicodedata


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text.lower())
    return " ".join(re.findall(r"[^\W_]+", text, flags=re.UNICODE))


def edit_distance(a: list, b: list) -> int:
    costs = list(range(len(b) + 1))
    for i, item in enumerate(a, 1):
        next_costs = [i]
        for j, other in enumerate(b, 1):
            next_costs.append(min(next_costs[-1] + 1, costs[j] + 1,
                                  costs[j - 1] + (item != other)))
        costs = next_costs
    return costs[-1]


def wer(reference: str, hypothesis: str) -> float:
    a = normalize(reference).split()
    b = normalize(hypothesis).split()
    return edit_distance(a, b) / max(1, len(a))


def cer(reference: str, hypothesis: str) -> float:
    a = list(normalize(reference).replace(" ", ""))
    b = list(normalize(hypothesis).replace(" ", ""))
    return edit_distance(a, b) / max(1, len(a))


def der(reference: list[dict], predicted: list[dict], step: float = 0.1) -> float | None:
    """Speaker-time error rate; count missing, false alarm and wrong speaker.

    Overlap is evaluated as speaker-sets. Maps model speaker IDs automatically.
    Inputs have start, end and speaker keys. Returns None for no reference speech.
    """
    ids_ref = sorted({str(x["speaker"]) for x in reference})
    ids_pred = sorted({str(x["speaker"]) for x in predicted})
    if not ids_ref:
        return None
    if len(ids_ref) > 6 or len(ids_pred) > 6:
        raise ValueError("Exact DER evaluator supports up to six speakers")
    duration = max([float(x["end"]) for x in (reference + predicted)] or [0])
    frames = max(1, round(duration / step))
    ref_sets = []
    pred_sets = []
    for i in range(frames):
        t = (i + 0.5) * step
        ref_sets.append({str(x["speaker"]) for x in reference if x["start"] <= t < x["end"]})
        pred_sets.append({str(x["speaker"]) for x in predicted if x["start"] <= t < x["end"]})
    target = sum(len(x) for x in ref_sets)
    if not target:
        return None
    best = float("inf")
    # Partial injection of predicted speaker IDs onto reference IDs.
    candidates = ids_ref + [f"EXTRA_{i}" for i in range(max(0, len(ids_pred) - len(ids_ref)))]
    for remap_to in permutations(candidates, len(ids_pred)):
        mapping = dict(zip(ids_pred, remap_to))
        penalty = 0
        for a, b in zip(ref_sets, pred_sets):
            c = {mapping[x] for x in b}
            penalty += max(len(a - c), len(c - a))
        best = min(best, penalty)
    return best / target


def word_alignment_mae(reference: list[dict], predicted: list[dict]) -> float | None:
    """Mean absolute time-boundary error for words with known reference times."""
    errors = []
    j = 0
    for target in reference:
        label = normalize(target["word"])
        for i in range(j, len(predicted)):
            if normalize(predicted[i].get("word", "")) == label:
                if "start" in predicted[i] and "end" in predicted[i]:
                    errors.extend([abs(predicted[i]["start"] - target["start"]),
                                   abs(predicted[i]["end"] - target["end"])])
                j = i + 1
                break
    return sum(errors) / len(errors) if errors else None
