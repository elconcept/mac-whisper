"""Offline tests: continuous embeddings, critical facts and curated speakers.

Semantic model download / inference is a separate *real* CI job. These tests
use a deterministic fake embedding encoder and require no ML packages.
"""
import json
from pathlib import Path

from scripts.evaluate_semantic import (
    EMBEDDING_MODELS, critical_warnings, cosine, compute_scores,
    reference_pairs, report_markdown, evaluate,
)


class FakeEncoder:
    def encode(self, strings, **kwargs):
        result = []
        for value in strings:
            text = value.removeprefix("query: ").removeprefix("passage: ")
            if text == "Dzień dobry":
                result.append([1.0, 0.0, 0.0])
            elif text == "Pytanie do sądu":
                result.append([0.0, 1.0, 0.0])
            elif text == "Głos z daleka":
                result.append([0.0, 0.0, 1.0])
            else:
                result.append([0.5, 0.5, 0.5])
        return result


def test_continuous_cosine_is_not_binary():
    assert cosine([1, 0], [1, 0]) == 1.0
    assert cosine([1, 0], [0, 1]) == 0.0
    assert 0 < cosine([1, 1], [1, 0]) < 1


def test_negation_number_and_room_flags_override_embedding_confidence():
    assert "NEGACJA" in critical_warnings(
        "Dowodu nie ma w aktach", "Dowód jest w aktach"
    )
    assert "NUMERY" in critical_warnings(
        "Telefon 73-39, pokój L-419", "Telefon 73-19, pokój L-419"
    )
    assert critical_warnings("Idzie pan na wydział", "Idzie pan na wydział") == []


def test_both_vector_encoders_use_documented_prefixes():
    class Capture(FakeEncoder):
        def encode(self, strings, **kwargs):
            self.inputs = strings
            return super().encode(strings, **kwargs)
    pairs = [
        {"time": 1, "speaker": "SPEAKER_00",
         "reference": "Pytanie do sądu", "hypothesis": "Pytanie do sądu"},
        {"time": 3, "speaker": "SPEAKER_01",
         "reference": "Dzień dobry", "hypothesis": "Dzień dobry"},
    ]
    for model in EMBEDDING_MODELS:
        encoder = Capture()
        scores = compute_scores(pairs, encoder, model)
        assert scores["cosine_macro"] == 1.0
        assert scores["roles"]["SPEAKER_00"] == 1.0
        assert scores["roles"]["SPEAKER_01"] == 1.0
        assert len(encoder.inputs) == (8 if "mmlw" in model else 4)
        assert encoder.inputs[0].startswith("query: ")
        assert ("passage: " in " ".join(encoder.inputs)) == ("mmlw" in model)


def test_time_windows_exclude_third_speaker_without_model_ids():
    curated = [
        {"start_s_rounded": 21, "speaker": "SPEAKER_01", "text": "Dzień dobry"},
        {"start_s_rounded": 25, "speaker": "SPEAKER_00", "text": "Pytanie do sądu"},
    ]
    context = curated + [
        {"start_s_rounded": 23, "speaker": "SPEAKER_02", "text": "Głos z daleka"}
    ]
    model = {"segments": [{"speaker": "WRONG_ID", "words": [
        {"word": "Dzień", "start": 21.1, "end": 21.4},
        {"word": "dobry", "start": 21.45, "end": 21.8},
        {"word": "Głos", "start": 23.1, "end": 23.4},
        {"word": "z", "start": 23.45, "end": 23.5},
        {"word": "daleka", "start": 23.55, "end": 23.8},
        {"word": "Pytanie", "start": 25.1, "end": 25.4},
        {"word": "do", "start": 25.5, "end": 25.7},
        {"word": "sądu", "start": 25.8, "end": 26.0},
    ]}]}
    pairs = reference_pairs(curated, model, context)
    assert [p["speaker"] for p in pairs] == ["SPEAKER_01", "SPEAKER_00"]
    assert pairs[0]["hypothesis"] == "Dzień dobry"
    assert pairs[1]["hypothesis"] == "Pytanie do sądu"
    assert all("daleka" not in p["hypothesis"] for p in pairs)


def test_score_end_to_end_with_injected_vectors(tmp_path):
    primary = tmp_path / "sample-pl.primary.reference.md"
    primary.write_text(
        "<!-- REFERENCE_STATUS: VERIFIED -->\n"
        "[00:01] **TK**: Pytanie do sądu\n"
        "[00:03] **URZĘDNICZKA SĄDOWA**: Dzień dobry\n",
        encoding="utf-8",
    )
    legacy = tmp_path / "sample-pl.reference.md"
    legacy.write_text(
        "<!-- REFERENCE_STATUS: VERIFIED -->\n"
        "[00:01] **SPEAKER_00**: Pytanie do sądu\n"
        "[00:03] **SPEAKER_01**: Dzień dobry\n"
        "[00:05] **SPEAKER_02**: Głos z daleka\n",
        encoding="utf-8",
    )
    folder = tmp_path / "hypotheses"
    folder.mkdir()
    (folder / "model.json").write_text(json.dumps({
        "segments": [{"words": [
            {"word": "Pytanie", "start": 1.2, "end": 1.4},
            {"word": "do", "start": 1.5, "end": 1.6},
            {"word": "sądu", "start": 1.7, "end": 1.8},
            {"word": "Dzień", "start": 3.2, "end": 3.4},
            {"word": "dobry", "start": 3.5, "end": 3.8},
        ]}]
    }), encoding="utf-8")
    output = tmp_path / "semantic.md"
    fake = {model: FakeEncoder() for model in EMBEDDING_MODELS}
    assert evaluate(primary, legacy, folder, output,
                    models=EMBEDDING_MODELS, encoders=fake)
    text = output.read_text(encoding="utf-8")
    assert "1.0000" in text
    assert all(model in text for model in EMBEDDING_MODELS)
    assert "Nie ustalamy arbitralnego progu" in text
    assert primary.read_text(encoding="utf-8").startswith(
        "<!-- REFERENCE_STATUS: VERIFIED -->"
    )
