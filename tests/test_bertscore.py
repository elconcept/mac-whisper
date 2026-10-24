"""Mocked BERTScore batch tests; full multilingual transformer runs in Actions."""
import json
from scripts.evaluate_bertscore import compute, evaluate, MODEL_TYPE


class Values:
    def __init__(self, values):
        self.values = values
    def tolist(self):
        return self.values


def fake_scorer(cands, refs, **kwargs):
    assert kwargs["model_type"] == MODEL_TYPE
    assert kwargs["lang"] == "pl"
    assert kwargs["rescale_with_baseline"] is False
    assert all(cands)
    assert len(cands) == len(refs)
    return (Values([.91] * len(cands)),
            Values([.83] * len(cands)),
            Values([.869] * len(cands)))


def test_bertscore_roles_and_missing_utterances():
    pairs = {
        "whisper": [
            {"speaker": "SPEAKER_00", "time": 21, "reference": "Zdążyłem",
             "hypothesis": "Zdążyłem"},
            {"speaker": "SPEAKER_01", "time": 23, "reference": "Niech pan położy",
             "hypothesis": ""},
        ],
        "parakeet": [
            {"speaker": "SPEAKER_00", "time": 21, "reference": "Zdążyłem",
             "hypothesis": "Zdążyłem"},
            {"speaker": "SPEAKER_01", "time": 23, "reference": "Niech pan położy",
             "hypothesis": "Niech pan położy"},
        ],
    }
    data = compute(pairs, scorer=fake_scorer)
    assert data["whisper"]["missing"] == 1
    assert data["whisper"]["roles"]["SPEAKER_01"]["f1"] == 0
    assert data["whisper"]["roles"]["SPEAKER_00"]["f1"] == .869
    assert data["whisper"]["f1"] < data["parakeet"]["f1"]


def test_bertscore_reference_and_output_are_separate(tmp_path):
    main = tmp_path / "sample-pl.primary.reference.md"
    main.write_text(
        "<!-- REFERENCE_STATUS: VERIFIED -->\n"
        "[00:21] **TK**: Zdążyłem.\n"
        "[00:23] **URZĘDNICZKA SĄDOWA**: Niech pan położy.\n",
        encoding="utf-8",
    )
    legacy = tmp_path / "sample-pl.reference.md"
    legacy.write_text(
        "<!-- REFERENCE_STATUS: VERIFIED -->\n"
        "[00:21] **SPEAKER_00**: Zdążyłem.\n"
        "[00:23] **SPEAKER_01**: Niech pan położy.\n",
        encoding="utf-8",
    )
    hypotheses = tmp_path / "hypotheses"
    hypotheses.mkdir()
    (hypotheses / "model.json").write_text(
        json.dumps({"segments": [{"words": [
            {"start": 21.0, "end": 21.4, "word": "Zdążyłem."},
            {"start": 23.0, "end": 23.4, "word": "Niech"},
            {"start": 23.4, "end": 23.6, "word": "pan"},
            {"start": 23.7, "end": 24.0, "word": "położy."},
        ]}]}), encoding="utf-8",
    )
    out = tmp_path / "bertscore.md"
    result = evaluate(main, legacy, hypotheses, out, scorer=fake_scorer)
    assert result["model"]["f1"] > .86
    assert "BERTScore" in out.read_text(encoding="utf-8")
    assert "Urzędniczka" in out.read_text(encoding="utf-8")
    assert "Zdążyłem." in main.read_text(encoding="utf-8")
