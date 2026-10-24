"""Reference versioning and metric reproducibility tests."""
from __future__ import annotations
import json
from pathlib import Path
from scripts.archive_e2e_artifacts import archive
from scripts.evaluate_reference import evaluate, parse_reference

def test_draft_reference_does_not_score(tmp_path):
    doc = tmp_path / "reference.md"
    doc.write_text(
        "# DRAFT\n<!-- REFERENCE_STATUS: DRAFT -->\n[00:01] **SPEAKER_00**: Dzień dobry.\n",
        encoding="utf-8",
    )
    assert parse_reference(doc) is None
    assert not evaluate(doc, tmp_path, tmp_path / "quality.md")
    assert not (tmp_path / "quality.md").exists()

def test_human_verified_reference_scores_models(tmp_path):
    doc = tmp_path / "reference.md"
    doc.write_text(
        "<!-- REFERENCE_STATUS: VERIFIED -->\n"
        "[00:00] **SPEAKER_00**: Dzień dobry.\n"
        "[00:03] **SPEAKER_01**: Jak się masz? **[NAKŁADAJĄCE SIĘ GŁOSY — SPRAWDŹ]**\n"
        "[00:05] **SPEAKER_02**: Dobrze dziękuję.\n",
        encoding="utf-8",
    )
    hyps = tmp_path / "hypotheses"
    hyps.mkdir()
    (hyps / "mock.json").write_text(json.dumps({"segments": [
        {"text": "Dzień dobry."}, {"text": "Jak się masz?"}, {"text": "Dobrze dziękuję."}
    ]}), encoding="utf-8")
    report = tmp_path / "score.md"
    assert evaluate(doc, hyps, report)
    content = report.read_text(encoding="utf-8")
    assert "| mock | — | — | — | — | — | 0.00% | — |" in content
    assert "| mock | 0.00% | 0.00% |" in content
    assert (tmp_path / "sample-pl.primary.reference.md").exists()
    assert "not calculated" in report.read_text(encoding="utf-8")

def test_artifact_archive_preserves_existing_human_reference(tmp_path):
    downloaded = tmp_path / "download" / "model-x"
    transcript_dir = downloaded / "model-x"
    transcript_dir.mkdir(parents=True)
    (downloaded / "model-x.report.json").write_text(
        json.dumps({"status": "passed", "mode": "model-x", "observed_speaker_count": 3,
                    "segments": 1, "elapsed_seconds": 2.0}), encoding="utf-8",
    )
    (transcript_dir / "sample-pl.json").write_text(
        json.dumps({"segments": [{"start": 0, "end": 1, "text": "Cześć"}]}), encoding="utf-8"
    )
    output = tmp_path / "output"
    output.mkdir()
    reference = output / "sample-pl.reference.md"
    reference.write_text("MANUAL CORRECTION", encoding="utf-8")
    archive(tmp_path / "download", output / "hypotheses", 123)
    assert (output / "hypotheses/model-x.json").exists()
    assert reference.read_text(encoding="utf-8") == "MANUAL CORRECTION"


def test_verified_manual_speaker_intervals_enable_approx_der(tmp_path):
    ref = tmp_path / "sample-pl.reference.md"
    ref.write_text(
        "<!-- REFERENCE_STATUS: VERIFIED -->\n"
        "[00:00] **SPEAKER_00**: Dzień dobry.\n"
        "[00:01] **SPEAKER_01**: Tak, dziękuję.\n"
        "[00:02] **SPEAKER_00**: Do widzenia.\n",
        encoding="utf-8",
    )
    (tmp_path / "sample-pl.speakers.reference.json").write_text(
        json.dumps({
            "verified": True,
            "intervals": [
                {"start": 0, "end": 1, "speaker": "HUMAN_A"},
                {"start": 1, "end": 2, "speaker": "HUMAN_B"},
                {"start": 2, "end": 3, "speaker": "HUMAN_A"},
            ],
        }), encoding="utf-8",
    )
    hyps = tmp_path / "hypotheses"
    hyps.mkdir()
    (hyps / "candidate.json").write_text(
        json.dumps({
            "segments": [
                {"start": 0, "end": 1, "text": "Dzień dobry.", "speaker": "SPEAKER_01"},
                {"start": 1, "end": 2, "text": "Tak, dziękuję.", "speaker": "SPEAKER_00"},
                {"start": 2, "end": 3, "text": "Do widzenia.", "speaker": "SPEAKER_01"},
            ]
        }), encoding="utf-8",
    )
    report = tmp_path / "quality.md"
    assert evaluate(ref, hyps, report)
    content = report.read_text(encoding="utf-8")
    assert "| candidate | — | — | — | — | — | 0.00% | 0.00% |" in content



def test_actual_verified_format_without_colon_is_not_lost(tmp_path):
    ref = tmp_path / "sample-pl.reference.md"
    ref.write_text(
        "<!-- REFERENCE_STATUS: VERIFIED -->\n"
        "[00:42] **SPEAKER_02**: Szum tła.\n"
        "[00:43] **SPEAKER_01** Drugi wydział.\n"
        "[00:45] **SPEAKER_00**: Mam pytanie.\n",
        encoding="utf-8",
    )
    turns = parse_reference(ref)
    assert len(turns) == 3
    assert turns[1]["speaker"] == "SPEAKER_01"
    assert turns[1]["text"] == "Drugi wydział."


def test_primary_metric_ignores_bystander_and_speaker_id_permutation():
    from scripts.primary_quality import score_primary, primary_view
    turns = [
        {"start_s_rounded": 0, "speaker": "SPEAKER_00", "text": "Ala ma kota"},
        {"start_s_rounded": 2, "speaker": "SPEAKER_02", "text": "Poboczne zakłócenie"},
        {"start_s_rounded": 4, "speaker": "SPEAKER_01", "text": "Tak to prawda"},
        {"start_s_rounded": 6, "speaker": "SPEAKER_02", "text": "Kolejny szum"},
    ]
    result = {"segments": [
        {"speaker": "MODEL_OTHER", "words": [
            {"start": 0.1, "end": 0.3, "word": "Ala"},
            {"start": 0.4, "end": 0.7, "word": "ma"},
            {"start": 0.8, "end": 1.1, "word": "kota"},
            {"start": 2.2, "end": 2.5, "word": "hałas"},
        ]},
        {"speaker": "MODEL_WRONG", "words": [
            {"start": 4.1, "end": 4.3, "word": "Tak"},
            {"start": 4.4, "end": 4.7, "word": "to"},
            {"start": 4.8, "end": 5.1, "word": "prawda"},
            {"start": 6.1, "end": 6.3, "word": "zaskoczenie"},
        ]},
    ]}
    quality = score_primary(turns, result)
    assert quality is not None
    assert quality["focus_wer"] == 0
    assert quality["focus_cer"] == 0
    assert quality["macro_wer"] == 0
    assert quality["scored_hypothesis_words"] == 6
    view = primary_view(turns)
    assert "**SPEAKER_00**" in view
    assert "**SPEAKER_01**" in view
    assert "**SPEAKER_02**" not in view
    assert "Poboczne zakłócenie" not in view


def test_same_second_different_speakers_is_excluded_not_invented():
    from scripts.primary_quality import focus_windows, primary_view
    turns = [
        {"start_s_rounded": 10, "speaker": "SPEAKER_00", "text": "Pytanie"},
        {"start_s_rounded": 10, "speaker": "SPEAKER_02", "text": "Przeszkadza"},
        {"start_s_rounded": 12, "speaker": "SPEAKER_01", "text": "Odpowiedź"},
    ]
    windows, meta = focus_windows(turns)
    assert [w["source_second"] for w in windows] == [12]
    assert meta["ambiguous_timestamp_seconds"] == [10]
    assert meta["coverage"] == .5
    assert "POMINIĘTE W METRYCE" in primary_view(turns)


def test_primary_macro_scores_both_main_people_equally():
    from scripts.primary_quality import score_primary
    turns = [
        {"start_s_rounded": 0, "speaker": "SPEAKER_00", "text": "Tak"},
        {"start_s_rounded": 2, "speaker": "SPEAKER_01", "text": "To ważna długa odpowiedź"},
        {"start_s_rounded": 6, "speaker": "SPEAKER_02", "text": "Głos poboczny"},
    ]
    result = {"segments": [{"words": [
        {"start": .1, "end": .2, "word": "Tak"},
        {"start": 2.1, "end": 2.2, "word": "Nic"},
    ]}]}
    stats = score_primary(turns, result)
    assert stats["per_speaker"]["SPEAKER_00"]["wer"] == 0
    assert stats["per_speaker"]["SPEAKER_01"]["wer"] > 0
    assert stats["macro_wer"] == stats["per_speaker"]["SPEAKER_01"]["wer"] / 2


def test_primary_metric_requires_word_timestamps():
    from scripts.primary_quality import score_primary
    turns = [{"start_s_rounded": 0, "speaker": "SPEAKER_00", "text": "Test"}]
    assert score_primary(turns, {"segments": [{"text": "Test", "start": 0, "end": 1}]}) is None
