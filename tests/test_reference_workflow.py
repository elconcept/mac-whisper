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
    assert "| mock | 0.00% | 0.00% | — |" in report.read_text(encoding="utf-8")
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
    assert "| candidate | 0.00% | 0.00% | 0.00% |" in report.read_text(encoding="utf-8")
