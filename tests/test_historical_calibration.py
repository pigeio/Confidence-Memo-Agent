import pytest
import os
import tempfile
from src.historical_calibration import (
    CalibrationRecord,
    InMemoryCalibrationStorage,
    JSONCalibrationStorage,
    HistoricalCalibrationEngine,
)


def test_calibration_record_serialization():
    rec = CalibrationRecord(
        proposal="Add CSV Export",
        predicted_score=85,
        predicted_confidence="High",
        predicted_recommendation="PROCEED_TO_BUILD",
        actual_outcome=1.0,
    )
    data = rec.to_dict()
    assert data["proposal"] == "Add CSV Export"
    assert data["predicted_score"] == 85
    assert data["actual_outcome"] == 1.0

    restored = CalibrationRecord.from_dict(data)
    assert restored.record_id == rec.record_id
    assert restored.proposal == rec.proposal
    assert restored.predicted_score == rec.predicted_score
    assert restored.actual_outcome == 1.0


def test_in_memory_storage():
    storage = InMemoryCalibrationStorage()
    assert storage.count() == 0

    rec = CalibrationRecord(
        proposal="Feature A",
        predicted_score=60,
        predicted_confidence="Moderate",
        predicted_recommendation="VALIDATE_FURTHER",
    )
    rec_id = storage.save_record(rec)
    assert storage.count() == 1
    assert storage.get_record(rec_id) == rec
    assert len(storage.get_all_records()) == 1

    with pytest.raises(TypeError):
        storage.save_record({"not": "a record"})

    storage.clear()
    assert storage.count() == 0
    assert storage.get_record(rec_id) is None


def test_json_calibration_storage():
    with tempfile.TemporaryDirectory() as tmp_dir:
        json_path = os.path.join(tmp_dir, "calibration_test.json")
        storage = JSONCalibrationStorage(file_path=json_path)

        assert storage.count() == 0

        rec = CalibrationRecord(
            proposal="Dark Theme",
            predicted_score=90,
            predicted_confidence="High",
            predicted_recommendation="PROCEED_TO_BUILD",
        )
        rec_id = storage.save_record(rec)
        assert storage.count() == 1

        # Re-instantiate storage to verify disk persistence
        storage2 = JSONCalibrationStorage(file_path=json_path)
        fetched = storage2.get_record(rec_id)
        assert fetched is not None
        assert fetched.proposal == "Dark Theme"
        assert fetched.predicted_score == 90

        storage2.clear()
        assert storage2.count() == 0


def test_calibration_engine_record_and_update():
    storage = InMemoryCalibrationStorage()
    engine = HistoricalCalibrationEngine(storage=storage)

    rec_id = engine.record_prediction(
        proposal="Auto-save drafts",
        predicted_score=80,
        predicted_confidence="High",
        predicted_recommendation="PROCEED_TO_BUILD",
    )

    rec = storage.get_record(rec_id)
    assert rec.actual_outcome is None

    # Update with ground-truth actual outcome
    updated = engine.record_actual_outcome(rec_id, actual_outcome=1.0, metadata_update={"reviewed_by": "PM"})
    assert updated.actual_outcome == 1.0
    assert updated.metadata["reviewed_by"] == "PM"

    # Non-existent ID error
    with pytest.raises(KeyError):
        engine.record_actual_outcome("non_existent_id", 1.0)

    # Invalid outcome value
    with pytest.raises(ValueError):
        engine.record_actual_outcome(rec_id, 2.5)


def test_calibration_metrics_brier_and_ece():
    storage = InMemoryCalibrationStorage()
    engine = HistoricalCalibrationEngine(storage=storage)

    # Empty metrics check
    empty_metrics = engine.calculate_calibration_metrics()
    assert empty_metrics["status"] == "INSUFFICIENT_DATA"
    assert empty_metrics["brier_score"] is None

    # Perfect calibration test scenario:
    # 2 predictions: score 100 with outcome 1.0, and score 0 with outcome 0.0
    id1 = engine.record_prediction("P1", 100, "High", "PROCEED_TO_BUILD")
    engine.record_actual_outcome(id1, 1.0)

    id2 = engine.record_prediction("P2", 0, "Low", "DEPRIORITIZE")
    engine.record_actual_outcome(id2, 0.0)

    metrics = engine.calculate_calibration_metrics(num_bins=2)
    assert metrics["evaluated_records"] == 2
    assert metrics["brier_score"] == 0.0
    assert metrics["expected_calibration_error"] == 0.0
    assert metrics["calibration_bias"] == 0.0
    assert metrics["overall_accuracy"] == 1.0
    assert metrics["status"] == "CALIBRATED"

    report = engine.generate_calibration_report()
    assert "Brier Score: 0.0000" in report["summary_text"]


def test_calibration_metrics_imperfect():
    storage = InMemoryCalibrationStorage()
    engine = HistoricalCalibrationEngine(storage=storage)

    # Add 4 records:
    # Rec 1: score 80 (0.8), outcome 1.0 -> (0.8-1)^2 = 0.04
    # Rec 2: score 80 (0.8), outcome 0.0 -> (0.8-0)^2 = 0.64
    # Rec 3: score 20 (0.2), outcome 0.0 -> (0.2-0)^2 = 0.04
    # Rec 4: score 20 (0.2), outcome 1.0 -> (0.2-1)^2 = 0.64
    # Mean Brier = (0.04 + 0.64 + 0.04 + 0.64)/4 = 0.34
    id1 = engine.record_prediction("P1", 80, "High", "PROCEED_TO_BUILD")
    engine.record_actual_outcome(id1, 1.0)

    id2 = engine.record_prediction("P2", 80, "High", "PROCEED_TO_BUILD")
    engine.record_actual_outcome(id2, 0.0)

    id3 = engine.record_prediction("P3", 20, "Low", "DEPRIORITIZE")
    engine.record_actual_outcome(id3, 0.0)

    id4 = engine.record_prediction("P4", 20, "Low", "DEPRIORITIZE")
    engine.record_actual_outcome(id4, 1.0)

    metrics = engine.calculate_calibration_metrics(num_bins=5)
    assert metrics["evaluated_records"] == 4
    assert metrics["brier_score"] == 0.34
    assert len(metrics["reliability_bins"]) == 5
    assert "High" in metrics["tier_metrics"]
    assert "Low" in metrics["tier_metrics"]
