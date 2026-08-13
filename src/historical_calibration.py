import os
import json
import uuid
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CalibrationRecord:
    """Represents a single feature proposal prediction and its eventual real-world outcome."""

    proposal: str
    predicted_score: int  # 0 to 100
    predicted_confidence: str  # 'Low', 'Moderate', 'High'
    predicted_recommendation: str  # e.g., 'PROCEED_TO_BUILD', 'VALIDATE_FURTHER', etc.
    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    actual_outcome: float | None = None  # 1.0 = positive/validated, 0.0 = negative/failed, None = pending
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "CalibrationRecord":
        return cls(
            record_id=data.get("record_id", str(uuid.uuid4())),
            proposal=data["proposal"],
            predicted_score=int(data["predicted_score"]),
            predicted_confidence=data["predicted_confidence"],
            predicted_recommendation=data["predicted_recommendation"],
            actual_outcome=(
                float(data["actual_outcome"])
                if data.get("actual_outcome") is not None
                else None
            ),
            timestamp=data.get(
                "timestamp", datetime.now(timezone.utc).isoformat()
            ),
            metadata=data.get("metadata", {}),
        )


class BaseCalibrationStorage(ABC):
    """Abstract interface for storing and retrieving historical calibration records."""

    @abstractmethod
    def save_record(self, record: CalibrationRecord) -> str:
        """Save or update a calibration record. Returns record_id."""
        pass

    @abstractmethod
    def get_record(self, record_id: str) -> CalibrationRecord | None:
        """Retrieve a specific record by its ID."""
        pass

    @abstractmethod
    def get_all_records(self) -> list[CalibrationRecord]:
        """Retrieve all stored records."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all stored records."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Return total count of stored records."""
        pass


class InMemoryCalibrationStorage(BaseCalibrationStorage):
    """High-speed in-memory implementation of BaseCalibrationStorage."""

    def __init__(self):
        self._records: dict[str, CalibrationRecord] = {}

    def save_record(self, record: CalibrationRecord) -> str:
        if not isinstance(record, CalibrationRecord):
            raise TypeError("record must be an instance of CalibrationRecord")
        self._records[record.record_id] = record
        return record.record_id

    def get_record(self, record_id: str) -> CalibrationRecord | None:
        return self._records.get(record_id)

    def get_all_records(self) -> list[CalibrationRecord]:
        return list(self._records.values())

    def clear(self) -> None:
        self._records.clear()

    def count(self) -> int:
        return len(self._records)


class JSONCalibrationStorage(BaseCalibrationStorage):
    """File-backed JSON calibration storage for local persistence."""

    def __init__(self, file_path: str = "data/calibration_history.json"):
        self.file_path = file_path
        self._ensure_file_exists()

    def _ensure_file_exists(self) -> None:
        directory = os.path.dirname(self.file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        if not os.path.exists(self.file_path):
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump([], f)

    def _read_data(self) -> dict[str, dict]:
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                records_list = json.load(f)
                return {item["record_id"]: item for item in records_list}
        except (json.JSONDecodeError, FileNotFoundError):
            return {}

    def _write_data(self, records_dict: dict[str, dict]) -> None:
        temp_path = f"{self.file_path}.tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(list(records_dict.values()), f, indent=2)
        os.replace(temp_path, self.file_path)

    def save_record(self, record: CalibrationRecord) -> str:
        if not isinstance(record, CalibrationRecord):
            raise TypeError("record must be an instance of CalibrationRecord")
        data = self._read_data()
        data[record.record_id] = record.to_dict()
        self._write_data(data)
        return record.record_id

    def get_record(self, record_id: str) -> CalibrationRecord | None:
        data = self._read_data()
        item = data.get(record_id)
        return CalibrationRecord.from_dict(item) if item else None

    def get_all_records(self) -> list[CalibrationRecord]:
        data = self._read_data()
        return [CalibrationRecord.from_dict(item) for item in data.values()]

    def clear(self) -> None:
        self._write_data({})

    def count(self) -> int:
        return len(self._read_data())


class HistoricalCalibrationEngine:
    """
    Deterministic Historical Calibration Engine.
    Computes statistical calibration metrics including Brier Score, Expected Calibration Error (ECE),
    binned reliability tables, accuracy across confidence tiers, and calibration bias.
    """

    def __init__(self, storage: BaseCalibrationStorage | None = None):
        self.storage = storage or InMemoryCalibrationStorage()

    def record_prediction(
        self,
        proposal: str,
        predicted_score: int,
        predicted_confidence: str,
        predicted_recommendation: str,
        metadata: dict | None = None,
    ) -> str:
        """
        Record a new prediction before actual outcome is known.

        Returns:
            str: Unique record_id.
        """
        record = CalibrationRecord(
            proposal=proposal,
            predicted_score=predicted_score,
            predicted_confidence=predicted_confidence,
            predicted_recommendation=predicted_recommendation,
            metadata=metadata or {},
        )
        return self.storage.save_record(record)

    def record_actual_outcome(
        self,
        record_id: str,
        actual_outcome: float | bool | int,
        metadata_update: dict | None = None,
    ) -> CalibrationRecord:
        """
        Record or update the ground-truth actual outcome (1.0 for success/true, 0.0 for failure/false).

        Returns:
            CalibrationRecord: The updated record.
        """
        record = self.storage.get_record(record_id)
        if not record:
            raise KeyError(f"Calibration record not found for ID: {record_id}")

        val = float(actual_outcome)
        if not (0.0 <= val <= 1.0):
            raise ValueError("actual_outcome must be a number or boolean between 0.0 and 1.0")

        record.actual_outcome = val
        if metadata_update:
            record.metadata.update(metadata_update)

        self.storage.save_record(record)
        return record

    def calculate_calibration_metrics(
        self,
        records: list[CalibrationRecord] | None = None,
        num_bins: int = 5,
    ) -> dict:
        """
        Calculate Brier score, ECE, tier accuracy, and reliability curve bins.

        Parameters:
            records (list[CalibrationRecord]): Optional subset of records to evaluate.
            num_bins (int): Number of bins for ECE and reliability diagrams (default 5).

        Returns:
            dict: Comprehensive calibration metrics and telemetry.
        """
        if records is None:
            records = self.storage.get_all_records()

        # Filter records that have actual outcomes
        evaluated_records = [r for r in records if r.actual_outcome is not None]
        total_evaluated = len(evaluated_records)

        if total_evaluated == 0:
            return {
                "total_records": len(records),
                "evaluated_records": 0,
                "pending_records": len(records),
                "brier_score": None,
                "expected_calibration_error": None,
                "calibration_bias": None,
                "overall_accuracy": None,
                "tier_metrics": {},
                "reliability_bins": [],
                "status": "INSUFFICIENT_DATA",
            }

        confidences = np.array(
            [r.predicted_score / 100.0 for r in evaluated_records], dtype=np.float64
        )
        outcomes = np.array(
            [r.actual_outcome for r in evaluated_records], dtype=np.float64
        )

        # 1. Brier Score: Mean Squared Error between predicted probability and actual outcome
        brier_score = float(np.mean((confidences - outcomes) ** 2))

        # 2. Overall Accuracy (assuming threshold 0.5 for predicted success)
        binary_predictions = (confidences >= 0.5).astype(float)
        binary_outcomes = (outcomes >= 0.5).astype(float)
        overall_accuracy = float(np.mean(binary_predictions == binary_outcomes))

        # 3. Calibration Bias: Mean(Confidence) - Mean(Outcome)
        # Positive = overconfident, Negative = underconfident
        calibration_bias = float(np.mean(confidences) - np.mean(outcomes))

        # 4. Expected Calibration Error (ECE) and Reliability Bins
        bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
        reliability_bins = []
        ece = 0.0

        for i in range(num_bins):
            bin_low = bin_boundaries[i]
            bin_high = bin_boundaries[i + 1]

            if i == num_bins - 1:
                in_bin = (confidences >= bin_low) & (confidences <= bin_high)
            else:
                in_bin = (confidences >= bin_low) & (confidences < bin_high)

            bin_count = int(np.sum(in_bin))
            if bin_count > 0:
                avg_conf = float(np.mean(confidences[in_bin]))
                avg_acc = float(np.mean(outcomes[in_bin]))
                gap = abs(avg_acc - avg_conf)
                ece += (bin_count / total_evaluated) * gap

                reliability_bins.append(
                    {
                        "bin_index": i,
                        "bin_range": f"{bin_low:.2f} - {bin_high:.2f}",
                        "count": bin_count,
                        "avg_confidence": round(avg_conf, 3),
                        "empirical_accuracy": round(avg_acc, 3),
                        "calibration_gap": round(gap, 3),
                    }
                )
            else:
                reliability_bins.append(
                    {
                        "bin_index": i,
                        "bin_range": f"{bin_low:.2f} - {bin_high:.2f}",
                        "count": 0,
                        "avg_confidence": 0.0,
                        "empirical_accuracy": 0.0,
                        "calibration_gap": 0.0,
                    }
                )

        # 5. Tier-based Breakdown (Low, Moderate, High)
        tier_metrics = {}
        for tier in ["Low", "Moderate", "High"]:
            tier_recs = [r for r in evaluated_records if r.predicted_confidence == tier]
            if tier_recs:
                t_confs = np.array([r.predicted_score / 100.0 for r in tier_recs])
                t_outs = np.array([r.actual_outcome for r in tier_recs])
                t_bin_preds = (t_confs >= 0.5).astype(float)
                t_bin_outs = (t_outs >= 0.5).astype(float)
                tier_metrics[tier] = {
                    "count": len(tier_recs),
                    "avg_score": round(float(np.mean(t_confs * 100)), 1),
                    "empirical_success_rate": round(float(np.mean(t_outs)), 3),
                    "accuracy": round(float(np.mean(t_bin_preds == t_bin_outs)), 3),
                }
            else:
                tier_metrics[tier] = {
                    "count": 0,
                    "avg_score": 0.0,
                    "empirical_success_rate": 0.0,
                    "accuracy": 0.0,
                }

        return {
            "total_records": len(records),
            "evaluated_records": total_evaluated,
            "pending_records": len(records) - total_evaluated,
            "brier_score": round(brier_score, 4),
            "expected_calibration_error": round(float(ece), 4),
            "calibration_bias": round(calibration_bias, 4),
            "overall_accuracy": round(overall_accuracy, 3),
            "tier_metrics": tier_metrics,
            "reliability_bins": reliability_bins,
            "status": "CALIBRATED" if ece < 0.15 else "CALIBRATION_DRIFT",
        }

    def generate_calibration_report(
        self, records: list[CalibrationRecord] | None = None
    ) -> dict:
        """Generate a complete structured report with human-readable diagnostic analysis."""
        metrics = self.calculate_calibration_metrics(records=records)

        if metrics["evaluated_records"] == 0:
            summary = "Historical Calibration: No evaluated records with ground-truth outcomes."
        else:
            brier = metrics["brier_score"]
            ece = metrics["expected_calibration_error"]
            bias = metrics["calibration_bias"]
            bias_desc = (
                "Well-calibrated"
                if abs(bias) < 0.05
                else ("Overconfident" if bias > 0 else "Underconfident")
            )
            summary = (
                f"Historical Calibration Report (N={metrics['evaluated_records']}):\n"
                f"- Brier Score: {brier:.4f} (Optimal: 0.0)\n"
                f"- Expected Calibration Error (ECE): {ece:.4f}\n"
                f"- Calibration Tendency: {bias_desc} (Bias: {bias:+.4f})\n"
                f"- Overall Accuracy: {metrics['overall_accuracy'] * 100:.1f}%\n"
                f"- Status: {metrics['status']}"
            )

        metrics["summary_text"] = summary
        return metrics
