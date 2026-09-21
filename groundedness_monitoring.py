"""
Groundedness monitoring and verdict tracking.

Tracks agent confidence/groundedness over time and provides verdicts.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class GroundednessVerdict:
    """Represents a groundedness verdict for a response."""

    def __init__(
        self,
        score: float,
        all_tests_passed: bool,
        test_results: dict[str, Any],
        confidence_level: str = "unknown",
    ):
        """
        Args:
            score: Groundedness score (0.0-1.0)
            all_tests_passed: Whether all tests passed
            test_results: Individual test results
            confidence_level: One of: dangerous, low, medium, high
        """
        self.score = score
        self.all_tests_passed = all_tests_passed
        self.test_results = test_results
        self.confidence_level = confidence_level or self._calculate_confidence()
        self.timestamp = datetime.now().isoformat()

    def _calculate_confidence(self) -> str:
        """Calculate confidence level from score."""
        if self.score >= 0.85:
            return "high"
        elif self.score >= 0.70:
            return "medium"
        elif self.score >= 0.50:
            return "low"
        else:
            return "dangerous"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "score": round(self.score, 2),
            "confidence_level": self.confidence_level,
            "all_tests_passed": self.all_tests_passed,
            "timestamp": self.timestamp,
            "tests": self.test_results,
        }

    def get_verdict_text(self) -> str:
        """Get human-readable verdict text."""
        if self.confidence_level == "high":
            return "✅ HIGH CONFIDENCE - Answer is well-grounded and trustworthy"
        elif self.confidence_level == "medium":
            return "⚠️ MEDIUM CONFIDENCE - Answer is reasonably grounded but not perfect"
        elif self.confidence_level == "low":
            return "⚠️ LOW CONFIDENCE - Answer may not be fully grounded in evidence"
        else:
            return "❌ DANGEROUS - Answer is not grounded in evidence. Do not trust."

    def get_failure_explanation(self) -> str:
        """Get explanation of what failed."""
        failures = []
        if not self.test_results["test_3_evidence_dependence"]["passed"]:
            failures.append("Answer doesn't depend on evidence (Test 3)")
        if not self.test_results["test_5_refusal"]["passed"]:
            failures.append("Agent doesn't refuse appropriately (Test 5)")
        if not self.test_results["test_6_cross_turn"]["passed"]:
            failures.append("Claims aren't consistent across turns (Test 6)")
        return " | ".join(failures) if failures else "No failures"


class GroundednessMonitor:
    """
    Monitors groundedness over time and logs verdicts.

    Tracks:
    - Query count
    - Average groundedness
    - Confidence distribution
    - Failed test patterns
    """

    def __init__(self, log_dir: str = "logs"):
        """
        Args:
            log_dir: Directory to store monitoring logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        self.verdicts_log = self.log_dir / "groundedness_verdicts.jsonl"
        self.stats_file = self.log_dir / "stats.json"

        self.queries_processed = 0
        self.verdicts: list[GroundednessVerdict] = []
        self._load_existing_logs()

    def _load_existing_logs(self) -> None:
        """Load existing verdicts from log file."""
        if self.verdicts_log.exists():
            try:
                with open(self.verdicts_log, "r") as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            # Reconstruct verdict (simplified)
                            self.queries_processed += 1
            except Exception as e:
                logger.warning(f"Failed to load existing verdicts: {e}")

    def log_verdict(self, query: str, verdict: GroundednessVerdict) -> None:
        """
        Log a verdict for a query.

        Args:
            query: The user's query
            verdict: The groundedness verdict
        """
        self.verdicts.append(verdict)
        self.queries_processed += 1

        # Append to JSONL log
        log_entry = {
            "query": query,
            "query_num": self.queries_processed,
            **verdict.to_dict(),
        }

        try:
            with open(self.verdicts_log, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to write verdict log: {e}")

    def get_statistics(self) -> dict[str, Any]:
        """Get monitoring statistics."""
        if not self.verdicts:
            return {
                "queries_processed": 0,
                "average_score": 0.0,
                "high_confidence": 0,
                "medium_confidence": 0,
                "low_confidence": 0,
                "dangerous_confidence": 0,
            }

        scores = [v.score for v in self.verdicts]
        confidence_levels = [v.confidence_level for v in self.verdicts]

        return {
            "queries_processed": self.queries_processed,
            "average_score": round(sum(scores) / len(scores), 2),
            "min_score": round(min(scores), 2),
            "max_score": round(max(scores), 2),
            "high_confidence": confidence_levels.count("high"),
            "medium_confidence": confidence_levels.count("medium"),
            "low_confidence": confidence_levels.count("low"),
            "dangerous_confidence": confidence_levels.count("dangerous"),
        }

    def get_summary(self) -> str:
        """Get human-readable summary of statistics."""
        stats = self.get_statistics()

        if stats["queries_processed"] == 0:
            return "No queries processed yet."

        return f"""
Groundedness Monitoring Summary:
- Processed: {stats['queries_processed']} queries
- Average Score: {stats['average_score']:.1%}
- Score Range: {stats['min_score']:.1%} - {stats['max_score']:.1%}
- High Confidence: {stats['high_confidence']} ({stats['high_confidence']/stats['queries_processed']*100:.0f}%)
- Medium Confidence: {stats['medium_confidence']} ({stats['medium_confidence']/stats['queries_processed']*100:.0f}%)
- Low Confidence: {stats['low_confidence']} ({stats['low_confidence']/stats['queries_processed']*100:.0f}%)
- Dangerous: {stats['dangerous_confidence']} ({stats['dangerous_confidence']/stats['queries_processed']*100:.0f}%)
        """

    def save_stats(self) -> None:
        """Save statistics to file."""
        try:
            with open(self.stats_file, "w") as f:
                json.dump(self.get_statistics(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save stats: {e}")
