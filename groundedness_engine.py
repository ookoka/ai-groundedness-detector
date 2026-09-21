"""
Groundedness engine for testing agent responses.

Integrates groundedness testing into the response pipeline.
Tests each answer for:
- Evidence Dependence (Test 3)
- Refusal Correctness (Test 5)
- Cross-Turn Retention (Test 6)
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from groundedness_poc_tester import (
    GroundednessPOCTester,
    TestResult,
)
from groundedness_monitoring import GroundednessVerdict, GroundednessMonitor

logger = logging.getLogger(__name__)


class GroundednessEngine:
    """
    Tests groundedness of agent responses.

    Usage:
        engine = GroundednessEngine()
        verdict = await engine.test_response(
            query="How often does promotion run?",
            answer="Based on evidence, monthly",
            evidence=[...],
            conversation_id="conv-123"
        )
        print(verdict.get_verdict_text())  # ✅ HIGH CONFIDENCE - ...
    """

    def __init__(
        self,
        evidence_provider: Callable[[str], list[dict]] = None,
        monitor: GroundednessMonitor = None,
    ):
        """
        Args:
            evidence_provider: Function to retrieve evidence (query -> list[dict])
            monitor: Optional GroundednessMonitor for tracking over time
        """
        self.evidence_provider = evidence_provider or self._default_evidence_provider
        self.monitor = monitor or GroundednessMonitor()

        # Create tester with a mock callable (will be set per test)
        self.tester = None
        self.conversation_history: dict[str, list[dict]] = {}

    @staticmethod
    def _default_evidence_provider(query: str) -> list[dict]:
        """Default evidence provider returns empty list."""
        return []

    def _create_answer_callable(self, answer: str, evidence: list[dict]) -> Callable:
        """Create a callable that returns the given answer (for testing)."""

        def answer_callable(prompt: str, evidence: list[dict] = None) -> str:
            if evidence is None or len(evidence) == 0:
                return "I don't have sufficient evidence to answer this question."
            return answer

        return answer_callable

    def _get_last_turn_claims(self, conversation_id: str) -> list[dict]:
        """Get claims from previous turn for Test 6."""
        if conversation_id not in self.conversation_history:
            return []

        history = self.conversation_history[conversation_id]
        if len(history) < 2:
            return []

        # Get last turn's answer
        last_turn = history[-1]
        return (
            [{"text": last_turn["answer"], "evidence_ids": last_turn.get("evidence_ids", [])}]
            if "answer" in last_turn
            else []
        )

    async def test_response(
        self,
        query: str,
        answer: str,
        evidence: list[dict],
        conversation_id: str = "default",
        alternative_evidence: list[dict] = None,
    ) -> GroundednessVerdict:
        """
        Test a response for groundedness.

        Args:
            query: Original user query
            answer: Agent's response
            evidence: Evidence used to generate response
            conversation_id: For tracking conversation history
            alternative_evidence: Alternative evidence for Test 3

        Returns:
            GroundednessVerdict with confidence level
        """
        try:
            # Initialize conversation history
            if conversation_id not in self.conversation_history:
                self.conversation_history[conversation_id] = []

            # Create tester for this specific answer
            answer_callable = self._create_answer_callable(answer, evidence)
            tester = GroundednessPOCTester(answer_callable)

            # Prepare alternative evidence
            if alternative_evidence is None:
                alternative_evidence = [
                    {
                        "id": "alt",
                        "text": "Alternative context that might change the answer",
                        "source": "alternative",
                    }
                ]

            # Get prior claims for cross-turn testing
            prior_claims = self._get_last_turn_claims(conversation_id)

            # Run tests
            logger.info(f"Testing groundedness for: {query[:50]}...")
            results = tester.run_full_test(
                prompt=query,
                evidence=evidence,
                alternative_evidence=alternative_evidence,
                required_source="evidence",
                prior_claims=prior_claims,
            )

            # Calculate verdict
            test_3_score = results["test_3_evidence_dependence"].score
            test_5_score = results["test_5_refusal"].score
            test_6_score = results["test_6_cross_turn"].score

            overall_score = (test_3_score + test_5_score + test_6_score) / 3.0
            all_passed = (
                results["test_3_evidence_dependence"].passed
                and results["test_5_refusal"].passed
                and results["test_6_cross_turn"].passed
            )

            # Create verdict
            test_results = {
                "test_3_evidence_dependence": {
                    "passed": results["test_3_evidence_dependence"].passed,
                    "score": round(test_3_score, 2),
                    "message": results["test_3_evidence_dependence"].message,
                },
                "test_5_refusal": {
                    "passed": results["test_5_refusal"].passed,
                    "score": round(test_5_score, 2),
                    "message": results["test_5_refusal"].message,
                },
                "test_6_cross_turn": {
                    "passed": results["test_6_cross_turn"].passed,
                    "score": round(test_6_score, 2),
                    "message": results["test_6_cross_turn"].message,
                },
            }

            verdict = GroundednessVerdict(
                score=overall_score,
                all_tests_passed=all_passed,
                test_results=test_results,
            )

            # Log verdict
            self.monitor.log_verdict(query, verdict)

            # Add to conversation history
            self.conversation_history[conversation_id].append(
                {
                    "query": query,
                    "answer": answer,
                    "evidence_ids": [e.get("id", "") for e in evidence],
                    "groundedness_score": overall_score,
                    "verdict": verdict.confidence_level,
                }
            )

            logger.info(
                f"Groundedness verdict: {verdict.confidence_level} "
                f"({overall_score:.1%}) for query: {query[:50]}..."
            )

            return verdict

        except Exception as e:
            logger.error(f"Error testing groundedness: {e}", exc_info=True)
            # Return low confidence on error
            return GroundednessVerdict(
                score=0.0,
                all_tests_passed=False,
                test_results={
                    "test_3_evidence_dependence": {
                        "passed": False,
                        "score": 0.0,
                        "message": f"Error: {str(e)}",
                    },
                    "test_5_refusal": {
                        "passed": False,
                        "score": 0.0,
                        "message": f"Error: {str(e)}",
                    },
                    "test_6_cross_turn": {
                        "passed": False,
                        "score": 0.0,
                        "message": f"Error: {str(e)}",
                    },
                },
                confidence_level="dangerous",
            )

    def get_monitoring_summary(self) -> str:
        """Get summary of groundedness monitoring."""
        return self.monitor.get_summary()

    def save_monitoring_data(self) -> None:
        """Save monitoring data to disk."""
        self.monitor.save_stats()
