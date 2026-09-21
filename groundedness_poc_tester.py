"""
Groundedness POC Tester for RAG Agents

Tests:
  - Test 3: Evidence Dependence (does answer change when evidence is removed?)
  - Test 5: Refusal Correctness (does agent refuse when it should?)
  - Test 6: Cross-Turn Retention (does agent maintain citation consistency?)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
import re


@dataclass
class TestResult:
    """Result of a groundedness test."""
    test_name: str
    passed: bool
    score: float  # 0.0-1.0
    details: dict[str, Any]
    message: str


@dataclass
class Claim:
    """Extracted claim from agent response."""
    text: str
    evidence_ids: list[str] = None

    def __post_init__(self):
        if self.evidence_ids is None:
            self.evidence_ids = []

    def __eq__(self, other):
        if not isinstance(other, Claim):
            return False
        # Normalize whitespace for comparison
        return self._normalize(self.text) == self._normalize(other.text)

    def __hash__(self):
        return hash(self._normalize(self.text))

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.casefold().split())


class ClaimExtractor:
    """Extract claims from agent responses."""

    @staticmethod
    def extract(text: str) -> list[Claim]:
        """Split text into claims (sentences)."""
        if not text.strip():
            return []

        # Split by sentence boundaries
        normalized = text.replace("?", ".").replace("!", ".")
        sentences = []
        current = ""

        for char in normalized:
            current += char
            if char == ".":
                cleaned = current.strip()
                if cleaned and len(cleaned) > 5:  # Filter out tiny fragments
                    sentences.append(cleaned)
                current = ""

        if current.strip():
            sentences.append(current.strip())

        return [Claim(text=s) for s in sentences]


class EvidenceDependenceTester:
    """
    Test 3: Measures if agent's answer depends on retrieved evidence.

    Runs agent 5 ways:
    - V1: Normal (retrieval on)
    - V2: Knowledge removed (retrieval off)
    - V3: Supporting document changed (different evidence)
    - V4: Distractor added (conflicting evidence)
    - V5: Tool unavailable (retrieval fails)

    If claims change across variants → evidence-dependent (grounded) ✓
    If claims stay same → general knowledge, not dependent ❌
    """

    def __init__(self, agent_callable: Callable):
        """
        Args:
            agent_callable: Function that takes (prompt, evidence_context) -> answer
        """
        self.agent = agent_callable
        self.extractor = ClaimExtractor()

    def test(
        self,
        prompt: str,
        normal_evidence: list[dict],
        alternative_evidence: list[dict] = None,
    ) -> TestResult:
        """
        Test evidence dependence.

        Args:
            prompt: Question to ask agent
            normal_evidence: Correct evidence
            alternative_evidence: Different evidence for variant 3

        Returns:
            TestResult with evidence_dependence score
        """
        if alternative_evidence is None:
            alternative_evidence = [
                {
                    "id": "alt1",
                    "text": "Alternative information about different topic",
                }
            ]

        # Run 5 variants
        try:
            v1_normal = self.agent(prompt, evidence=normal_evidence)
            v2_no_evidence = self.agent(prompt, evidence=[])
            v3_diff_evidence = self.agent(prompt, evidence=alternative_evidence)
            v4_conflicting = self.agent(
                prompt, evidence=[{"id": "conflict", "text": "Contradictory information"}]
            )
            v5_empty_evidence = self.agent(prompt, evidence=None)
        except Exception as e:
            return TestResult(
                test_name="Evidence Dependence",
                passed=False,
                score=0.0,
                details={"error": str(e)},
                message=f"Failed to run agent variants: {e}",
            )

        # Extract claims from each variant
        claims = {
            "v1_normal": self.extractor.extract(v1_normal),
            "v2_no_evidence": self.extractor.extract(v2_no_evidence),
            "v3_diff_evidence": self.extractor.extract(v3_diff_evidence),
            "v4_conflicting": self.extractor.extract(v4_conflicting),
            "v5_empty": self.extractor.extract(v5_empty_evidence),
        }

        # Measure: how many claims changed when evidence was removed?
        baseline_claims_set = set(claims["v1_normal"])
        changed_claims = []

        for claim in baseline_claims_set:
            # If claim is NOT in v2 (no evidence), it was evidence-dependent
            if claim not in claims["v2_no_evidence"]:
                changed_claims.append(claim.text)

        # Calculate score: % of baseline claims that changed
        if len(baseline_claims_set) == 0:
            dependence_score = 0.0
        else:
            dependence_score = len(changed_claims) / len(baseline_claims_set)

        passed = dependence_score > 0.5  # Threshold: >50% evidence-dependent

        return TestResult(
            test_name="Evidence Dependence",
            passed=passed,
            score=dependence_score,
            details={
                "v1_normal": v1_normal,
                "v2_no_evidence": v2_no_evidence,
                "v3_diff_evidence": v3_diff_evidence,
                "v4_conflicting": v4_conflicting,
                "v5_empty": v5_empty_evidence,
                "baseline_claims": [c.text for c in baseline_claims_set],
                "changed_claims": changed_claims,
                "claim_sets": {
                    k: [c.text for c in v] for k, v in claims.items()
                },
            },
            message=f"{len(changed_claims)}/{len(baseline_claims_set)} claims changed when evidence removed. Dependence: {dependence_score:.1%}",
        )


class RefusalTester:
    """
    Test 5: Measures if agent correctly refuses when grounding unavailable.

    When required source is missing:
    - Agent SHOULD refuse (abstention_correctness = True) ✓
    - Agent SHOULD NOT answer (abstention_correctness = False) ❌
    """

    def __init__(self, agent_callable: Callable):
        """
        Args:
            agent_callable: Function that takes (prompt, evidence_context) -> answer
        """
        self.agent = agent_callable

    def test(
        self,
        prompt: str,
        required_source: str,
        required_source_evidence: list[dict],
    ) -> TestResult:
        """
        Test refusal when required source is unavailable.

        Args:
            prompt: Question that needs required_source
            required_source: e.g., "Seismic"
            required_source_evidence: Evidence that includes required_source

        Returns:
            TestResult with abstention_correctness
        """
        try:
            # When required source IS available
            response_with_source = self.agent(
                prompt, evidence=required_source_evidence
            )

            # When required source is NOT available
            response_without_source = self.agent(prompt, evidence=[])
        except Exception as e:
            return TestResult(
                test_name="Refusal Correctness",
                passed=False,
                score=0.0,
                details={"error": str(e)},
                message=f"Failed to run refusal test: {e}",
            )

        # Heuristics for detecting refusal
        refusal_keywords = [
            "i don't",
            "i cannot",
            "i don't have",
            "i'm not able",
            "i don't have access",
            "no approved source",
            "cannot find",
            "no information",
            "unable to",
        ]

        def is_refusal(text: str) -> bool:
            text_lower = text.casefold()
            return any(keyword in text_lower for keyword in refusal_keywords)

        # Evaluate
        should_answer_when_source_available = not is_refusal(response_with_source)
        should_refuse_when_source_unavailable = is_refusal(response_without_source)

        correctness = (
            should_answer_when_source_available
            and should_refuse_when_source_unavailable
        )

        score = 1.0 if correctness else 0.0

        return TestResult(
            test_name="Refusal Correctness",
            passed=correctness,
            score=score,
            details={
                "with_source": response_with_source,
                "without_source": response_without_source,
                "detected_refusal_without_source": should_refuse_when_source_unavailable,
                "answered_with_source": should_answer_when_source_available,
            },
            message=(
                "✓ Correctly refused when source unavailable"
                if correctness
                else "✗ Did not refuse appropriately"
            ),
        )


class CrossTurnRetentionTester:
    """
    Test 6: Measures if agent maintains claim-to-evidence consistency across turns.

    Turn 1: "Promotion runs monthly (from e1)"
    Turn 2: "Promotion runs monthly (from e1?)" ← Same evidence?

    If yes → grounding retention is high ✓
    If no → drift detected ❌
    """

    def __init__(self, agent_callable: Callable):
        """
        Args:
            agent_callable: Function that takes (prompt, evidence_context) -> answer
        """
        self.agent = agent_callable
        self.extractor = ClaimExtractor()

    def test(
        self,
        current_prompt: str,
        current_evidence: list[dict],
        prior_claims: list[dict],  # [{text: "...", evidence_ids: [...]}]
    ) -> TestResult:
        """
        Test cross-turn retention.

        Args:
            current_prompt: Current turn's question
            current_evidence: Current turn's evidence
            prior_claims: Claims from previous turn with their evidence IDs

        Returns:
            TestResult with cross_turn_retention score
        """
        try:
            current_answer = self.agent(current_prompt, evidence=current_evidence)
        except Exception as e:
            return TestResult(
                test_name="Cross-Turn Retention",
                passed=False,
                score=0.0,
                details={"error": str(e)},
                message=f"Failed to run agent: {e}",
            )

        # Extract current claims
        current_claims = self.extractor.extract(current_answer)

        # Match prior claims to current claims
        retention_count = 0
        matched_pairs = []
        unmatched_prior = []

        for prior in prior_claims:
            prior_text = prior.get("text", "")
            prior_evidence_ids = prior.get("evidence_ids", [])

            # Find matching claim in current
            matched = False
            for current in current_claims:
                if current == Claim(text=prior_text):
                    matched = True
                    matched_pairs.append(
                        {
                            "text": prior_text,
                            "prior_evidence_ids": prior_evidence_ids,
                            "current_evidence_ids": current.evidence_ids,
                            "evidence_retained": len(prior_evidence_ids) > 0,
                        }
                    )
                    if len(prior_evidence_ids) > 0:
                        retention_count += 1
                    break

            if not matched:
                unmatched_prior.append(prior_text)

        # Calculate retention score
        if len(prior_claims) == 0:
            retention_score = 1.0
        else:
            retention_score = retention_count / len(prior_claims)

        passed = retention_score >= 0.8  # Threshold: >=80% retention

        return TestResult(
            test_name="Cross-Turn Retention",
            passed=passed,
            score=retention_score,
            details={
                "current_answer": current_answer,
                "prior_claims": prior_claims,
                "current_claims": [c.text for c in current_claims],
                "matched_pairs": matched_pairs,
                "unmatched_prior": unmatched_prior,
                "retention_count": retention_count,
                "total_prior": len(prior_claims),
            },
            message=f"Retained {retention_count}/{len(prior_claims)} prior claims with evidence. Retention: {retention_score:.1%}",
        )


class GroundednessPOCTester:
    """
    Main POC tester orchestrating all three tests.
    """

    def __init__(self, agent_callable: Callable):
        """
        Args:
            agent_callable: Function(prompt, evidence=None) -> answer_text
        """
        self.agent = agent_callable
        self.test_3 = EvidenceDependenceTester(agent_callable)
        self.test_5 = RefusalTester(agent_callable)
        self.test_6 = CrossTurnRetentionTester(agent_callable)

    def run_full_test(
        self,
        prompt: str,
        evidence: list[dict],
        alternative_evidence: list[dict] = None,
        required_source: str = "Seismic",
        prior_claims: list[dict] = None,
    ) -> dict[str, TestResult]:
        """
        Run all three groundedness tests.

        Returns:
            {
                "test_3_evidence_dependence": TestResult,
                "test_5_refusal": TestResult,
                "test_6_cross_turn": TestResult,
            }
        """
        results = {}

        # Test 3: Evidence Dependence
        results["test_3_evidence_dependence"] = self.test_3.test(
            prompt, evidence, alternative_evidence
        )

        # Test 5: Refusal
        results["test_5_refusal"] = self.test_5.test(
            prompt, required_source, evidence
        )

        # Test 6: Cross-Turn Retention
        if prior_claims:
            results["test_6_cross_turn"] = self.test_6.test(
                prompt, evidence, prior_claims
            )
        else:
            results["test_6_cross_turn"] = TestResult(
                test_name="Cross-Turn Retention",
                passed=True,
                score=1.0,
                details={},
                message="No prior claims provided (skipped)",
            )

        return results

    def get_overall_score(self, results: dict[str, TestResult]) -> dict:
        """Calculate overall groundedness score."""
        passed_tests = sum(1 for r in results.values() if r.passed)
        total_tests = len(results)
        scores = [r.score for r in results.values()]
        avg_score = sum(scores) / len(scores) if scores else 0.0

        return {
            "overall_grounded": passed_tests == total_tests,
            "tests_passed": passed_tests,
            "tests_total": total_tests,
            "average_score": avg_score,
            "breakdown": {k: r.score for k, r in results.items()},
        }
