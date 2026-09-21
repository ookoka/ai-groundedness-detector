"""
Groundedness-Testing Custom Engine Agent for Teams.

A RAG agent that:
1. Retrieves evidence
2. Generates answers
3. Tests itself for groundedness (Test 3, 5, 6)
4. Returns answer + groundedness score to user

Integrates with groundedness_poc_tester.py
"""

from __future__ import annotations

import json
import logging
from typing import Any

from groundedness_poc_tester import (
    GroundednessPOCTester,
    TestResult,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GroundednessTestingAgent:
    """
    Custom Engine Agent that tests its own responses for groundedness.

    Workflow:
    1. User asks a question
    2. Agent retrieves evidence from knowledge base
    3. Agent generates answer
    4. Agent tests answer for groundedness (Evidence Dependence, Refusal, Cross-Turn)
    5. Return {answer, groundedness_score, test_results, confidence}
    """

    def __init__(self, knowledge_base: dict[str, Any] = None):
        """
        Args:
            knowledge_base: Dict mapping topics to evidence lists
        """
        self.knowledge_base = knowledge_base or self._default_kb()
        self.tester = GroundednessPOCTester(self._generate_answer)
        self.conversation_history = []

    @staticmethod
    def _default_kb() -> dict[str, list[dict]]:
        """Default knowledge base with promotional info."""
        return {
            "promotions": [
                {
                    "id": "e1",
                    "text": "Monthly promotions are established according to approved guidance in Seismic",
                    "source": "Seismic",
                    "topic": "promotion_frequency",
                },
                {
                    "id": "e2",
                    "text": "Promotional schedules occur first week of each month",
                    "source": "Seismic",
                    "topic": "promotion_timing",
                },
                {
                    "id": "e3",
                    "text": "Seismic tracks all promotional activities and serves as source of truth",
                    "source": "Seismic",
                    "topic": "seismic_tool",
                },
                {
                    "id": "e4",
                    "text": "Promotion budget is reviewed quarterly per policy",
                    "source": "Seismic",
                    "topic": "promotion_budget",
                },
            ],
            "approvals": [
                {
                    "id": "e5",
                    "text": "All promotions require approval from sales director",
                    "source": "Seismic",
                    "topic": "approval_process",
                },
                {
                    "id": "e6",
                    "text": "Approval workflow is documented in Seismic system",
                    "source": "Seismic",
                    "topic": "approval_workflow",
                },
            ],
        }

    def _retrieve_evidence(self, query: str) -> list[dict]:
        """
        Retrieve relevant evidence from knowledge base.

        Args:
            query: User's question

        Returns:
            List of relevant evidence items
        """
        query_lower = query.casefold()
        relevant = []

        for category, items in self.knowledge_base.items():
            for item in items:
                # Simple keyword matching
                if any(
                    keyword in query_lower
                    for keyword in [
                        item.get("topic", "").casefold(),
                        item.get("text", "").casefold()[:20],
                    ]
                ):
                    relevant.append(item)

        # If nothing matched, return all items from promotions category
        if not relevant:
            relevant = self.knowledge_base.get("promotions", [])

        return relevant

    def _generate_answer(
        self, prompt: str, evidence: list[dict] = None
    ) -> str:
        """
        Generate answer based on prompt and evidence.

        Args:
            prompt: User question
            evidence: Retrieved evidence

        Returns:
            Answer text
        """
        if evidence is None or len(evidence) == 0:
            return "I don't have access to the required knowledge base. Please provide evidence from Seismic or an approved source to answer this question accurately."

        # Build answer from evidence
        evidence_text = " ".join([item.get("text", "") for item in evidence])

        # Simple heuristic-based answer generation
        prompt_lower = prompt.casefold()

        if "promotion" in prompt_lower and "often" in prompt_lower:
            return f"Based on the knowledge base: {evidence_text[:180]}... The promotion runs monthly according to approved guidance."

        if "promotion" in prompt_lower and "timing" in prompt_lower:
            return f"According to approved guidance: {evidence_text[:150]}... Promotions are scheduled for the first week of each month."

        if "promotion" in prompt_lower and "budget" in prompt_lower:
            return f"Based on policy documentation: {evidence_text[:150]}... The promotion budget is reviewed quarterly."

        if "seismic" in prompt_lower:
            return f"From the knowledge base: {evidence_text[:180]}... Seismic is our primary system for tracking promotional activities."

        if "approval" in prompt_lower:
            return f"According to our approval process: {evidence_text[:180]}... All promotions must be approved by the sales director as documented in Seismic."

        # Generic answer using evidence
        return f"Based on available information: {evidence_text[:200]}... I can provide this answer based on our knowledge base."

    def process_user_query(self, query: str) -> dict[str, Any]:
        """
        Process user query and test groundedness.

        Args:
            query: User's question

        Returns:
            {
                "answer": str,
                "groundedness_score": float (0.0-1.0),
                "confident": bool (score >= 0.7),
                "tests": {
                    "test_3_evidence_dependence": TestResult,
                    "test_5_refusal": TestResult,
                    "test_6_cross_turn": TestResult
                },
                "evidence": list[dict],
                "turn": int
            }
        """
        try:
            # Step 1: Retrieve evidence
            evidence = self._retrieve_evidence(query)
            logger.info(f"Retrieved {len(evidence)} evidence items for query: {query}")

            # Step 2: Generate answer
            answer = self._generate_answer(query, evidence)
            logger.info(f"Generated answer: {answer[:100]}...")

            # Step 3: Prepare for groundedness testing
            # For Test 3: use alternative evidence (simulating evidence change)
            alternative_evidence = [
                {
                    "id": "alt1",
                    "text": "Alternative: Promotions run quarterly with different schedule",
                    "source": "Alternative",
                }
            ]

            # For Test 6: extract prior claims from conversation history
            prior_claims = []
            if len(self.conversation_history) > 0:
                last_turn = self.conversation_history[-1]
                if "claims" in last_turn:
                    prior_claims = last_turn["claims"]
                    logger.info(f"Using {len(prior_claims)} prior claims for Test 6")

            # Step 4: Run groundedness tests
            results = self.tester.run_full_test(
                prompt=query,
                evidence=evidence,
                alternative_evidence=alternative_evidence,
                required_source="Seismic",
                prior_claims=prior_claims,
            )

            # Step 5: Calculate overall groundedness score
            scores = [r.score for r in results.values() if r.score >= 0.0]
            overall_score = sum(scores) / len(scores) if scores else 0.0
            all_passed = all(r.passed for r in results.values())

            # Step 6: Build response
            response = {
                "answer": answer,
                "groundedness_score": round(overall_score, 2),
                "confident": overall_score >= 0.7,
                "all_tests_passed": all_passed,
                "tests": {
                    "test_3_evidence_dependence": {
                        "passed": results["test_3_evidence_dependence"].passed,
                        "score": results["test_3_evidence_dependence"].score,
                        "message": results["test_3_evidence_dependence"].message,
                    },
                    "test_5_refusal": {
                        "passed": results["test_5_refusal"].passed,
                        "score": results["test_5_refusal"].score,
                        "message": results["test_5_refusal"].message,
                    },
                    "test_6_cross_turn": {
                        "passed": results["test_6_cross_turn"].passed,
                        "score": results["test_6_cross_turn"].score,
                        "message": results["test_6_cross_turn"].message,
                    },
                },
                "evidence_count": len(evidence),
                "evidence_sources": list(set(e.get("source", "unknown") for e in evidence)),
                "turn": len(self.conversation_history) + 1,
            }

            # Step 7: Add to conversation history
            self.conversation_history.append(
                {
                    "query": query,
                    "answer": answer,
                    "groundedness_score": response["groundedness_score"],
                    "evidence": evidence,
                    # Store claims in the format expected by Test 6
                    "claims": [
                        {"text": answer, "evidence_ids": [e.get("id", "") for e in evidence]}
                    ],
                }
            )

            logger.info(
                f"Query processed. Groundedness: {response['groundedness_score']:.1%}"
            )
            return response

        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            return {
                "answer": f"Error: {str(e)}",
                "groundedness_score": 0.0,
                "confident": False,
                "error": str(e),
            }

    def get_stats(self) -> dict[str, Any]:
        """Get agent statistics."""
        if not self.conversation_history:
            return {
                "total_queries": 0,
                "average_groundedness": 0.0,
                "confident_answers": 0,
            }

        scores = [t["groundedness_score"] for t in self.conversation_history]
        confident = sum(1 for s in scores if s >= 0.7)

        return {
            "total_queries": len(self.conversation_history),
            "average_groundedness": round(sum(scores) / len(scores), 2),
            "confident_answers": confident,
            "low_confidence_count": len(scores) - confident,
        }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Initialize agent
    agent = GroundednessTestingAgent()

    # Example queries
    queries = [
        "How often does the promotion run?",
        "What is the promotion timing?",
        "What budget is allocated for promotions?",
        "How often does the promotion run?",  # Same query = can check cross-turn retention
    ]

    print("\n" + "=" * 70)
    print("Groundedness-Testing Custom Engine Agent Demo")
    print("=" * 70)

    for i, query in enumerate(queries, 1):
        print(f"\n--- Query {i} ---")
        print(f"User: {query}")

        response = agent.process_user_query(query)

        print(f"\nAgent Answer: {response['answer']}")
        print(f"\nGroundedness Score: {response['groundedness_score']:.1%}")
        print(f"Confident: {'Yes ✓' if response['confident'] else 'No ✗'}")

        print("\nTest Results:")
        for test_name, test_result in response["tests"].items():
            status = "✓ PASSED" if test_result["passed"] else "✗ FAILED"
            print(
                f"  {test_name}: {test_result['score']:.1%} {status} - {test_result['message']}"
            )

    print("\n" + "=" * 70)
    print("Agent Statistics")
    print("=" * 70)
    stats = agent.get_stats()
    print(f"Total Queries: {stats['total_queries']}")
    print(f"Average Groundedness: {stats['average_groundedness']:.1%}")
    print(f"Confident Answers: {stats['confident_answers']}")
    print(f"Low Confidence Answers: {stats['low_confidence_count']}")
