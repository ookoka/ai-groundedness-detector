#!/usr/bin/env python3
"""
Groundedness-Testing Custom Engine Agent

A Teams Custom Engine Agent that:
1. Receives user queries
2. Retrieves evidence
3. Generates answers
4. Tests for groundedness automatically
5. Returns answer + confidence verdict + test results

Ready for interactive demo or Teams integration.
"""

import asyncio
import json
from typing import Any

from groundedness_engine import GroundednessEngine
from groundedness_monitoring import GroundednessMonitor, GroundednessVerdict
from groundedness_poc_tester import GroundednessPOCTester
from my_data_source import MyDataSource


class CustomEngineAgent:
    """Teams Custom Engine Agent with integrated groundedness testing."""

    def __init__(self):
        """Initialize the agent with all components."""
        self.data_source = MyDataSource()
        self.monitor = GroundednessMonitor(log_dir="logs")
        self.engine = GroundednessEngine(monitor=self.monitor)
        self.conversation_history = {}

    async def process_message(
        self, conversation_id: str, user_message: str
    ) -> dict[str, Any]:
        """
        Process a user message in Teams conversation.

        Args:
            conversation_id: Unique identifier for the conversation
            user_message: User's input message

        Returns:
            {
                "message": formatted_message,
                "verdict": verdict_object,
                "answer": raw_answer,
                "evidence": evidence_list,
                "test_results": individual_test_results
            }
        """
        print(f"\n{'='*70}")
        print(f"Processing: {user_message}")
        print(f"Conversation ID: {conversation_id}")
        print(f"{'='*70}")

        # Step 1: Retrieve evidence
        print("\n[1/4] Retrieving evidence...")
        evidence_context = self.data_source.render_data(user_message)
        evidence_list = [
            {
                "id": f"ref_{i}",
                "text": evidence_context.output[: 100 + i * 50],
                "source": f"Reference {i+1}",
            }
            for i in range(min(3, len(evidence_context.output) // 50))
        ]
        print(f"  ✓ Found {len(evidence_list)} evidence items")

        # Step 2: Generate answer (mock - in real scenario, call LLM)
        print("\n[2/4] Generating answer...")
        answer = f"Based on the knowledge base: {evidence_context.output[:150]}... Answer to your query."
        print(f"  ✓ Answer generated ({len(answer)} chars)")

        # Step 3: Test groundedness
        print("\n[3/4] Testing groundedness...")
        verdict = await self.engine.test_response(
            query=user_message,
            answer=answer,
            evidence=evidence_list,
            conversation_id=conversation_id,
        )
        print(
            f"  ✓ Groundedness Score: {verdict.score:.0%} ({verdict.confidence_level.upper()})"
        )

        # Step 4: Format response
        print("\n[4/4] Formatting response...")
        verdict_text = verdict.get_verdict_text()
        formatted_message = f"{verdict_text}\n[Confidence: {verdict.confidence_level.upper()} {verdict.score:.0%}]\n\n{answer}"
        print(f"  ✓ Message formatted and ready to send")

        # Store in conversation history
        if conversation_id not in self.conversation_history:
            self.conversation_history[conversation_id] = []
        self.conversation_history[conversation_id].append(
            {
                "query": user_message,
                "answer": answer,
                "verdict": verdict,
                "evidence": evidence_list,
            }
        )

        return {
            "message": formatted_message,
            "verdict": verdict,
            "answer": answer,
            "evidence": evidence_list,
            "test_results": verdict.test_results,
        }

    def get_conversation_summary(self, conversation_id: str) -> str:
        """Get summary of conversation history."""
        if conversation_id not in self.conversation_history:
            return f"No conversation history for {conversation_id}"

        history = self.conversation_history[conversation_id]
        summary = f"\nConversation Summary ({conversation_id}):\n"
        summary += f"{'='*70}\n"
        for i, turn in enumerate(history, 1):
            verdict = turn["verdict"]
            summary += f"Turn {i}: {turn['query'][:40]}...\n"
            summary += f"  Score: {verdict.score:.0%} ({verdict.confidence_level})\n"
            summary += f"  Evidence: {len(turn['evidence'])} items\n"
            summary += "\n"
        return summary

    def get_monitoring_stats(self) -> str:
        """Get overall groundedness monitoring statistics."""
        return self.monitor.get_summary()


async def interactive_demo():
    """Run interactive demo where you can test the agent."""
    agent = CustomEngineAgent()
    conversation_id = "demo-2026-09-21"

    print("\n" + "="*70)
    print("CUSTOM ENGINE AGENT - INTERACTIVE DEMO")
    print("="*70)
    print("\nThis agent tests itself for groundedness on every response.")
    print("Type 'quit' to exit, 'stats' for statistics, 'history' for conversation.\n")

    queries = [
        "How often does the promotion run?",
        "What is the promotion timing?",
        "When are promotions scheduled?",
    ]

    for query in queries:
        result = await agent.process_message(conversation_id, query)

        print("\n" + "-" * 70)
        print("AGENT RESPONSE (Teams Message):")
        print("-" * 70)
        print(result["message"])

        print("\n" + "-" * 70)
        print("TEST RESULTS BREAKDOWN:")
        print("-" * 70)
        for test_name, test_result in result["test_results"].items():
            status = "✓ PASSED" if test_result["passed"] else "✗ FAILED"
            print(f"{test_name}:")
            print(f"  Score: {test_result['score']:.0%} {status}")
            print(f"  Message: {test_result['message']}\n")

        print("\n[Press Enter to continue to next query...]\n")
        await asyncio.sleep(2)

    # Final stats
    print("\n" + "="*70)
    print("FINAL STATISTICS")
    print("="*70)
    print(agent.get_monitoring_stats())
    print(agent.get_conversation_summary(conversation_id))


async def main():
    """Main entry point."""
    await interactive_demo()


if __name__ == "__main__":
    asyncio.run(main())
