#!/usr/bin/env python3
"""
Interactive Groundedness Testing Demo - Perfect for Video Recording

Shows:
1. User enters a query
2. Agent generates response
3. Tests run automatically
4. Results displayed with verdicts
"""

import asyncio
import sys
from pathlib import Path
from groundedness_engine import GroundednessEngine
from groundedness_monitoring import GroundednessMonitor

# Ensure logging is visible
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class InteractiveDemoAgent:
    """Simple agent for recording demo"""
    
    def __init__(self):
        self.monitor = GroundednessMonitor(log_dir="demo_logs")
        self.engine = GroundednessEngine(monitor=self.monitor)
        
        # Simple knowledge base
        self.knowledge_base = {
            "promotions": "Monthly promotions are scheduled for the first week of each month",
            "budget": "Promotional budgets are reviewed quarterly",
            "scheduling": "Promotions occur every 30 days starting from January 1st",
            "approval": "All promotions require manager approval before launch"
        }
    
    def _generate_answer(self, query: str) -> tuple[str, list]:
        """Generate answer and evidence"""
        query_lower = query.lower()
        evidence = []
        answers = []
        
        if "promotion" in query_lower and "often" in query_lower:
            answer = "Monthly promotions are scheduled for the first week of each month"
            evidence = [{"id": "1", "text": self.knowledge_base["promotions"], "source": "Promotion Policy"}]
            answers.append(answer)
        
        elif "budget" in query_lower:
            answer = "Promotional budgets are reviewed quarterly"
            evidence = [{"id": "2", "text": self.knowledge_base["budget"], "source": "Budget Policy"}]
            answers.append(answer)
        
        elif "approval" in query_lower:
            answer = "All promotions require manager approval before launch"
            evidence = [{"id": "3", "text": self.knowledge_base["approval"], "source": "Approval Policy"}]
            answers.append(answer)
        
        else:
            answer = "I don't have specific information about that in the knowledge base"
            evidence = []
            answers.append(answer)
        
        return " ".join(answers), evidence
    
    async def process_query(self, query: str, conv_id: str = "demo-session") -> None:
        """Process a query and display results"""
        print("\n" + "="*80)
        print(f"USER QUERY: {query}")
        print("="*80)
        
        # Generate answer
        print("\n[AGENT PROCESSING...]")
        answer, evidence = self._generate_answer(query)
        
        print(f"\n📝 GENERATED ANSWER:")
        print(f"   {answer}")
        
        if evidence:
            print(f"\n📚 EVIDENCE USED:")
            for ev in evidence:
                print(f"   [{ev['id']}] {ev['source']}: {ev['text'][:60]}...")
        else:
            print(f"\n📚 EVIDENCE USED: None")
        
        # Run groundedness tests
        print(f"\n[TESTING GROUNDEDNESS...]")
        verdict = await self.engine.test_response(
            query=query,
            answer=answer,
            evidence=evidence,
            conversation_id=conv_id
        )
        
        # Display verdict
        print(f"\n{'='*80}")
        print(f"GROUNDEDNESS VERDICT")
        print(f"{'='*80}")
        print(f"\n{verdict.get_verdict_text()}")
        print(f"Overall Score: {verdict.score:.0%}")
        print(f"Confidence Level: {verdict.confidence_level.upper()}")
        print(f"All Tests Passed: {verdict.all_tests_passed}")
        
        # Detailed test results
        print(f"\n{'DETAILED TEST RESULTS':^80}")
        print("-" * 80)
        
        for test_name, test_result in verdict.test_results.items():
            status = "✅ PASS" if test_result["passed"] else "❌ FAIL"
            test_label = test_name.replace("test_", "Test ").replace("_", " ").title()
            print(f"\n{test_label}: {status}")
            print(f"  Score: {test_result['score']:.0%}")
            print(f"  Details: {test_result['message']}")
        
        print(f"\n{'='*80}\n")


async def main():
    """Main demo loop"""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + "GROUNDEDNESS TESTING AGENT - INTERACTIVE DEMO".center(78) + "║")
    print("║" + "Perfect for Recording & Demonstration".center(78) + "║")
    print("╚" + "="*78 + "╝")
    print("\nThis demo shows the groundedness testing framework in action.")
    print("Type your queries and watch as the agent:")
    print("  1. Generates responses based on knowledge")
    print("  2. Runs 3 groundedness tests automatically")
    print("  3. Displays confidence verdicts\n")
    
    agent = InteractiveDemoAgent()
    
    # Demo queries for recording
    demo_queries = [
        "How often does the promotion run?",
        "When are promotional budgets reviewed?",
        "What approvals are needed for promotions?",
        "Tell me about promotion scheduling"
    ]
    
    print("Press Ctrl+C to exit\n")
    
    for i, query in enumerate(demo_queries, 1):
        await agent.process_query(query, conv_id=f"demo-turn-{i}")
        
        if i < len(demo_queries):
            input("Press ENTER to continue to next query...")
    
    # Summary
    print("\n" + "="*80)
    print("MONITORING SUMMARY")
    print("="*80)
    print(agent.monitor.get_summary())
    
    print("\n✅ Demo Complete! Logs saved to: ./demo_logs/")
    print(f"   - Verdicts: ./demo_logs/groundedness_verdicts.jsonl")
    print(f"   - Statistics: ./demo_logs/stats.json")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user")
        sys.exit(0)
