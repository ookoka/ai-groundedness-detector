import unittest

from ai_groundedness_detector import (
    AgentRun,
    Claim,
    DetectorTrace,
    Evidence,
    GroundednessDetector,
    GroundingPolicy,
    Intervention,
    ToolCallTrace,
)
from groundedness_agent_wrapper import validate_agent_output
from validator_agent import ValidatorAgent


def trace(turn_id=1, intervention=Intervention.FULL_AGENT):
    return DetectorTrace(
        evaluation_run_id="run-20260915-0042",
        agent_id="agent-j-dev",
        agent_version="2.4.1",
        conversation_id="conversation-1",
        turn_id=turn_id,
        scenario_id="seismic-product-guidance",
        grounding_configuration_version="grounding-v7",
        intervention=intervention,
    )


class GroundednessDetectorTests(unittest.TestCase):
    def test_reports_copilot_route_policy_and_support_metrics(self):
        run = AgentRun(
            trace=trace(),
            prompt="What is the approved product guidance?",
            answer="The approved cadence is monthly.",
            observed_route="Industry search",
            evidence=(Evidence(id="e1", source="Industry search", text="Monthly cadence."),),
            claims=(
                Claim(
                    id="c1",
                    text="The approved cadence is monthly.",
                    supporting_evidence_ids=("e1",),
                    cited_evidence_ids=("e1",),
                ),
            ),
        )
        policy = GroundingPolicy(required_source="Seismic")

        result = GroundednessDetector().evaluate(run, policy)

        self.assertEqual(result.required_source, "Seismic")
        self.assertEqual(result.observed_route, "Industry search")
        self.assertFalse(result.metrics["grounding_route_correctness"])
        self.assertTrue(result.metrics["evidence_support_precision"])
        self.assertFalse(result.metrics["grounding_policy_adherence"])
        self.assertIn("grounding_route_incorrect", result.issues)

    def test_flags_unsupported_claims_and_bad_citations(self):
        run = AgentRun(
            trace=trace(),
            prompt="Summarize the plan.",
            answer="The plan runs weekly and was approved by Contoso.",
            observed_route="Seismic",
            evidence=(Evidence(id="e1", source="Seismic", text="The plan runs weekly."),),
            claims=(
                Claim(
                    id="c1",
                    text="The plan runs weekly.",
                    supporting_evidence_ids=("e1",),
                    cited_evidence_ids=("missing",),
                ),
                Claim(id="c2", text="The plan was approved by Contoso."),
            ),
        )
        policy = GroundingPolicy(required_source="Seismic")

        result = GroundednessDetector().evaluate(run, policy)

        self.assertEqual(result.unsupported_claims, ("c2",))
        self.assertEqual(result.invalid_citations, ("c1",))
        self.assertEqual(result.metrics["evidence_support_precision"], 0.5)
        self.assertEqual(result.metrics["citation_correctness"], 0.5)

    def test_measures_evidence_dependence_with_controlled_variants(self):
        full_run = AgentRun(
            trace=trace(),
            prompt="How often does the promotion run?",
            answer="The promotion runs monthly.",
            observed_route="Seismic",
            evidence=(Evidence(id="e1", source="Seismic", text="Promotion runs monthly."),),
            claims=(
                Claim(
                    id="c1",
                    text="The promotion runs monthly.",
                    supporting_evidence_ids=("e1",),
                    cited_evidence_ids=("e1",),
                ),
            ),
        )
        removed_run = AgentRun(
            trace=trace(intervention=Intervention.KNOWLEDGE_REMOVED),
            prompt=full_run.prompt,
            answer="I do not have an approved source for that.",
            refused=True,
        )
        changed_document_run = AgentRun(
            trace=trace(intervention=Intervention.SUPPORTING_DOCUMENT_CHANGED),
            prompt=full_run.prompt,
            answer="The promotion runs quarterly.",
            observed_route="Seismic",
            evidence=(Evidence(id="e2", source="Seismic", text="Promotion runs quarterly."),),
            claims=(
                Claim(
                    id="c1",
                    text="The promotion runs quarterly.",
                    supporting_evidence_ids=("e2",),
                    cited_evidence_ids=("e2",),
                ),
            ),
        )

        result = GroundednessDetector().evaluate(
            full_run,
            GroundingPolicy(required_source="Seismic"),
            counterfactual_runs=(removed_run, changed_document_run),
        )

        self.assertEqual(result.metrics["evidence_dependence"], 1.0)
        self.assertEqual(
            [observation["intervention"] for observation in result.counterfactual_observations],
            ["knowledge-removed", "supporting-document-changed"],
        )

    def test_captures_tool_groundedness_against_structured_tool_output(self):
        run = AgentRun(
            trace=trace(),
            prompt="Find product guidance.",
            answer="No Seismic result was used.",
            observed_route="Industry search",
            evidence=(Evidence(id="e1", source="Industry search", text="General content."),),
            tool_calls=(
                ToolCallTrace(
                    tool_expected="search_seismic",
                    tool_selected="search_industry_content",
                    arguments_valid=True,
                    execution_succeeded=True,
                    result_used_in_answer=False,
                ),
            ),
        )

        result = GroundednessDetector().evaluate(
            run,
            GroundingPolicy(required_source="Seismic", expected_tool="search_seismic"),
        )

        self.assertEqual(result.metrics["tool_output_utilization"], 0.0)
        self.assertFalse(result.tool_call_results[0]["selected_expected_tool"])
        self.assertIn("tool_output_not_fully_utilized", result.issues)

    def test_checks_abstention_when_approved_source_is_missing(self):
        answered_run = AgentRun(
            trace=trace(),
            prompt="Use Seismic to answer.",
            answer="The answer is monthly.",
            observed_route="general knowledge",
        )
        refused_run = AgentRun(
            trace=trace(),
            prompt="Use Seismic to answer.",
            answer="I do not have an approved source for that.",
            refused=True,
        )
        policy = GroundingPolicy(required_source="Seismic")

        answered_result = GroundednessDetector().evaluate(answered_run, policy)
        refused_result = GroundednessDetector().evaluate(refused_run, policy)

        self.assertFalse(answered_result.metrics["abstention_correctness"])
        self.assertTrue(refused_result.metrics["abstention_correctness"])
        self.assertTrue(refused_result.metrics["grounding_route_correctness"])
        self.assertTrue(refused_result.metrics["grounding_policy_adherence"])

    def test_detects_cross_turn_grounding_retention_drift(self):
        prior = Claim(
            id="c1",
            text="The promotion runs monthly.",
            supporting_evidence_ids=("e1",),
            cited_evidence_ids=("e1",),
        )
        current = Claim(
            id="c2",
            text="The promotion runs monthly.",
            supporting_evidence_ids=(),
            cited_evidence_ids=(),
        )
        run = AgentRun(
            trace=trace(turn_id=4),
            prompt="Summarize the decision.",
            answer="The promotion runs monthly.",
            observed_route="Seismic",
            evidence=(Evidence(id="e1", source="Seismic", text="Promotion runs monthly."),),
            claims=(current,),
            prior_turn_claims=(prior,),
        )

        result = GroundednessDetector().evaluate(run, GroundingPolicy(required_source="Seismic"))

        self.assertEqual(result.metrics["cross_turn_grounding_retention"], 0.0)
        self.assertIn("cross_turn_grounding_drift", result.issues)

    def test_validator_wrapper_converts_agent_output_to_detector_schema(self):
        raw_output = {
            "prompt": "How often does the promotion run?",
            "answer": "The promotion runs monthly.",
            "observed_route": "Seismic",
            "evidence": [
                {"id": "e1", "source": "Seismic", "text": "Promotion runs monthly."},
            ],
            "claims": [
                {
                    "id": "c1",
                    "text": "The promotion runs monthly.",
                    "supporting_evidence_ids": ["e1"],
                    "cited_evidence_ids": ["e1"],
                }
            ],
            "required_source": "Seismic",
            "expected_tool": "search_seismic",
        }

        result = validate_agent_output(raw_output)

        self.assertTrue(result["metrics"]["grounding_route_correctness"])
        self.assertTrue(result["metrics"]["grounding_policy_adherence"])
        self.assertEqual(result["issues"], ())

    def test_validator_agent_builds_payload_from_raw_answer(self):
        agent = ValidatorAgent(required_source="Seismic", expected_tool="search_seismic")
        payload = agent.build_payload(
            prompt="How often does the promotion run?",
            answer="The promotion runs monthly.",
            evidence=[
                {"id": "e1", "source": "Seismic", "text": "Promotion runs monthly."},
            ],
            observed_route="Seismic",
        )

        self.assertEqual(payload["required_source"], "Seismic")
        self.assertEqual(payload["expected_tool"], "search_seismic")
        self.assertEqual(payload["claims"][0]["text"], "The promotion runs monthly.")

        result = agent.validate(
            prompt="How often does the promotion run?",
            answer="The promotion runs monthly.",
            evidence=[
                {"id": "e1", "source": "Seismic", "text": "Promotion runs monthly."},
            ],
            observed_route="Seismic",
        )

        self.assertTrue(result["metrics"]["grounding_route_correctness"])
        self.assertTrue(result["metrics"]["grounding_policy_adherence"])


if __name__ == "__main__":
    unittest.main()
