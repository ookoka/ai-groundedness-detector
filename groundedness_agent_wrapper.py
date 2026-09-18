from __future__ import annotations

from typing import Any, Iterable

from ai_groundedness_detector import (
    AgentRun,
    Claim,
    DetectorTrace,
    Evidence,
    GroundednessDetector,
    GroundingPolicy,
)


def validate_agent_output(raw_output: dict[str, Any]) -> dict[str, Any]:
    """Convert a raw agent response into the detector schema and evaluate it.

    This function is the reusable adapter layer for multi-agent or tool-based validation.
    It takes output from an agent or orchestration step and turns it into:
    - Evidence objects
    - Claim objects
    - AgentRun metadata
    - GroundingPolicy
    - a groundedness evaluation result
    """
    evidence = tuple(
        Evidence(
            id=item["id"],
            source=item.get("source", "unknown"),
            text=item.get("text", ""),
            source_type=item.get("source_type", "document"),
            citation=item.get("citation"),
        )
        for item in raw_output.get("evidence", [])
    )

    claims = tuple(
        Claim(
            id=item["id"],
            text=item.get("text", ""),
            supporting_evidence_ids=tuple(item.get("supporting_evidence_ids", ())),
            cited_evidence_ids=tuple(item.get("cited_evidence_ids", ())),
            contradicted_by_evidence_ids=tuple(item.get("contradicted_by_evidence_ids", ())),
            derived_from_tool_call_ids=tuple(item.get("derived_from_tool_call_ids", ())),
        )
        for item in raw_output.get("claims", [])
    )

    run = AgentRun(
        trace=DetectorTrace(
            evaluation_run_id=raw_output.get("evaluation_run_id", "run-demo"),
            agent_id=raw_output.get("agent_id", "agent-j"),
            agent_version=raw_output.get("agent_version", "1.0"),
            conversation_id=raw_output.get("conversation_id", "demo-conversation"),
            turn_id=int(raw_output.get("turn_id", 1)),
            scenario_id=raw_output.get("scenario_id", "demo-scenario"),
            grounding_configuration_version=raw_output.get("grounding_configuration_version", "v1"),
        ),
        prompt=raw_output.get("prompt", ""),
        answer=raw_output.get("answer", ""),
        observed_route=raw_output.get("observed_route"),
        evidence=evidence,
        claims=claims,
        tool_calls=tuple(
            _tool_call_from_dict(item)
            for item in raw_output.get("tool_calls", [])
        ),
        refused=bool(raw_output.get("refused", False)),
        prior_turn_claims=tuple(
            Claim(
                id=item["id"],
                text=item.get("text", ""),
                supporting_evidence_ids=tuple(item.get("supporting_evidence_ids", ())),
                cited_evidence_ids=tuple(item.get("cited_evidence_ids", ())),
                contradicted_by_evidence_ids=tuple(item.get("contradicted_by_evidence_ids", ())),
                derived_from_tool_call_ids=tuple(item.get("derived_from_tool_call_ids", ())),
            )
            for item in raw_output.get("prior_turn_claims", [])
        ),
    )

    policy = GroundingPolicy(
        grounding_required=raw_output.get("grounding_required", True),
        required_source=raw_output.get("required_source"),
        allowed_sources=tuple(raw_output.get("allowed_sources", ())),
        prohibited_source_combinations=tuple(
            tuple(combo) for combo in raw_output.get("prohibited_source_combinations", ())
        ),
        expected_tool=raw_output.get("expected_tool"),
        require_citations=raw_output.get("require_citations", True),
        refuse_when_no_approved_source=raw_output.get("refuse_when_no_approved_source", True),
    )

    result = GroundednessDetector().evaluate(
        run,
        policy,
        counterfactual_runs=(
            _convert_counterfactual_run(item) for item in raw_output.get("counterfactual_runs", [])
        ),
    )
    return {
        "trace": result.trace,
        "required_source": result.required_source,
        "observed_route": result.observed_route,
        "metrics": result.metrics,
        "issues": result.issues,
        "unsupported_claims": result.unsupported_claims,
        "contradictory_claims": result.contradictory_claims,
        "invalid_citations": result.invalid_citations,
        "tool_call_results": result.tool_call_results,
        "counterfactual_observations": result.counterfactual_observations,
    }


def _tool_call_from_dict(raw_tool_call: dict[str, Any]):
    from ai_groundedness_detector import ToolCallTrace

    return ToolCallTrace(
        tool_expected=raw_tool_call.get("tool_expected"),
        tool_selected=raw_tool_call.get("tool_selected", ""),
        arguments_valid=bool(raw_tool_call.get("arguments_valid", False)),
        execution_succeeded=bool(raw_tool_call.get("execution_succeeded", False)),
        result_used_in_answer=bool(raw_tool_call.get("result_used_in_answer", False)),
        claims_supported=tuple(raw_tool_call.get("claims_supported", ())),
        result=raw_tool_call.get("result"),
    )


def _convert_counterfactual_run(raw_run: dict[str, Any]) -> AgentRun:
    return AgentRun(
        trace=DetectorTrace(
            evaluation_run_id=raw_run.get("evaluation_run_id", "counterfactual-run"),
            agent_id=raw_run.get("agent_id", "agent-j"),
            agent_version=raw_run.get("agent_version", "1.0"),
            conversation_id=raw_run.get("conversation_id", "demo-conversation"),
            turn_id=int(raw_run.get("turn_id", 1)),
            scenario_id=raw_run.get("scenario_id", "demo-scenario"),
            grounding_configuration_version=raw_run.get("grounding_configuration_version", "v1"),
            intervention=raw_run.get("intervention", "full-agent"),
        ),
        prompt=raw_run.get("prompt", ""),
        answer=raw_run.get("answer", ""),
        observed_route=raw_run.get("observed_route"),
        evidence=tuple(
            Evidence(
                id=item["id"],
                source=item.get("source", "unknown"),
                text=item.get("text", ""),
                source_type=item.get("source_type", "document"),
                citation=item.get("citation"),
            )
            for item in raw_run.get("evidence", [])
        ),
        claims=tuple(
            Claim(
                id=item["id"],
                text=item.get("text", ""),
                supporting_evidence_ids=tuple(item.get("supporting_evidence_ids", ())),
                cited_evidence_ids=tuple(item.get("cited_evidence_ids", ())),
                contradicted_by_evidence_ids=tuple(item.get("contradicted_by_evidence_ids", ())),
                derived_from_tool_call_ids=tuple(item.get("derived_from_tool_call_ids", ())),
            )
            for item in raw_run.get("claims", [])
        ),
        refused=bool(raw_run.get("refused", False)),
    )
