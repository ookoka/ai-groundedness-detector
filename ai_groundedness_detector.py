"""Black-box groundedness evaluation primitives for Copilot-style agents.

The detector expects the caller to provide the observed response trace: claims,
evidence, selected knowledge/tool route, citations, tool calls, policy, and any
controlled counterfactual runs. It then computes route, support, dependence,
policy, tool-utilization, abstention, and cross-turn retention signals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable


class Intervention(str, Enum):
    """Supported run variants for evidence-dependence testing."""

    FULL_AGENT = "full-agent"
    KNOWLEDGE_REMOVED = "knowledge-removed"
    SUPPORTING_DOCUMENT_CHANGED = "supporting-document-changed"
    DISTRACTOR_ADDED = "distractor-added"
    TOOL_UNAVAILABLE = "tool-unavailable"


@dataclass(frozen=True)
class DetectorTrace:
    evaluation_run_id: str
    agent_id: str
    agent_version: str
    conversation_id: str
    turn_id: int
    scenario_id: str
    grounding_configuration_version: str
    intervention: Intervention = Intervention.FULL_AGENT
    instruction_version: str | None = None
    knowledge_configuration: str | None = None
    plugin_configuration: str | None = None
    test_user_identity: str | None = None
    test_environment: str | None = None


@dataclass(frozen=True)
class Evidence:
    id: str
    source: str
    text: str
    source_type: str = "document"
    citation: str | None = None


@dataclass(frozen=True)
class Claim:
    id: str
    text: str
    supporting_evidence_ids: tuple[str, ...] = ()
    cited_evidence_ids: tuple[str, ...] = ()
    contradicted_by_evidence_ids: tuple[str, ...] = ()
    derived_from_tool_call_ids: tuple[str, ...] = ()

    @property
    def is_supported(self) -> bool:
        return bool(self.supporting_evidence_ids) and not self.contradicted_by_evidence_ids


@dataclass(frozen=True)
class ToolCallTrace:
    tool_expected: str | None
    tool_selected: str
    arguments_valid: bool
    execution_succeeded: bool
    result_used_in_answer: bool
    claims_supported: tuple[str, ...] = ()
    result: Any = None


@dataclass(frozen=True)
class GroundingPolicy:
    grounding_required: bool = True
    required_source: str | None = None
    allowed_sources: tuple[str, ...] = ()
    prohibited_source_combinations: tuple[tuple[str, ...], ...] = ()
    expected_tool: str | None = None
    require_citations: bool = True
    refuse_when_no_approved_source: bool = True


@dataclass(frozen=True)
class AgentRun:
    trace: DetectorTrace
    prompt: str
    answer: str
    claims: tuple[Claim, ...] = ()
    evidence: tuple[Evidence, ...] = ()
    observed_route: str | None = None
    tool_calls: tuple[ToolCallTrace, ...] = ()
    refused: bool = False
    prior_turn_claims: tuple[Claim, ...] = ()


@dataclass(frozen=True)
class EvaluationResult:
    trace: DetectorTrace
    required_source: str | None
    observed_route: str | None
    metrics: dict[str, float | bool]
    issues: tuple[str, ...]
    unsupported_claims: tuple[str, ...]
    contradictory_claims: tuple[str, ...]
    invalid_citations: tuple[str, ...]
    tool_call_results: tuple[dict[str, Any], ...]
    counterfactual_observations: tuple[dict[str, Any], ...]


class GroundednessDetector:
    """Evaluate groundedness from black-box run traces and counterfactuals."""

    def evaluate(
        self,
        run: AgentRun,
        policy: GroundingPolicy,
        counterfactual_runs: Iterable[AgentRun] = (),
    ) -> EvaluationResult:
        evidence_ids = {item.id for item in run.evidence}
        source_names = {item.source for item in run.evidence}
        issues: list[str] = []

        unsupported_claims = tuple(claim.id for claim in run.claims if not claim.is_supported)
        contradictory_claims = tuple(
            claim.id for claim in run.claims if claim.contradicted_by_evidence_ids
        )
        invalid_citations = tuple(
            claim.id
            for claim in run.claims
            if any(citation_id not in evidence_ids for citation_id in claim.cited_evidence_ids)
            or (policy.require_citations and claim.is_supported and not claim.cited_evidence_ids)
        )

        if unsupported_claims:
            issues.append("unsupported_or_contradictory_claims")
        if invalid_citations:
            issues.append("invalid_or_missing_citations")

        route_correct = self._route_correct(run, policy, source_names)
        if not route_correct:
            issues.append("grounding_route_incorrect")

        blended_prohibited_sources = self._blended_prohibited_sources(
            source_names, policy.prohibited_source_combinations
        )
        if blended_prohibited_sources:
            issues.append("prohibited_source_blending")

        tool_results = tuple(self._evaluate_tool_call(tool_call) for tool_call in run.tool_calls)
        tool_output_utilization = self._tool_output_utilization(tool_results, policy)
        if tool_results and tool_output_utilization < 1:
            issues.append("tool_output_not_fully_utilized")

        evidence_support_precision = self._ratio(
            sum(1 for claim in run.claims if claim.is_supported), len(run.claims)
        )
        citation_correctness = self._ratio(len(run.claims) - len(invalid_citations), len(run.claims))
        evidence_coverage = self._evidence_coverage(run.claims, evidence_ids)
        abstention_correctness = self._abstention_correctness(run, policy, source_names)
        if not abstention_correctness:
            issues.append("abstention_incorrect")

        policy_adherence = (
            route_correct
            and not blended_prohibited_sources
            and (not policy.require_citations or not invalid_citations)
            and abstention_correctness
        )

        counterfactual_observations = tuple(
            self._compare_counterfactual(run, variant) for variant in counterfactual_runs
        )
        evidence_dependence = self._evidence_dependence(counterfactual_observations, run.claims)
        cross_turn_retention = self._cross_turn_grounding_retention(run)
        if cross_turn_retention < 1:
            issues.append("cross_turn_grounding_drift")

        metrics: dict[str, float | bool] = {
            "grounding_route_correctness": route_correct,
            "evidence_support_precision": evidence_support_precision,
            "citation_correctness": citation_correctness,
            "evidence_coverage": evidence_coverage,
            "evidence_dependence": evidence_dependence,
            "grounding_policy_adherence": policy_adherence,
            "tool_output_utilization": tool_output_utilization,
            "abstention_correctness": abstention_correctness,
            "cross_turn_grounding_retention": cross_turn_retention,
        }

        return EvaluationResult(
            trace=run.trace,
            required_source=policy.required_source,
            observed_route=run.observed_route,
            metrics=metrics,
            issues=tuple(issues),
            unsupported_claims=unsupported_claims,
            contradictory_claims=contradictory_claims,
            invalid_citations=invalid_citations,
            tool_call_results=tool_results,
            counterfactual_observations=counterfactual_observations,
        )

    def _route_correct(
        self, run: AgentRun, policy: GroundingPolicy, source_names: set[str]
    ) -> bool:
        if not policy.grounding_required:
            return True
        if policy.required_source:
            if policy.required_source not in source_names and run.refused:
                return True
            return run.observed_route == policy.required_source and policy.required_source in source_names
        if policy.allowed_sources:
            if not source_names.intersection(policy.allowed_sources) and run.refused:
                return True
            return bool(source_names.intersection(policy.allowed_sources))
        return bool(run.evidence) or run.refused

    def _evaluate_tool_call(self, tool_call: ToolCallTrace) -> dict[str, Any]:
        selected_expected_tool = (
            tool_call.tool_expected is None or tool_call.tool_expected == tool_call.tool_selected
        )
        return {
            "tool_expected": tool_call.tool_expected,
            "tool_selected": tool_call.tool_selected,
            "arguments_valid": tool_call.arguments_valid,
            "execution_succeeded": tool_call.execution_succeeded,
            "result_used_in_answer": tool_call.result_used_in_answer,
            "claims_supported": list(tool_call.claims_supported),
            "selected_expected_tool": selected_expected_tool,
        }

    def _tool_output_utilization(
        self, tool_results: tuple[dict[str, Any], ...], policy: GroundingPolicy
    ) -> float:
        relevant_results = [
            result
            for result in tool_results
            if policy.expected_tool is None or result["tool_expected"] == policy.expected_tool
        ]
        if not relevant_results:
            return 1.0
        usable_results = [
            result
            for result in relevant_results
            if result["selected_expected_tool"]
            and result["arguments_valid"]
            and result["execution_succeeded"]
            and result["result_used_in_answer"]
        ]
        return self._ratio(len(usable_results), len(relevant_results))

    def _abstention_correctness(
        self, run: AgentRun, policy: GroundingPolicy, source_names: set[str]
    ) -> bool:
        approved_source_available = (
            policy.required_source in source_names
            if policy.required_source
            else bool(source_names.intersection(policy.allowed_sources)) or bool(run.evidence)
        )
        if policy.refuse_when_no_approved_source and not approved_source_available:
            return run.refused
        if approved_source_available and run.refused:
            return False
        return True

    def _compare_counterfactual(self, run: AgentRun, variant: AgentRun) -> dict[str, Any]:
        baseline_claims = {claim.id: self._normalise(claim.text) for claim in run.claims}
        variant_claims = {claim.id: self._normalise(claim.text) for claim in variant.claims}
        changed_claims = sorted(
            claim_id
            for claim_id, text in baseline_claims.items()
            if variant_claims.get(claim_id) != text
        )
        preserved_claims = sorted(
            claim_id
            for claim_id, text in baseline_claims.items()
            if variant_claims.get(claim_id) == text
        )
        return {
            "intervention": variant.trace.intervention.value,
            "refused": variant.refused,
            "changed_claims": changed_claims,
            "preserved_claims": preserved_claims,
        }

    def _evidence_dependence(
        self, observations: tuple[dict[str, Any], ...], claims: tuple[Claim, ...]
    ) -> float:
        supported_claim_ids = {claim.id for claim in claims if claim.is_supported}
        if not supported_claim_ids:
            return 1.0
        dependence_observations = [
            observation
            for observation in observations
            if observation["intervention"]
            in {
                Intervention.KNOWLEDGE_REMOVED.value,
                Intervention.SUPPORTING_DOCUMENT_CHANGED.value,
                Intervention.TOOL_UNAVAILABLE.value,
            }
        ]
        if not dependence_observations:
            return 0.0
        dependent_claim_ids: set[str] = set()
        for observation in dependence_observations:
            if observation["refused"]:
                dependent_claim_ids.update(supported_claim_ids)
            dependent_claim_ids.update(supported_claim_ids.intersection(observation["changed_claims"]))
        return self._ratio(len(dependent_claim_ids), len(supported_claim_ids))

    def _cross_turn_grounding_retention(self, run: AgentRun) -> float:
        repeated_prior_claims = []
        current_by_text = {self._normalise(claim.text): claim for claim in run.claims}
        for prior_claim in run.prior_turn_claims:
            current_claim = current_by_text.get(self._normalise(prior_claim.text))
            if current_claim is None:
                continue
            repeated_prior_claims.append(
                bool(set(prior_claim.supporting_evidence_ids).intersection(current_claim.supporting_evidence_ids))
                and bool(current_claim.cited_evidence_ids)
            )
        if not repeated_prior_claims:
            return 1.0
        return self._ratio(sum(repeated_prior_claims), len(repeated_prior_claims))

    def _evidence_coverage(self, claims: tuple[Claim, ...], evidence_ids: set[str]) -> float:
        if not evidence_ids:
            return 1.0
        used_evidence_ids = {
            evidence_id for claim in claims for evidence_id in claim.supporting_evidence_ids
        }
        return self._ratio(len(evidence_ids.intersection(used_evidence_ids)), len(evidence_ids))

    def _blended_prohibited_sources(
        self, source_names: set[str], prohibited_combinations: tuple[tuple[str, ...], ...]
    ) -> bool:
        return any(set(combination).issubset(source_names) for combination in prohibited_combinations)

    def _normalise(self, text: str) -> str:
        return " ".join(text.casefold().split())

    def _ratio(self, numerator: int, denominator: int) -> float:
        if denominator == 0:
            return 1.0
        return numerator / denominator
