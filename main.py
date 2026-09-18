from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from validator_agent import ValidatorAgent


class EvidenceItem(BaseModel):
    id: str
    source: str
    text: str
    source_type: str = "document"
    citation: str | None = None


class ClaimItem(BaseModel):
    id: str
    text: str
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    cited_evidence_ids: list[str] = Field(default_factory=list)
    contradicted_by_evidence_ids: list[str] = Field(default_factory=list)
    derived_from_tool_call_ids: list[str] = Field(default_factory=list)


class ToolCallItem(BaseModel):
    tool_expected: str | None = None
    tool_selected: str
    arguments_valid: bool = True
    execution_succeeded: bool = True
    result_used_in_answer: bool = False
    claims_supported: list[str] = Field(default_factory=list)
    result: Any = None


class ValidationRequest(BaseModel):
    prompt: str
    answer: str
    observed_route: str | None = None
    required_source: str | None = None
    expected_tool: str | None = None
    evidence: list[EvidenceItem] = Field(default_factory=list)
    claims: list[ClaimItem] = Field(default_factory=list)
    tool_calls: list[ToolCallItem] = Field(default_factory=list)
    refused: bool = False


app = FastAPI(title="Groundedness Validator Agent")
validator = ValidatorAgent(required_source="Seismic", expected_tool="search_seismic")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/validate")
def validate(request: ValidationRequest) -> dict[str, Any]:
    evidence_payload = [
        {
            "id": item.id,
            "source": item.source,
            "text": item.text,
            "source_type": item.source_type,
            "citation": item.citation,
        }
        for item in request.evidence
    ]

    claims_payload = [
        {
            "id": item.id,
            "text": item.text,
            "supporting_evidence_ids": item.supporting_evidence_ids,
            "cited_evidence_ids": item.cited_evidence_ids,
            "contradicted_by_evidence_ids": item.contradicted_by_evidence_ids,
            "derived_from_tool_call_ids": item.derived_from_tool_call_ids,
        }
        for item in request.claims
    ]

    tool_calls_payload = [
        {
            "tool_expected": item.tool_expected,
            "tool_selected": item.tool_selected,
            "arguments_valid": item.arguments_valid,
            "execution_succeeded": item.execution_succeeded,
            "result_used_in_answer": item.result_used_in_answer,
            "claims_supported": item.claims_supported,
            "result": item.result,
        }
        for item in request.tool_calls
    ]

    validator.required_source = request.required_source or validator.required_source
    validator.expected_tool = request.expected_tool or validator.expected_tool

    result = validator.validate(
        prompt=request.prompt,
        answer=request.answer,
        evidence=evidence_payload,
        observed_route=request.observed_route,
        tool_calls=tool_calls_payload,
    )

    verdict = {
        "grounded": result["metrics"]["grounding_policy_adherence"],
        "metrics": result["metrics"],
        "issues": result["issues"],
        "unsupported_claims": result["unsupported_claims"],
        "invalid_citations": result["invalid_citations"],
    }
    return verdict


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
