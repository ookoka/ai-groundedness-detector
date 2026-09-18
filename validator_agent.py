from __future__ import annotations

from typing import Any

from groundedness_agent_wrapper import validate_agent_output


class ValidatorAgent:
    """A small agent that turns a raw answer into detector-ready payloads.

    For a hackathon, this is the simplest way to simulate a validator agent in a
    multi-agent architecture without tightly coupling to a specific platform.
    """

    def __init__(self, required_source: str | None = None, expected_tool: str | None = None):
        self.required_source = required_source
        self.expected_tool = expected_tool

    def build_payload(
        self,
        prompt: str,
        answer: str,
        evidence: list[dict[str, Any]],
        observed_route: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        prior_turn_claims: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        claims = []
        for index, sentence in enumerate(_split_sentences(answer), start=1):
            claim_id = f"c{index}"
            supporting_ids = []
            for item in evidence:
                if item.get("text", "").casefold() in sentence.casefold() or sentence.casefold() in item.get("text", "").casefold():
                    supporting_ids.append(item["id"])
            claims.append(
                {
                    "id": claim_id,
                    "text": sentence,
                    "supporting_evidence_ids": supporting_ids,
                    "cited_evidence_ids": supporting_ids,
                }
            )

        payload = {
            "prompt": prompt,
            "answer": answer,
            "observed_route": observed_route,
            "evidence": evidence,
            "claims": claims,
            "required_source": self.required_source,
            "expected_tool": self.expected_tool,
            "grounding_required": True,
            "require_citations": True,
            "refuse_when_no_approved_source": True,
            "tool_calls": tool_calls or [],
            "prior_turn_claims": prior_turn_claims or [],
        }
        return payload

    def validate(
        self,
        prompt: str,
        answer: str,
        evidence: list[dict[str, Any]],
        observed_route: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        prior_turn_claims: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        payload = self.build_payload(
            prompt=prompt,
            answer=answer,
            evidence=evidence,
            observed_route=observed_route,
            tool_calls=tool_calls,
            prior_turn_claims=prior_turn_claims,
        )
        return validate_agent_output(payload)


def _split_sentences(text: str) -> list[str]:
    if not text.strip():
        return []
    normalized = text.replace("?", ".").replace("!", ".")
    parts = []
    current = ""
    for char in normalized:
        current += char
        if char == ".":
            cleaned = current.strip()
            if cleaned:
                parts.append(cleaned)
            current = ""
    if current.strip():
        parts.append(current.strip())
    return parts
