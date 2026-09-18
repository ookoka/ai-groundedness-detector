# ai-groundedness-detector
An Agent for evaluating the groundedness of LLM responses

## What this detector evaluates

This repository provides dependency-free Python primitives for black-box
groundedness evaluation of Copilot-style agents. The detector treats each answer
as a trace across:

1. user prompt
2. Copilot orchestration
3. knowledge and tool selection
4. evidence utilization
5. final answer and citations

The core evaluator computes these quality signals:

- `grounding_route_correctness`
- `evidence_support_precision`
- `citation_correctness`
- `evidence_coverage`
- `evidence_dependence`
- `grounding_policy_adherence`
- `tool_output_utilization`
- `abstention_correctness`
- `cross_turn_grounding_retention`

## Minimal usage

```python
from ai_groundedness_detector import (
    AgentRun,
    Claim,
    DetectorTrace,
    Evidence,
    GroundednessDetector,
    GroundingPolicy,
)

run = AgentRun(
    trace=DetectorTrace(
        evaluation_run_id="run-20260915-0042",
        agent_id="agent-j-dev",
        agent_version="2.4.1",
        conversation_id="copilot-conversation-id",
        turn_id=3,
        scenario_id="seismic-product-guidance",
        grounding_configuration_version="grounding-v7",
    ),
    prompt="According to the approved guidance, how often does promotion run?",
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

result = GroundednessDetector().evaluate(
    run,
    GroundingPolicy(required_source="Seismic", expected_tool="search_seismic"),
)
print(result.metrics)
```

For evidence-dependence checks, evaluate a full run together with controlled
variants such as `knowledge-removed`, `supporting-document-changed`,
`distractor-added`, and `tool-unavailable`. The evaluator compares claim
behavior across those variants to determine whether the final answer depended on
the configured enterprise evidence rather than merely being supportable by it.

Run the focused tests with:

```bash
python -m unittest
```
