# Trace corpora for `tracebench` (arena#41)

## This directory's file is a SAMPLE, not real production data

`production-traces.sample.jsonl` is a **recorded-format fixture**: fourteen
hand-shaped traces in the real `TraceRecord` schema, covering every detector
(repeated tool calls, misrouting, step-limit loop, schema violation,
latency/cost spikes, frequent escalation, high variance) plus one clean
trace that mines nothing. It exists so the tests, the demo
(`python -m agenteval_bench.tracebench_demo`), and the scheduled workflow
can prove the miner end to end without a production system.

It is **not** a set of real production incidents. Do not cite it as evidence
about any real agent. The traceability guarantee is structural — every case
the miner emits carries its trace id, incident id, pattern, and evidence —
not a claim about this file's contents.

## Pointing the miner at real traces

Export your production traces as JSONL, one `TraceRecord` per line:

```json
{"trace_id": "tr_9f31", "incident_id": "inc-2026-118", "agent": "support-agent",
 "tool_calls": [{"name": "search_docs"}, {"name": "search_docs"}],
 "steps": 6, "step_limit": 10, "duration_ms": 12000, "cost_usd": 0.20,
 "outcome": "failure", "schema_violations": 0,
 "expected_output": "escalated to a human with the docs link",
 "error": "search returned no results"}
```

Field notes:

- `tool_calls[].expected` is the tool the planner intended, when your
  tracing records it. Misrouting is detectable only where intent is
  recorded; the miner refuses to guess.
- `expected_output` is required: it is the oracle the mined case is scored
  against. A trace without an oracle is evidence, not a benchmark case, and
  is refused at the boundary.
- `incident_id` is the production incident the trace belongs to. Every
  mined case carries it, so a case is auditable back to the incident.

Then run:

```sh
agenteval-bench tracebench mine --corpus <your-export.jsonl> \
  --owner <team-or-person> --cadence weekly --out <dir>
agenteval-bench tracebench regress --corpus <your-export.jsonl> \
  --store <benchmark-store> --suite <name> --version <new-id> \
  [--prev <base-version-id>] \
  --candidate-outputs <case-id-to-output.json> \
  --owner <team-or-person> --cadence weekly --out <dir>
```
