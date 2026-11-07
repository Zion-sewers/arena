# JUDGES.md — judge decorrelation: HOST != FAMILY

Executable form: `src/agenteval_bench/judges.py`, demo
`src/agenteval_bench/judges_demo.py`, tests `tests/test_judges.py`,
CLI `agenteval-bench judges --registry <file.json>`.

## The trap

Judges whose findings are compared must come from different MODEL FAMILIES.
Two hosts serving the same weights carry zero information: nvidia-nim and
openrouter are both multi-vendor resellers, and both can serve deepseek,
qwen, and llama. Picking "nvidia" and "openrouter" as two independent
reviewers can silently produce the same model twice, and the gate would
report agreement that carries no information. Provider-name diversity is
NOT decorrelation.

Source: claude-setup `tools/review/actors.json` catalog A.5 (the
`_decorrelation_contract`) and `tools/review/allocate.py`, which this module
adapts into arena's typed stdlib-only style. The semantics are the source's;
the code is arena's.

## Rules

- Allocation is keyed on `model_family`, never on `host` or provider name.
  Same-family pairs are rejected, and the rejection is named in the report
  (`rejected`, with the reason), not silently dropped.
- `VARIES-BY-MODEL` is not a family. A reseller whose served model varies
  can never fill a second slot, because its family cannot be shown to differ
  from a partner it may itself be serving. It may be picked only as a single
  opinion when nothing else is available, and the report says so.
- Unreachable judges appear in the report as explicit holes (`holes`, with
  the judge, host, family, and reason) — never picked, never silently
  dropped. Reachability is declared in the registry (`measured` /
  `declared` / `unreachable`), or measured live when the caller supplies a
  `probe`. A probe that returns False makes a hole; a probe that raises
  makes a hole naming the exception. A probe that returns True stamps the
  judge's effective reachability as `measured` in the report, and the
  override is noted when it differs from the declaration, so a picked
  judge is never labelled unreachable.
- A thin allocation (fewer distinct reachable families than `want`) says so
  in `notes` and reports `satisfied: false`. It never silently returns a
  short list as if it were full.
- The report accounts for every registry judge exactly once — `picked`,
  `holes`, `rejected`, or `surplus` (reachable judges with distinct
  families the quota did not need). A same-family rejection and a
  VARIES-BY-MODEL exclusion are both typed `rejected` entries, not prose.
- Allocation is deterministic: candidates are ordered by (reachability
  rank, id), so the same registry always allocates the same way.

## Registry format

A JSON document: a mapping with a `judges` list, or a bare list. Each entry
carries `id`, `host`, `model_family` (non-empty strings, required),
`model` and `reachable` (optional; defaults `""` and `"declared"`).
The loader fails closed: duplicate ids, unknown keys, a missing required
field, or a `reachable` outside the three values all raise `ValueError`,
so a typo cannot silently become a judge with no declared family.

```json
{"judges": [
  {"id": "deepseek-nim", "host": "nvidia-nim",
   "model_family": "deepseek-r1", "model": "deepseek-ai/DeepSeek-R1",
   "reachable": "measured"},
  {"id": "codex", "host": "openai", "model_family": "openai-gpt",
   "model": "gpt-5", "reachable": "unreachable"}
]}
```

## CLI

```bash
agenteval-bench judges --registry registry.json --want 2 [--json] [--out report.json]
```

Exit 0 = the allocation satisfied the request; exit 2 = thin allocation;
exit 1 = unusable registry. The report JSON carries `picked`, `families`,
`holes`, `rejected`, and `notes`. The CLI runs no probe: reachability in
its output is as declared in the registry, and the human-readable output
says so. A registry that claims `"reachable": "measured"` without any
probe behind it is an unverified claim, not a measurement. A copyable
example registry ships at `examples/judges-registry.json`.

## Coverage boundary

This module allocates judges; it does not call them. Arena has no live
judge transport (README: "LLM-as-judge is planned, not implemented"), so
reachability is either declared in the registry or measured by a
caller-supplied probe — the module itself makes no network call. The
alt-test judge labels (`src/alt_test/`) are recorded data scored after the
fact; nothing here wires allocation into alt-test yet. That wiring (allocate
the judges, then run alt-test on their recorded labels) is a named
follow-up, not part of this unit.
