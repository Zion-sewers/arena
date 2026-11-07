# COVERAGE.md — coverage requirements before learning or ranking

Executable form: `src/agenteval_bench/coverage.py`, demo
`src/agenteval_bench/coverage_demo.py`, tests `tests/test_coverage.py`,
CLI `agenteval-bench alt-test --data <file.jsonl> [--min-cell-n 50]
[--min-total 50]`.

## The trap

A judge labels 20 items, wins every comparison it enters, and tops the
leaderboard at omega 1.0, rho 1.0. It was never measured: 20 shared items
per (judge, annotator) pair is not a ranking, it is a small sample that
happened to look perfect. The ungated leaderboard cannot tell the two
apart — it sorts by the point estimate and drops the sparse cells from
the denominator without a word. A leaderboard over tiny samples is NOT
a ranking.

Source: Perplexity sample-size research (catalog B-47) — INSPIRED. The
binding constraint is per-(state, action) coverage, not total count:
~50 observations per cell minimum. Sparse cells are masked out.
Conservative fallback where data is thin.

## Rules

- A cell is one (state, action) pair. In the alt-test bridge the state
  is the left-out annotator (the reference population) and the action is
  the judge under test; observations are the items the pair shares
  (`AnnotatorComparison.n`). The generic entry point
  `assess_observations` takes raw per-cell counts for learning gates
  that are not alt-test.
- The per-cell floor is 50 by default (`CoveragePolicy.min_per_cell`),
  the total floor 50 (`min_total`, aligned with
  `AltTestConfig.min_items`). Both are overridable on the CLI
  (`--min-cell-n`, `--min-total`); a non-default policy is carried on
  the board, so a ranking minted under a lax floor is distinguishable
  from a default-policy one.
- A judge is rankable iff EVERY one of its cells is covered. One sparse
  cell masks the judge from the ranking, whatever its omega. Masked
  judges are still SHOWN — with their diagnostic score and thinnest cell
  named — never silently dropped.
- Sparse cells fall back to a conservative default: a cell's reported
  rate is the Wilson 95% interval LOWER bound on its win rate
  (`decision.wilson_interval`, established), never the point estimate.
  Thin data is punished by the interval width, not by a hand-picked
  penalty: 5/5 wins falls back to ~0.57, 50/50 to ~0.93, 120/120 to
  ~0.97. A ranked judge's score is the mean lower bound over its
  COVERED cells only; masked cells never enter a ranked score.
- A leaderboard with no rankable judge is not a ranking. `require_ranking`
  refuses with a typed `RankingRefused` naming every masked judge and
  its thinnest cell; the `alt-test` CLI prints the refusal and exits 1.
- The coverage record cannot be forged: `CellCoverage` derives its
  status from n against the floor at construction, and
  `RankedLeaderboard` refuses a `ranked` entry that is not rankable or
  a `masked` entry that is. Wins are recovered exactly from the
  report's (rho_f, n) — rho_f is the mean of n binary wins — and a cell
  whose wins cannot be recovered is a corrupt report that refuses
  loudly instead of guessing.
- `run_alt_test` is untouched: same verdicts, same omega/rho, same
  fail-closed `InsufficientCoverage`. The coverage layer consumes the
  report and never rewrites it.

## Defaults

`min_per_cell` 50 is the source floor (~50 observations per cell).
`min_total` 50 aligns with alt-test's `min_items`. A dataset below the
total floor is not a ranking even when every cell in it happens to be
covered.

## What it does not do

- It does not change the alt-test statistics: omega, rho, p-values,
  and FDR control are computed exactly as before. It gates which rows
  of the leaderboard count as a ranking.
- It does not mask sparse bins in `calibration.py`: a bin below the
  floor is UNCALIBRATED there today, but its observations still enter
  the ECE diagnostic. Applying this module's cell machinery to
  calibration bins is a named follow-up, not a claimed consumer.
