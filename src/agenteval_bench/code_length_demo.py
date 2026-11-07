"""Seeded end-to-end demo of code-length selection + capacity meter (arena#50).

Builds the traps the acceptance criteria forbid and shows the module
refusing them:

1. Parameter count is not complexity: a 10M-parameter routing table
   of repeated rules compresses shorter than a 100k-parameter table
   of random weights, at equal held-out fit — the shorter code wins
   despite 100x the parameters.
2. The equal-performance rule: a version that fits slightly better
   (inside the band) but carries a much longer code loses to the more
   compressible version, and the report names the fit deficit against
   the code saved.
3. The two rules differ exactly where the trade is large: a tiny-code
   version with a much worse fit wins the pure MDL rule but NOT the
   equal-performance rule, which refuses to trade a large fit loss
   for code.
4. The capacity meter: a synthetic training loop whose capacity grows
   every checkpoint while held-out log-loss stalls after step 2 — the
   loop stops at the expected step, and the counter-case (held-out
   still improving) keeps going.
5. The measurement itself: key order cannot change a description
   length, and a repetitive artifact measures shorter than a random
   one of the same size.
"""

from __future__ import annotations

import random

from agenteval_bench.code_length import (
    CapacityReading,
    ModelVersion,
    assess_capacity,
    measure_description_bits,
    select_by_total_length,
    select_version,
)


def _expect(condition: bool, label: str) -> None:
    if not condition:
        raise AssertionError(label)
    print(f"  ok    {label}")


def _repeated_routing_table(n_rules: int) -> dict[str, object]:
    """A big routing table made of identical rules: compressible."""
    return {"rules": [{"match": "summarize", "route": "local"} for _ in range(n_rules)]}


def _random_weights(rng: random.Random, n: int) -> dict[str, object]:
    """A table of random weights: incompressible."""
    return {f"w-{i:05d}": rng.random() for i in range(n)}


def main() -> int:
    print("Code-length selection + capacity meter demo (arena#50)")

    # 1. Parameter count is not complexity: same entry count, and the
    # 10M-parameter version's artifact compresses ~100x shorter than
    # the 100k-parameter version's, because its rules repeat.
    rng = random.Random(7)
    big_repeated = _repeated_routing_table(10_000)
    small_random = _random_weights(rng, 10_000)
    big_bits = measure_description_bits(big_repeated)
    small_bits = measure_description_bits(small_random)
    _expect(
        big_bits < small_bits,
        f"10k identical rules ({big_bits} bits) compress shorter than "
        f"10k random weights ({small_bits} bits)",
    )
    report = select_version(
        [
            ModelVersion(
                name="router-big-repeated",
                description_bits=big_bits,
                held_out_n=500,
                held_out_mean_log_loss_bits=0.40,
                parameter_count=10_000_000,
            ),
            ModelVersion(
                name="router-small-random",
                description_bits=small_bits,
                held_out_n=500,
                held_out_mean_log_loss_bits=0.40,
                parameter_count=100_000,
            ),
        ]
    )
    _expect(
        report.winner.name == "router-big-repeated",
        "the 100x-parameter version wins: its code is shorter",
    )
    _expect(report.fit_deficit_bits == 0.0, "equal fit, so no deficit to trade")

    # 2. Equal performance, pick the more compressible.
    versions = [
        ModelVersion(
            name="judge-long-code",
            description_bits=9000,
            held_out_n=500,
            held_out_mean_log_loss_bits=0.395,
        ),
        ModelVersion(
            name="judge-short-code",
            description_bits=1200,
            held_out_n=500,
            held_out_mean_log_loss_bits=0.400,
        ),
        ModelVersion(
            name="judge-tiny-code",
            description_bits=100,
            held_out_n=500,
            held_out_mean_log_loss_bits=0.450,
        ),
    ]
    report = select_version(versions, equivalence_band_bits=0.01)
    _expect(
        report.winner.name == "judge-short-code",
        "the slightly worse fit wins inside the band: its code is shorter",
    )
    _expect(
        abs(report.fit_deficit_bits - 0.005) < 1e-9,
        f"the report names the 0.005 bits/item deficit ({report.fit_deficit_bits})",
    )
    _expect(report.code_saving_bits == 7800.0, "the report names the 7800 saved bits")
    print(f"        {report.rationale}")

    # 3. The two rules differ exactly where the trade is large.
    mdl = select_by_total_length(versions)
    _expect(
        mdl.winner.name == "judge-tiny-code",
        "unbanded, the tiny code wins: it costs 500 x 0.055 = 27.5 bits "
        "of fit and saves 1100 code bits, total 325 vs 1400",
    )
    _expect(
        report.winner.name != mdl.winner.name,
        "the band rule refuses the large fit loss the pure rule accepts",
    )

    # 4. The capacity meter stops a training loop that only grows.
    # Capacity grows 100 bits per checkpoint; held-out improves to
    # step 2, then stalls flat. Patience 3: the meter first has a
    # baseline at step 3 (window 1-3, best before = step 0) and must
    # still be continuing (improvement 0.40 -> 0.38 beats the bar);
    # by step 5 the window 3-5 shows growth with no improvement.
    readings = [
        CapacityReading(step=0, capacity_bits=1000,
                        held_out_mean_log_loss_bits=0.40,
                        train_mean_log_loss_bits=0.38),
        CapacityReading(step=1, capacity_bits=1100,
                        held_out_mean_log_loss_bits=0.39,
                        train_mean_log_loss_bits=0.36),
        CapacityReading(step=2, capacity_bits=1200,
                        held_out_mean_log_loss_bits=0.38,
                        train_mean_log_loss_bits=0.34),
        CapacityReading(step=3, capacity_bits=1300,
                        held_out_mean_log_loss_bits=0.38,
                        train_mean_log_loss_bits=0.32),
        CapacityReading(step=4, capacity_bits=1400,
                        held_out_mean_log_loss_bits=0.38,
                        train_mean_log_loss_bits=0.30),
        CapacityReading(step=5, capacity_bits=1500,
                        held_out_mean_log_loss_bits=0.38,
                        train_mean_log_loss_bits=0.28),
    ]
    stopped_at: int | None = None
    for i in range(1, len(readings) + 1):
        verdict = assess_capacity(readings[:i])
        if verdict.decision == "stop":
            stopped_at = readings[i - 1].step
            print(f"        stop at step {stopped_at}: {verdict.reason}")
            break
    _expect(stopped_at == 5, f"the loop stops at step 5 (got {stopped_at})")
    verdict = assess_capacity(readings)
    _expect(verdict.capacity_growth_bits == 200, "the verdict names 200 bits of growth")
    _expect(
        abs(verdict.max_generalization_gap_bits - 0.10) < 1e-9,
        "the verdict names the widening gap (0.38 - 0.28 = 0.10)",
    )

    # 4b. Counter-case: held-out still improving, capacity growth is fine.
    improving = [
        CapacityReading(step=s, capacity_bits=1000 + 100 * s,
                        held_out_mean_log_loss_bits=0.40 - 0.02 * s,
                        train_mean_log_loss_bits=0.38 - 0.02 * s)
        for s in range(6)
    ]
    verdict = assess_capacity(improving)
    _expect(verdict.decision == "continue", "growth with gain keeps training")

    # 5. The measurement itself: canonical and compressive.
    table = {"b": 1, "a": [1, 2, 3], "c": {"z": "x", "y": "w"}}
    reordered = {"c": {"y": "w", "z": "x"}, "a": [1, 2, 3], "b": 1}
    _expect(
        measure_description_bits(table) == measure_description_bits(reordered),
        "key order cannot change a description length",
    )
    repeated = {"k": "aaaaaaaaaa" * 100}
    randomish = {"k": "".join(rng.choice("abcdefgh") for _ in range(1000))}
    _expect(
        measure_description_bits(repeated) < measure_description_bits(randomish),
        "a repetitive artifact measures shorter than a random one",
    )

    print("demo checks passed: code length selects, capacity stops, measurement honest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
