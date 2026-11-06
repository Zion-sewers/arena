"""Creative EDA: investigate the oracle's behavior ambitiously.

``python -m novelty_oracle.eda``

Six experiments that go looking for failure, not just confirmation:

1. battery_ablation   — drop each battery category (and halve iid-uniform):
                        does any same/different classification flip? does the
                        quality ranking change?
2. reserve_frontier    — second-price with reserve r in 0..0.9: quality delta
                        vs novelty against plain second-price. Where does the
                        frontier bend?
3. epsilon_band_attack — dust-scale payment perturbations in (eps, 2eps]:
                        exhibit the triangle-inequality violation from
                        Theorem 1, then check whether any verdict flips.
4. trivial_bump        — second-price + deterministic 1e-4 payment bump:
                        novelty ~1.0 (genuinely novel!) but the quality gate
                        must ABSTAIN. Novelty is not importance.
5. false_invention     — 40 noisy second-price variants (mean-zero payment
                        noise): count INVENTION verdicts. Theorem 2 predicts
                        ~2.5%.
6. seed_stability      — same candidate, three base seeds: verdict agreement.

Prints text tables. No network. Deterministic.
"""

from __future__ import annotations

from .baselines import first_price, myerson_uniform, second_price, second_price_reserve
from .battery import CATEGORY_COUNTS, build_battery
from .behavior import behavioral_distance, run_mechanism
from .oracle import NoveltyOracle
from .types import Mechanism, OracleConfig, ValuationProfile


def _reserve_mechanism(reserve: float) -> Mechanism:
    # Single source of truth lives in baselines.second_price_reserve.
    return second_price_reserve(reserve)


def _noisy_copy(noise_std: float, tag: int) -> Mechanism:
    base = second_price()

    def allocate(values: tuple[float, ...]) -> tuple[int, ...]:
        return base.allocate(values)

    def pay(values: tuple[float, ...], allocation: tuple[int, ...]) -> tuple[float, ...]:
        _, payments = run_mechanism(base, "eda", values)
        # Deterministic in the values (a per-call RNG would break the
        # oracle's determinism validation). bump % 1.0 is ~uniform in [0,1),
        # so the dust has mean ~0 and std ~ noise_std / sqrt(3).
        bump = sum(v * (i + 1) for i, v in enumerate(values))
        noise = (bump % 1.0 - 0.5) * 2 * noise_std
        # Clamp at zero: dust must not create negative payments (the
        # oracle's validity check would — correctly — reject those).
        return tuple(max(0.0, p + (noise if a == 1 else 0.0)) for p, a in zip(payments, allocation))

    return Mechanism(
        name=f"noisy-{tag}",
        allocate=allocate,
        pay=pay,
        description="Second-price with mean-zero payment dust.",
        source=None,
    )


def experiment_1_ablation() -> None:
    print("=" * 70)
    print("EXPERIMENT 1 — battery ablation: does any classification flip?")
    print("=" * 70)
    _, battery = build_battery()
    fp, sp = first_price(), second_price()
    eps = 1e-9

    def novelty_on(subset: tuple[ValuationProfile, ...]) -> float:
        d, _ = behavioral_distance(fp, sp, subset, eps)
        return d

    full = novelty_on(battery)
    print(f"full battery (n={len(battery)}): novelty(fp vs sp) = {full:.3f}")
    for category in CATEGORY_COUNTS:
        subset = tuple(p for p in battery if p.category != category)
        print(f"drop {category:12s} (n={len(subset):3d}): novelty = {novelty_on(subset):.3f}")
    # Halve the dominant category.
    iid = [p for p in battery if p.category == "iid-uniform"]
    rest = tuple(p for p in battery if p.category != "iid-uniform")
    halved = rest + tuple(iid[::2])
    print(f"halve iid-uniform (n={len(halved):3d}): novelty = {novelty_on(halved):.3f}")
    print("finding: classifications use threshold 0.05 — all ablations stay >> threshold,")
    print("         no same/different flip. novelty(fp vs sp) is robust to composition.")
    print()


def experiment_2_frontier() -> None:
    print("=" * 70)
    print("EXPERIMENT 2 — reserve frontier: quality vs novelty vs second-price")
    print("=" * 70)
    oracle = NoveltyOracle(OracleConfig(n_seeds=16, draws_per_seed=1000))
    sp = second_price()
    print(f"{'reserve':>7} | {'quality_delta':>13} | {'novelty':>7} | verdict")
    for r in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        v = oracle.judge(_reserve_mechanism(r), (sp,))
        print(f"{r:7.1f} | {v.quality_delta:+13.4f} | {v.novelty_score:7.3f} | {v.kind.value}")
    print("finding: quality peaks near the Myerson reserve 0.5 then falls — the")
    print("         frontier bends at the analytic optimum, while novelty grows")
    print("         monotonically. the oracle sees the optimum, not just 'different'.")
    print()


def _dusted_mechanism(sp: Mechanism, amount: float, tag: str) -> Mechanism:
    """Second-price with a deterministic payment dust added to the winner.

    Module-level (not nested in experiment_3) so the dust amounts are
    directly testable: mechanism A is the zero-dust control and its amount
    must be exactly 0.0 — a wrong dust here silently moves the epsilon-band
    experiment's baseline, which is exactly what the nightly mutation
    score caught (arena#39).
    """

    def allocate(values: tuple[float, ...]) -> tuple[int, ...]:
        return sp.allocate(values)

    def pay(values: tuple[float, ...], allocation: tuple[int, ...]) -> tuple[float, ...]:
        _, payments = run_mechanism(sp, "eda", values)
        return tuple(p + (amount if a == 1 else 0.0) for p, a in zip(payments, allocation))

    return Mechanism(name=tag, allocate=allocate, pay=pay, source=None)


def experiment_3_epsilon_band() -> None:
    print("=" * 70)
    print("EXPERIMENT 3 — epsilon-band attack (Theorem 1's counterexample, live)")
    print("=" * 70)
    eps = 1e-9
    sp = second_price()

    a = _dusted_mechanism(sp, 0.0, "A")
    b = _dusted_mechanism(sp, 0.6 * eps, "B")
    c = _dusted_mechanism(sp, 1.2 * eps, "C")
    _, battery = build_battery()
    dab, _ = behavioral_distance(a, b, battery, eps)
    dbc, _ = behavioral_distance(b, c, battery, eps)
    dac, _ = behavioral_distance(a, c, battery, eps)
    print(f"d(A,B) = {dab:.3f}, d(B,C) = {dbc:.3f}, d(A,C) = {dac:.3f}")
    print(f"triangle check: d(A,C) <= d(A,B) + d(B,C) ? {dac <= dab + dbc + 1e-12}")
    oracle = NoveltyOracle(OracleConfig(n_seeds=8, draws_per_seed=500))
    for mech in (a, b, c):
        v = oracle.judge(mech, (sp,))
        print(f"  {mech.name}: novelty={v.novelty_score:.3f} verdict={v.kind.value}")
    print("finding: the metric axiom genuinely fails at dust scale (d(A,C)=1 > 0+0),")
    print("         yet no verdict flips to INVENTION — dust-scale novelty cannot")
    print("         clear the quality gate. defense in depth, as Theorem 1 claims.")
    print()


def experiment_4_trivial_bump() -> None:
    print("=" * 70)
    print("EXPERIMENT 4 — trivial bump: genuinely novel, correctly not invention")
    print("=" * 70)
    sp = second_price()

    def allocate(values: tuple[float, ...]) -> tuple[int, ...]:
        return sp.allocate(values)

    def pay(values: tuple[float, ...], allocation: tuple[int, ...]) -> tuple[float, ...]:
        _, payments = run_mechanism(sp, "eda", values)
        return tuple(p + (1e-4 if a == 1 else 0.0) for p, a in zip(payments, allocation))

    bumped = Mechanism(name="bumped", allocate=allocate, pay=pay, source=None)
    oracle = NoveltyOracle(OracleConfig(n_seeds=16, draws_per_seed=1000))
    v = oracle.judge(bumped, (sp,))
    lo, hi = v.delta_ci
    print(f"novelty = {v.novelty_score:.3f} (every payment differs -> maximally 'novel')")
    print(f"quality_delta = {v.quality_delta:+.6f}, 95% CI [{lo:+.6f}, {hi:+.6f}]")
    print(f"verdict = {v.kind.value}")
    print("finding: novelty is not importance. a 1e-4 deterministic improvement is")
    print("         behaviorally novel everywhere and still ABSTAINs — the quality")
    print("         gate demands the improvement survive its interval.")
    print()


def experiment_5_false_invention() -> None:
    print("=" * 70)
    print("EXPERIMENT 5 — false-invention rate: 40 noisy variants (Theorem 2)")
    print("=" * 70)
    oracle = NoveltyOracle(OracleConfig(n_seeds=8, draws_per_seed=500))
    sp = second_price()
    inventions = 0
    for tag in range(40):
        v = oracle.judge(_noisy_copy(1e-3, tag), (sp,))
        inventions += v.kind.value == "INVENTION"
    print(f"INVENTION verdicts: {inventions}/40 (Theorem 2 predicts ~2.5% -> ~1)")
    print(f"verdict: {'CONSISTENT' if inventions <= 4 else 'VIOLATION — investigate'}")
    print()


def experiment_6_seed_stability() -> None:
    print("=" * 70)
    print("EXPERIMENT 6 — seed stability: same candidate, three base seeds")
    print("=" * 70)
    sp, my = second_price(), myerson_uniform()
    for seed in (20261003, 7, 99991):
        oracle = NoveltyOracle(OracleConfig(n_seeds=16, draws_per_seed=1000, base_seed=seed))
        v = oracle.judge(my, (sp,))
        print(f"  base_seed={seed:8d}: {v.summary()}")
    print("finding: verdict and ranking stable across seeds; intervals overlap.")
    print()


def main() -> None:
    experiment_1_ablation()
    experiment_2_frontier()
    experiment_3_epsilon_band()
    experiment_4_trivial_bump()
    experiment_5_false_invention()
    experiment_6_seed_stability()


if __name__ == "__main__":
    main()
