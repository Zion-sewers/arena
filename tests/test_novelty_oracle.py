"""Independent property tests for the Novelty Oracle (Law Article 1).

Each test has an independent oracle (a separately computed expectation),
not a restatement of the implementation:

1. baseline-vs-itself -> ~zero novelty (identity property);
2. behaviorally-identical-but-syntactically-different -> HIGH behavioral
   similarity (the anti-gaming test: "odd syntax without gains" must not
   inflate novelty);
3. genuinely different mechanisms -> high novelty;
4. quality measurement is seed-stable within the reported interval;
5. verdict decision logic at its boundaries (pure function);
6. invalid / non-deterministic mechanisms raise typed errors, never
   silent scores.
"""

from __future__ import annotations

import pytest

from novelty_oracle import (
    NoveltyOracle,
    OracleConfig,
    first_price,
    myerson_uniform,
    second_price,
)
from novelty_oracle.behavior import (
    InvalidOutcome,
    NonDeterministicMechanism,
    ast_identical,
    behavioral_distance,
    validate_mechanism,
)
from novelty_oracle.demo import obfuscated_second_price
from novelty_oracle.heldout import check_ranking, run_heldout
from novelty_oracle.oracle import decide_verdict
from novelty_oracle.types import Mechanism, VerdictKind

SMALL = OracleConfig(n_seeds=8, draws_per_seed=500)


def test_baseline_vs_itself_zero_novelty() -> None:
    """Identity: a baseline scores ~zero novelty against itself."""
    oracle = NoveltyOracle(SMALL)
    sp = second_price()
    dist, _ = behavioral_distance(sp, sp, oracle.battery, oracle.config.epsilon)
    assert dist == 0.0
    assert ast_identical(sp, sp)


def test_obfuscated_copy_high_similarity() -> None:
    """Anti-gaming: syntactic restructuring must not inflate novelty.

    Independent expectation: the obfuscated implementation computes the
    same (allocation, payments) function, so behavioral distance must be
    ~0 even though the ASTs differ.
    """
    oracle = NoveltyOracle(SMALL)
    plain, obf = second_price(), obfuscated_second_price()
    assert not ast_identical(plain, obf)  # syntactically different: precondition
    dist, _ = behavioral_distance(plain, obf, oracle.battery, oracle.config.epsilon)
    assert dist < 0.01, f"obfuscation inflated novelty to {dist}"
    verdict = oracle.judge(obf, (plain,))
    assert verdict.novelty_score < 0.01
    assert verdict.kind == VerdictKind.NOT_INVENTION


def test_genuinely_different_high_novelty() -> None:
    """First-price vs second-price: different payment rule -> high novelty.

    Independent expectation: allocations always agree (same winner) but
    payments differ on ~every profile without exact ties.
    """
    oracle = NoveltyOracle(SMALL)
    fp, sp = first_price(), second_price()
    dist, _ = behavioral_distance(fp, sp, oracle.battery, oracle.config.epsilon)
    assert dist > 0.5, f"expected high novelty, got {dist}"


def test_quality_seed_stable_within_interval() -> None:
    """Two independent seed budgets give overlapping quality CIs."""
    sp = second_price()
    stats_a = (
        NoveltyOracle(OracleConfig(n_seeds=8, draws_per_seed=500, base_seed=111))
        .judge(sp, (first_price(),))
        .evidence.quality.candidate
    )
    stats_b = (
        NoveltyOracle(OracleConfig(n_seeds=8, draws_per_seed=500, base_seed=777))
        .judge(sp, (first_price(),))
        .evidence.quality.candidate
    )
    # CIs must overlap: neither run rules out the other's mean.
    assert stats_a.ci_low <= stats_b.mean_payoff <= stats_a.ci_high
    assert stats_b.ci_low <= stats_a.mean_payoff <= stats_b.ci_high


def test_verdict_boundaries() -> None:
    """Pure decision logic at its boundaries."""
    cfg = OracleConfig()
    # Clearly better + novel -> INVENTION.
    assert decide_verdict(0.1, 0.05, 0.15, 0.5, cfg) == VerdictKind.INVENTION
    # Better but not novel -> NOT_INVENTION (a variant).
    assert decide_verdict(0.1, 0.05, 0.15, 0.01, cfg) == VerdictKind.NOT_INVENTION
    # Novel but worse -> NOT_INVENTION (novelty is not importance).
    assert decide_verdict(-0.1, -0.15, -0.05, 0.9, cfg) == VerdictKind.NOT_INVENTION
    # CI straddles zero + novel -> ABSTAIN, never a guess.
    assert decide_verdict(0.02, -0.01, 0.05, 0.9, cfg) == VerdictKind.ABSTAIN
    # CI straddles zero but NOT novel -> NOT_INVENTION (nothing to abstain
    # about: novelty is decided exactly, so a copy is a copy).
    assert decide_verdict(0.02, -0.01, 0.05, 0.0, cfg) == VerdictKind.NOT_INVENTION
    assert decide_verdict(0.0, -0.01, 0.01, 0.0, cfg) == VerdictKind.NOT_INVENTION


def test_invalid_mechanism_raises_typed_error() -> None:
    """Negative payments raise InvalidOutcome, never a silent score."""
    sp = second_price()

    def bad_pay(values: tuple[float, ...], allocation: tuple[int, ...]) -> tuple[float, ...]:
        return tuple(-1.0 for _ in values)

    bad = Mechanism(name="bad", allocate=sp.allocate, pay=bad_pay)
    with pytest.raises(InvalidOutcome):
        validate_mechanism(bad)


def test_nondeterministic_mechanism_raises_typed_error() -> None:
    """A mechanism that flips a coin per call raises NonDeterministicMechanism."""
    import random

    sp = second_price()
    rng = random.Random(0)

    def flippy_allocate(values: tuple[float, ...]) -> tuple[int, ...]:
        base = sp.allocate(values)
        if rng.random() < 0.5:
            return tuple(0 for _ in values)
        return base

    flippy = Mechanism(name="flippy", allocate=flippy_allocate, pay=sp.pay)
    with pytest.raises(NonDeterministicMechanism):
        validate_mechanism(flippy)


def test_judge_rejects_empty_baselines() -> None:
    oracle = NoveltyOracle(SMALL)
    with pytest.raises(ValueError):
        oracle.judge(second_price(), ())


def test_heldout_accuracy_bar() -> None:
    """The falsifiable criterion: >=90% held-out classification accuracy."""
    result = run_heldout()
    assert result.accuracy >= 0.90, f"held-out accuracy {result.accuracy:.1%}"


def test_ranking_matches_brute_force() -> None:
    """Simulated quality ranking matches brute-force battery ranking."""
    oracle = NoveltyOracle(SMALL)
    assert check_ranking(oracle)


def test_myerson_reserve_beats_plain_second_price_on_iid() -> None:
    """Myerson (1981), empirically: on i.i.d. regular values, the optimal
    reserve (0.5 for U[0,1]) weakly improves on plain second-price.

    Independent expectation: this is the analytic optimum for the
    iid-uniform slice; the test restricts to that slice because the
    mixed battery deliberately includes distributions where the
    uniform-tuned reserve degrades (documented in battery.py).
    """
    from novelty_oracle.behavior import run_mechanism

    oracle = NoveltyOracle(SMALL)
    iid = [p for p in oracle.battery if p.category == "iid-uniform"]
    assert len(iid) > 0
    revenues: dict[str, float] = {}
    for mech in (second_price(), myerson_uniform()):
        total = 0.0
        for profile in iid:
            _, payments = run_mechanism(mech, profile.profile_id, profile.values)
            total += sum(payments)
        revenues[mech.name] = total / len(iid)
    assert revenues["myerson-uniform"] >= revenues["second-price"], revenues


# --- mutation-survivor pins (arena#39 nightly: these mutants survived) ---


def test_battery_category_counts_are_canonical() -> None:
    """The canonical battery is 200 profiles with pinned category counts.

    A negated count (120 -> -120) survived the suite before this test:
    nothing asserted the battery's composition, so the construction
    silently changed shape.
    """
    from novelty_oracle.battery import CATEGORY_COUNTS, build_battery

    assert CATEGORY_COUNTS == {
        "iid-uniform": 120,
        "asymmetric": 30,
        "tie": 10,
        "dominant": 10,
        "zero": 5,
        "skewed-beta": 25,
    }
    assert sum(CATEGORY_COUNTS.values()) == 200
    _, profiles = build_battery()
    assert len(profiles) == 200


def test_empty_evidence_placeholder_is_all_zeros() -> None:
    """The placeholder is the zero baseline: every field must read zero.

    A negated n_seeds (0 -> 1) survived before this test: nothing
    asserted the placeholder's values, so a verdict built on it would
    silently carry one seed's worth of fake evidence.
    """
    from novelty_oracle.types import empty_evidence_placeholder

    ev = empty_evidence_placeholder()
    assert ev.quality.candidate.n_seeds == 0
    assert ev.quality.candidate.draws_per_seed == 0
    assert ev.quality.candidate.mean_payoff == 0.0
    assert ev.quality.quality_delta == 0.0
    assert ev.novelty.novelty_score == 0.0
    assert ev.novelty.n_profiles == 0


def test_dusted_mechanism_zero_dust_pays_second_price() -> None:
    """Mechanism A is the zero-dust control: it must pay exactly
    second-price, and a dusted mechanism must add exactly its dust to
    the winner only.

    A negated zero dust (0.0 -> 1) survived before this test: the dust
    amounts lived in a nested closure nothing asserted, so the
    epsilon-band experiment's baseline silently moved by 1.0.
    """
    from novelty_oracle.behavior import run_mechanism
    from novelty_oracle.eda import _dusted_mechanism

    sp = second_price()
    values = (1.0, 0.5, 0.25)
    allocation = sp.allocate(values)
    _, base_payments = run_mechanism(sp, "eda", values)
    winner = allocation.index(1)

    zero = _dusted_mechanism(sp, 0.0, "A")
    assert zero.pay(values, allocation) == base_payments

    dust = 0.6e-9
    dusted = _dusted_mechanism(sp, dust, "B")
    paid = dusted.pay(values, allocation)
    assert paid[winner] == pytest.approx(base_payments[winner] + dust)
    for i, p in enumerate(base_payments):
        if i != winner:
            assert paid[i] == p


def test_noisy_copy_zero_noise_pays_second_price() -> None:
    """A zero-noise copy must pay exactly second-price: non-winners pay
    the base payment, nothing added.

    The else-branch 0.0 in _noisy_copy's pay (losers add 0.0) survived
    the nightly mutation score before this test: experiment_5's helper
    ran only as a demo, never under pytest, so a negated 0.0 (losers pay
    +1.0) changed the experiment's mechanisms silently.
    """
    from novelty_oracle.behavior import run_mechanism
    from novelty_oracle.eda import _noisy_copy

    sp = second_price()
    values = (1.0, 0.5, 0.25)
    allocation = sp.allocate(values)
    _, base_payments = run_mechanism(sp, "eda", values)
    quiet = _noisy_copy(0.0, 0)
    assert quiet.pay(values, allocation) == base_payments


def test_noisy_copy_nonzero_noise_lands_on_winner() -> None:
    """Nonzero noise must actually land on the winner's payment.

    The zero-noise test above cannot kill the ``a == 1`` -> ``a == -1``
    mutant: with noise 0 the winner pays base either way, so the mutant
    survived the nightly measurement (7/8). With noise_std 0.1 and
    values (1.0, 0.5, 0.25) the dust is exactly +0.05 on the winner
    (bump 2.75 -> 0.75 -> (0.75 - 0.5) * 2 * 0.1), so a mutant that
    never adds noise pays base 0.5 and fails here.
    """
    from novelty_oracle.behavior import run_mechanism
    from novelty_oracle.eda import _noisy_copy

    sp = second_price()
    values = (1.0, 0.5, 0.25)
    allocation = sp.allocate(values)
    _, base_payments = run_mechanism(sp, "eda", values)
    noisy = _noisy_copy(0.1, 0)
    paid = noisy.pay(values, allocation)
    assert paid[0] == pytest.approx(base_payments[0] + 0.05)
    assert paid[1:] == base_payments[1:]
