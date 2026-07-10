"""
Statistical and behavioral tests for fast_thompson.

The core idea: a Thompson sampler over Beta posteriors must produce scores
distributed as Beta(successes + 1, failures + 1). We verify this by drawing
many samples from fast_thompson and comparing them against numpy's
np.random.beta using two-sample Kolmogorov-Smirnov tests and moment checks.

Run with:  pytest tests/ -v
"""

import numpy as np
import pytest
from scipy import stats

import fast_thompson as ft

N_SAMPLES = 20_000
KS_PVALUE_MIN = 1e-3  # reject only on very strong evidence of mismatch
RNG = np.random.default_rng(12345)


def draw_ft_scores(successes, failures, n, item_id=1, seed=987654321):
    """Draw n Beta samples for a single item via repeated top-1 sampling."""
    sampler = ft.Thompson([ft.Item(item_id, successes, failures)], seed=seed)
    return np.array([sampler.sample(1, [])[0].score for _ in range(n)])


def draw_numpy_scores(successes, failures, n):
    return RNG.beta(successes + 1.0, failures + 1.0, size=n)


# ---------------------------------------------------------------------------
# Distribution correctness: exact (Marsaglia-Tsang) region, counts < 30
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("s,f", [(0, 0), (2, 5), (10, 3), (25, 25)])
def test_exact_region_matches_numpy_beta(s, f):
    ours = draw_ft_scores(s, f, N_SAMPLES)
    theirs = draw_numpy_scores(s, f, N_SAMPLES)

    ks = stats.ks_2samp(ours, theirs)
    assert ks.pvalue > KS_PVALUE_MIN, (
        f"Beta({s + 1},{f + 1}): KS stat={ks.statistic:.4f}, p={ks.pvalue:.2e}"
    )

    # Sanity on moments too
    assert abs(ours.mean() - theirs.mean()) < 0.01
    assert abs(ours.std() - theirs.std()) < 0.01


def test_uniform_prior_is_uniform():
    """Item with 0 successes / 0 failures has a Beta(1,1) == U(0,1) posterior."""
    scores = draw_ft_scores(0, 0, N_SAMPLES)
    ks = stats.kstest(scores, "uniform")
    assert ks.pvalue > KS_PVALUE_MIN
    assert 0.0 <= scores.min() and scores.max() <= 1.0


# ---------------------------------------------------------------------------
# Distribution correctness: normal-approximation region, large counts
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("s,f", [(500, 500), (1000, 800)])
def test_approx_region_matches_numpy_beta_symmetric(s, f):
    """Near-symmetric, high-mass posteriors: the normal approximation should be
    statistically indistinguishable from the true Beta even under KS."""
    ours = draw_ft_scores(s, f, N_SAMPLES)
    theirs = draw_numpy_scores(s, f, N_SAMPLES)
    ks = stats.ks_2samp(ours, theirs)
    assert ks.pvalue > KS_PVALUE_MIN, (
        f"Beta({s + 1},{f + 1}): KS stat={ks.statistic:.4f}, p={ks.pvalue:.2e}"
    )


@pytest.mark.parametrize("s,f", [(2000, 300), (300, 2000)])
def test_approx_region_moments_match_numpy_beta(s, f):
    """Skewed but still within the isApprox threshold: check mean/std closely
    (the approximation is normal, so tiny skew differences are expected)."""
    a, b = s + 1.0, f + 1.0
    true_mean = a / (a + b)
    true_std = np.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))

    ours = draw_ft_scores(s, f, N_SAMPLES)
    theirs = draw_numpy_scores(s, f, N_SAMPLES)

    # Both should be close to analytic moments and to each other
    assert abs(ours.mean() - true_mean) < 3 * true_std / np.sqrt(N_SAMPLES) * 5
    assert abs(ours.mean() - theirs.mean()) < 0.005
    assert abs(ours.std() - theirs.std()) < 0.005
    assert np.all((ours >= 0.0) & (ours <= 1.0))


def test_exact_vs_approx_boundary_consistency():
    """An item just below and just above the exact/approx border should give
    almost identical distributions — the switch must not create a jump."""
    below = draw_ft_scores(28, 28, N_SAMPLES)   # exact path (alpha,beta < 30)
    above = draw_ft_scores(31, 31, N_SAMPLES)   # approx path
    # Compare against numpy for each
    for scores, (s, f) in [(below, (28, 28)), (above, (31, 31))]:
        theirs = draw_numpy_scores(s, f, N_SAMPLES)
        ks = stats.ks_2samp(scores, theirs)
        assert ks.pvalue > KS_PVALUE_MIN, f"({s},{f}) p={ks.pvalue:.2e}"


# ---------------------------------------------------------------------------
# Thompson-sampling behavior: top-k selection frequencies vs numpy simulation
# ---------------------------------------------------------------------------

def test_topk_selection_frequencies_match_numpy():
    """Simulate the bandit choice: P(item i wins) under fast_thompson should
    match a pure-numpy Thompson sampling simulation."""
    arms = [(1, 18, 12), (2, 15, 15), (3, 12, 18)]  # overlapping posteriors
    n_rounds = 20_000

    sampler = ft.Thompson([ft.Item(*a) for a in arms], seed=2024)
    ft_wins = np.zeros(3)
    for _ in range(n_rounds):
        winner = sampler.sample(1, [])[0].id
        ft_wins[winner - 1] += 1
    ft_freq = ft_wins / n_rounds

    np_wins = np.zeros(3)
    alphas = np.array([s + 1.0 for _, s, _ in arms])
    betas = np.array([f + 1.0 for _, _, f in arms])
    for _ in range(n_rounds):
        draws = RNG.beta(alphas, betas)
        np_wins[np.argmax(draws)] += 1
    np_freq = np_wins / n_rounds

    # Frequencies should agree within a few percentage points
    assert np.all(np.abs(ft_freq - np_freq) < 0.03), (ft_freq, np_freq)
    # And the ordering must be strictly by empirical success rate
    assert ft_freq[0] > ft_freq[1] > ft_freq[2]


def test_topk_returns_requested_count_and_unique_ids():
    items = [ft.Item(i, int(RNG.integers(0, 50)), int(RNG.integers(0, 50)))
             for i in range(100)]
    sampler = ft.Thompson(items, seed=7)
    result = sampler.sample(10, [])
    ids = [r.id for r in result]
    assert len(ids) == 10
    assert len(set(ids)) == 10


def test_topk_actually_returns_highest_scores():
    """The k returned scores must be the k largest among all items."""
    items = [ft.Item(i, 5, 5) for i in range(50)]
    sampler = ft.Thompson(items, seed=11)
    top = sampler.sample(50, [])          # all scores, one full draw
    all_scores = sorted((r.score for r in top), reverse=True)

    sampler2 = ft.Thompson(items, seed=11)  # same seed -> same draw
    top5 = sorted((r.score for r in sampler2.sample(5, [])), reverse=True)
    assert np.allclose(top5, all_scores[:5])


# ---------------------------------------------------------------------------
# Forbidden-set semantics
# ---------------------------------------------------------------------------

def test_forbidden_items_never_returned():
    items = [ft.Item(i, 10, 1) for i in range(20)]
    sampler = ft.Thompson(items, seed=3)
    forbidden = [0, 5, 13]  # includes id 0 (regression: sentinel collision)
    for _ in range(200):
        ids = {r.id for r in sampler.sample(20, forbidden)}
        assert ids.isdisjoint(forbidden)
        assert len(ids) == 20 - len(forbidden)


def test_forbidding_everything_returns_nothing():
    items = [ft.Item(i, 1, 1) for i in range(5)]
    sampler = ft.Thompson(items, seed=4)
    assert sampler.sample(5, [0, 1, 2, 3, 4]) == []


def test_large_forbidden_list_triggering_set_resize():
    n = 1000
    items = [ft.Item(i, 2, 2) for i in range(n)]
    sampler = ft.Thompson(items, seed=5)
    forbidden = list(range(0, n, 2))  # 500 ids, forces hash-set growth
    ids = {r.id for r in sampler.sample(n, forbidden)}
    assert ids == set(range(1, n, 2))


# ---------------------------------------------------------------------------
# Regressions for the SIMD RNG bugs
# ---------------------------------------------------------------------------

def test_no_duplicate_scores_from_simd_lanes():
    """Regression: identical SIMD lane seeding used to make every random value
    appear 4x, yielding duplicate/collapsed scores across items."""
    items = [ft.Item(i, 1, 1) for i in range(256)]  # > 32 -> vectorized path
    sampler = ft.Thompson(items, seed=99)
    scores = [r.score for r in sampler.sample(256, [])]
    # 256 continuous Beta(2,2) draws collide with probability ~0
    assert len(set(scores)) == len(scores)


def test_successive_draws_are_not_repeated():
    """Regression: the ring buffer used to wrap around and replay stale
    random numbers instead of refilling."""
    scores = draw_ft_scores(5, 5, 5_000)
    # No long runs of identical consecutive values
    assert np.all(np.diff(scores) != 0.0)
    # Lag-1 autocorrelation of an i.i.d. stream must be ~0
    r = np.corrcoef(scores[:-1], scores[1:])[0, 1]
    assert abs(r) < 0.05


def test_seed_reproducibility_and_variation():
    items = [ft.Item(i, 3, 4) for i in range(10)]
    a = [(r.id, r.score) for r in ft.Thompson(items, seed=123).sample(10, [])]
    b = [(r.id, r.score) for r in ft.Thompson(items, seed=123).sample(10, [])]
    c = [(r.id, r.score) for r in ft.Thompson(items, seed=124).sample(10, [])]
    assert a == b
    assert a != c


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))