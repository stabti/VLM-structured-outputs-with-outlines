"""Confidence intervals and paired tests for the scaled study (stdlib only, no scipy).

Unit of analysis = the image. A field accuracy is a proportion k/n -> Wilson interval.
Two methods scored on the SAME images are compared with McNemar's exact test on the
discordant pairs (not by eyeballing overlapping CIs).

`uv run stats.py` runs the self-checks.
"""
import math


def wilson(k, n, z=1.96):
    """95% Wilson score interval for k successes out of n -> (lo, hi). (0, 1) if n == 0."""
    if n == 0:
        return 0.0, 1.0
    p, z2 = k / n, z * z
    centre = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar(a_ok, b_ok):
    """Exact two-sided McNemar test on paired booleans (same images, methods A and B).
    Entries where either side is None (GT missing) are skipped.
    -> (only_a, only_b, p): images A got right and B missed, the reverse, and the p-value."""
    pairs = [(a, b) for a, b in zip(a_ok, b_ok) if a is not None and b is not None]
    only_a = sum(a and not b for a, b in pairs)
    only_b = sum(b and not a for a, b in pairs)
    n = only_a + only_b
    if n == 0:
        return only_a, only_b, 1.0
    tail = sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2**n
    return only_a, only_b, min(1.0, 2 * tail)


def fmt(k, n):
    """'19/20 = 95% [76-99]' — count, rate and Wilson CI, for the result tables."""
    if n == 0:
        return "n/a"
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {100 * k / n:.0f}% [{100 * lo:.0f}-{100 * hi:.0f}]"


if __name__ == "__main__":
    lo, hi = wilson(50, 100)
    assert abs(lo - 0.4038) < 1e-3 and abs(hi - 0.5962) < 1e-3, (lo, hi)  # textbook value
    assert wilson(0, 10)[0] == 0.0 and wilson(10, 10)[1] == 1.0 and wilson(0, 0) == (0.0, 1.0)
    lo, hi = wilson(180, 361)  # our test split at p~0.5: half-width ~ +-0.05
    assert abs((hi - lo) / 2 - 0.0516) < 2e-3, (lo, hi)
    assert mcnemar([1] * 10, [0] * 10) == (10, 0, 2 * (1 / 2**10))  # 10 vs 0 -> p = 0.00195
    assert mcnemar([1, 0, 1, 0], [1, 0, 0, 1])[2] == 1.0  # 1 vs 1 discordant -> no evidence
    assert mcnemar([1, None, 0], [0, 1, 0]) == (1, 0, 1.0)  # None pairs skipped
    assert mcnemar([1, 1], [1, 1])[2] == 1.0  # no discordant pairs
    print("self-checks OK;", fmt(19, 20))
