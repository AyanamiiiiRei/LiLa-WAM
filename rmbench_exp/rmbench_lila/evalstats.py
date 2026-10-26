"""Small, dependency-free statistics for success rates (plan step 11).

wilson_ci       95% Wilson interval for k successes in n trials (better than ±1.96·se
                for small n or rates near 0/1)
mcnemar_exact   paired comparison of two policies on the SAME seeds: only discordant
                pairs (one succeeds, the other fails) carry information
"""

from __future__ import annotations

import math


def wilson_ci(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def binom_two_sided_p(k: int, n: int) -> float:
    """Exact two-sided binomial test of k out of n against p = 0.5."""
    if n == 0:
        return 1.0
    pk = [math.comb(n, i) * 0.5 ** n for i in range(n + 1)]
    obs = pk[k]
    return min(1.0, sum(p for p in pk if p <= obs * (1 + 1e-12)))


def mcnemar_exact(a: dict, b: dict) -> dict:
    """a, b: {seed: success_bool} for two policies. Uses seeds present in both."""
    common = sorted(set(a) & set(b))
    a_only = sum(1 for s in common if a[s] and not b[s])
    b_only = sum(1 for s in common if b[s] and not a[s])
    return {"paired_n": len(common), "a_only": a_only, "b_only": b_only,
            "p_value": binom_two_sided_p(min(a_only, b_only), a_only + b_only),
            "rate_a": sum(a[s] for s in common) / len(common) if common else float("nan"),
            "rate_b": sum(b[s] for s in common) / len(common) if common else float("nan")}
