"""Small, dependency-free statistics helpers used by the benchmark suite."""
import math

Z95 = 1.959963984540054


def percentile(values, q):
    """Linear-interpolated percentile, q in [0, 100]."""
    if not values:
        return float("nan")
    data = sorted(values)
    if len(data) == 1:
        return data[0]
    pos = (len(data) - 1) * q / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return data[lo]
    return data[lo] + (data[hi] - data[lo]) * (pos - lo)


def summarize(values):
    """Median, quartiles and spread of a list of samples."""
    n = len(values)
    if n == 0:
        return {"n": 0}
    mean = sum(values) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1)) if n > 1 else 0.0
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "median": percentile(values, 50),
        "q1": percentile(values, 25),
        "q3": percentile(values, 75),
        "min": min(values),
        "max": max(values),
    }


def wilson(successes, n, z=Z95):
    """Wilson score interval for a binomial proportion. Returns (lo, hi)."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mean_ci(values, z=Z95):
    """Normal-approximation confidence interval for a mean. Returns (mean, half_width)."""
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"))
    mean = sum(values) / n
    if n == 1:
        return (mean, float("inf"))
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return (mean, z * math.sqrt(var / n))


def paired_diff(a, b, z=Z95):
    """Mean of (b - a) over paired samples with a normal-approximation CI.

    Used when two runs played the same seeded games, so per-game differences
    isolate changes in decisions from dice luck.
    """
    if len(a) != len(b):
        raise ValueError("paired samples must have equal length")
    return mean_ci([y - x for x, y in zip(a, b)], z)


def two_prop_diff(k1, n1, k2, n2, z=Z95):
    """Difference of proportions (p2 - p1) with an unpooled normal CI."""
    if n1 == 0 or n2 == 0:
        return (float("nan"), float("nan"))
    p1, p2 = k1 / n1, k2 / n2
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    return (p2 - p1, z * se)
