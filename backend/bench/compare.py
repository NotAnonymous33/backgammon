"""Compare two runs metric by metric and classify each change.

Statuses: improved, regressed, unchanged (including within noise), new,
removed, info (informational metrics are never flagged).
"""
from .stats import paired_diff, two_prop_diff
from .store import sections, summary

DEFAULT_NOISE_FLOOR = 0.10  # for timing metrics recorded without their own floor


def _rel_range(m):
    """Spread of a run's repeated samples relative to its value."""
    spread = m.get("spread") or {}
    value = m.get("value")
    if not value or "min" not in spread:
        return 0.0
    return abs(spread["max"] - spread["min"]) / abs(value)


def _direction(delta, better):
    if delta == 0 or better == "none":
        return "unchanged"
    good = delta > 0 if better == "higher" else delta < 0
    return "improved" if good else "regressed"


def _scores(outcomes):
    return [{"1": 1.0, "0": 0.0, "d": 0.5}.get(o) for o in outcomes]


def compare_metric(base, head, same_host=True):
    row = {k: head.get(k, base.get(k)) for k in ("key", "section", "label", "unit", "kind", "better")}
    bv, hv = base["value"], head["value"]
    row.update(base=bv, head=hv)
    kind = row["kind"]
    try:
        row["delta"] = hv - bv
        row["delta_pct"] = (hv - bv) / abs(bv) if bv else None
    except TypeError:
        row.update(delta=None, delta_pct=None, status="info")
        return row

    if kind == "timing":
        floor = max(base.get("noise_floor", DEFAULT_NOISE_FLOOR), head.get("noise_floor", DEFAULT_NOISE_FLOOR))
        threshold = max(floor, _rel_range(base), _rel_range(head))
        row["threshold_pct"] = threshold
        pct = row["delta_pct"]
        if pct is None or abs(pct) < threshold:
            row["status"] = "unchanged"
        else:
            row["status"] = _direction(row["delta"], row["better"])
        if not same_host:
            row["note"] = "different machines; speed is not comparable"
            row["status"] = "info"
    elif kind == "count":
        row["status"] = _direction(row["delta"], row["better"])
    elif kind == "rate":
        bo, ho = base.get("outcomes"), head.get("outcomes")
        paired = (
            bo and ho and len(bo) == len(ho) and base.get("pairing") == head.get("pairing")
        )
        if paired:
            pairs = [(x, y) for x, y in zip(_scores(bo), _scores(ho)) if x is not None and y is not None]
            diff, half = paired_diff([p[0] for p in pairs], [p[1] for p in pairs])
            row["test"] = f"paired over {len(pairs)} identical-dice games"
        else:
            diff, half = two_prop_diff(bv * base.get("n", 0), base.get("n", 0),
                                       hv * head.get("n", 0), head.get("n", 0))
            row["test"] = "unpaired (different seeds or game counts)"
        row["diff_ci"] = [diff - half, diff + half]
        significant = half == half and abs(diff) > half  # half == half filters NaN
        row["status"] = _direction(diff, row["better"]) if significant else "unchanged"
        if not significant and diff != 0:
            row["note"] = "within 95% confidence interval"
    else:
        row["status"] = "info"
    return row


def compare_runs(base, head):
    same_host = base.get("env", {}).get("host", {}).get("fingerprint") == \
        head.get("env", {}).get("host", {}).get("fingerprint")
    bm = {m["key"]: m for m in base.get("metrics", [])}
    hm = {m["key"]: m for m in head.get("metrics", [])}

    # Only compare sections both runs measured; a partial run is not a removal.
    shared = sections(base) & sections(head)
    rows = []
    for key, m in hm.items():
        if key in bm:
            rows.append(compare_metric(bm[key], m, same_host))
        elif m["section"] in shared:
            rows.append({**_row_stub(m), "base": None, "head": m["value"], "status": "new"})
    for key, m in bm.items():
        if key not in hm and m["section"] in shared:
            rows.append({**_row_stub(m), "base": m["value"], "head": None, "status": "removed"})

    warnings = []
    one_sided = sorted(sections(base) ^ sections(head))
    if one_sided:
        warnings.append(f"Not compared (measured in only one run): {', '.join(one_sided)}.")
    if not same_host:
        warnings.append("Runs come from different machines: speed metrics are shown but not judged.")
    if base.get("config", {}).get("positions") != head.get("config", {}).get("positions"):
        warnings.append("Position suites differ: rules and speed metrics measure different positions.")
    if base.get("config", {}).get("seed") != head.get("config", {}).get("seed"):
        warnings.append("Seeds differ: strength comparisons are unpaired and less sensitive.")
    for label, run in (("base", base), ("head", head)):
        for w in run.get("warnings", []):
            warnings.append(f"{label} {run['id']}: {w}")

    counts = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    return {
        "base": summary(base),
        "head": summary(head),
        "warnings": warnings,
        "counts": counts,
        "rows": rows,
    }


def _row_stub(m):
    return {k: m.get(k) for k in ("key", "section", "label", "unit", "kind", "better")}
