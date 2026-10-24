"""Run records on disk and the metric format they contain.

Every run is one JSON file in ``bench/results`` (git-ignored: results are
machine-specific, and keeping them out of git means they survive checkouts
of older commits). A metric looks like::

    {"key": "speed.movegen", "section": "speed", "label": "...", "unit": "pos/s",
     "kind": "timing", "better": "higher", "value": 51234.0, "samples": [...]}

``kind`` controls how runs are compared (see compare.py):
    timing  repeated measurements; noise estimated from their spread
    count   deterministic integer; any change is real
    rate    win rate over seeded games; ``outcomes`` allows paired tests
    mean    mean over fixed items (e.g. positions); ``samples`` allows paired tests
    rating  Elo rating against a fixed anchor, with a confidence interval
    value   informational, never flagged
"""
import json
import os
import re

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
SCHEMA = 1


def metric(key, label, value, unit="", kind="value", better="none", **extra):
    m = {
        "key": key,
        "section": re.match(r"[a-z]+", key).group(0),  # keys embed agent specs, which may contain dots
        "label": label,
        "unit": unit,
        "kind": kind,
        "better": better,
        "value": value,
    }
    m.update({k: v for k, v in extra.items() if v is not None})
    return m


def save(run, results_dir=RESULTS_DIR):
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"{run['id']}.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(run, f, indent=1)
        f.write("\n")
    os.replace(tmp, path)
    return path


def _run_files(results_dir):
    if not os.path.isdir(results_dir):
        return []
    return [f for f in os.listdir(results_dir) if f.endswith(".json")]


def load_all(results_dir=RESULTS_DIR):
    """All runs, oldest first."""
    runs = []
    for name in _run_files(results_dir):
        try:
            with open(os.path.join(results_dir, name)) as f:
                runs.append(json.load(f))
        except (OSError, json.JSONDecodeError):
            continue
    runs.sort(key=lambda r: r.get("started_at", ""))
    return runs


class RefError(LookupError):
    pass


def resolve(ref, runs):
    """Find a run by id, unique id prefix, label, ``latest`` or ``latest~N``.

    Labels resolve to the most recent run carrying that label.
    """
    if not runs:
        raise RefError("no benchmark runs recorded yet; run `python -m bench run` first")
    m = re.fullmatch(r"latest(?:~(\d+))?", ref)
    if m:
        back = int(m.group(1) or 0)
        if back >= len(runs):
            raise RefError(f"{ref}: only {len(runs)} run(s) recorded")
        return runs[-1 - back]
    for run in runs:
        if run["id"] == ref:
            return run
    labelled = [r for r in runs if r.get("label") == ref]
    if labelled:
        return labelled[-1]
    prefixed = [r for r in runs if r["id"].startswith(ref)]
    if len(prefixed) == 1:
        return prefixed[0]
    if len(prefixed) > 1:
        raise RefError(f"{ref!r} is ambiguous: " + ", ".join(r["id"] for r in prefixed[:5]))
    raise RefError(f"no run matches {ref!r}")


def sections(run):
    return set(run.get("config", {}).get("sections") or {m["section"] for m in run.get("metrics", [])})


def default_base(runs, head):
    """Most recent earlier run of the same suite that measured every section ``head`` did."""
    earlier = [r for r in runs if r["started_at"] < head["started_at"] and r["id"] != head["id"]]
    for run in reversed(earlier):
        if run.get("suite") == head.get("suite") and sections(head) <= sections(run):
            return run
    return earlier[-1] if earlier else None


def summary(run):
    """Compact listing entry (no samples or outcomes)."""
    env = run.get("env", {})
    git = env.get("git", {})
    return {
        "id": run["id"],
        "label": run.get("label"),
        "started_at": run.get("started_at"),
        "duration_s": run.get("duration_s"),
        "suite": run.get("suite"),
        "sections": run.get("config", {}).get("sections", []),
        "git": {k: git.get(k) for k in ("sha", "short", "branch", "dirty", "subject")},
        "host": env.get("host", {}).get("fingerprint"),
        "warnings": run.get("warnings", []),
        "metrics": {
            m["key"]: {k: m.get(k) for k in ("label", "value", "unit", "better", "kind", "ci", "section")}
            for m in run.get("metrics", [])
        },
    }
