"""Command-line interface: ``python -m bench <command>`` from the backend directory."""
import argparse
import json
import os
import sys

from . import compare, positions, store, suites
from .agents import SpecError

STATUS_STYLE = {
    "improved": ("32", "▲ improved"),
    "regressed": ("31", "▼ regressed"),
    "unchanged": ("2", "  unchanged"),
    "new": ("36", "+ new"),
    "removed": ("33", "- removed"),
    "info": ("2", "  info"),
}


def _color(text, code):
    if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
        return text
    return f"\033[{code}m{text}\033[0m"


def fmt(value, unit):
    if value is None:
        return "—"
    if isinstance(value, float) and value != value:
        return "n/a"
    if unit == "fraction":
        return f"{value:.1%}"
    if unit == "ms":
        return f"{value:.2f} ms"
    if unit.endswith("/s"):
        return f"{value:,.1f} {unit}" if abs(value) < 100 else f"{value:,.0f} {unit}"
    if isinstance(value, float):
        return f"{value:.2f} {unit}".strip()
    return f"{value} {unit}".strip()


def fmt_delta(row):
    if row.get("delta") is None:
        return ""
    if row["kind"] == "rate" and row.get("diff_ci"):
        lo, hi = row["diff_ci"]
        diff = row.get("diff", (lo + hi) / 2)
        return f"{diff * 100:+.1f} pts [{lo * 100:+.1f}, {hi * 100:+.1f}]"
    if row.get("delta_pct") is not None and row["kind"] == "timing":
        return f"{row['delta_pct']:+.1%}"
    if row["delta"] == 0:
        return "0"
    return f"{row['delta']:+g}" if isinstance(row["delta"], int) else f"{row['delta']:+.3g}"


def _table(rows, headers):
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(headers)]
    line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    out = [line, "  ".join("─" * w for w in widths)]
    for r in rows:
        out.append("  ".join(c.ljust(w) for c, w in zip(r, widths)))
    return out


def print_comparison(result, show_all=True):
    b, h = result["base"], result["head"]
    print(f"base  {b['id']}  {b['label'] or ''}  {b['git'].get('subject') or ''}")
    print(f"head  {h['id']}  {h['label'] or ''}  {h['git'].get('subject') or ''}")
    for w in result["warnings"]:
        print(_color(f"! {w}", "33"))
    rows = result["rows"]
    for section in suites.SECTIONS:
        sec = [r for r in rows if r["section"] == section and (show_all or r["status"] not in ("unchanged", "info"))]
        if not sec:
            continue
        print(f"\n{section.upper()}")
        table = [
            [r["label"], fmt(r["base"], r["unit"]), fmt(r["head"], r["unit"]), fmt_delta(r),
             STATUS_STYLE[r["status"]][1]]
            for r in sec
        ]
        lines = _table(table, ["metric", "base", "head", "change", "status"])
        print("\n".join(lines[:2]))
        for r, line in zip(sec, lines[2:]):
            print(_color(line, STATUS_STYLE[r["status"]][0]))
    c = result["counts"]
    print(f"\n{c.get('improved', 0)} improved, {c.get('regressed', 0)} regressed, "
          f"{c.get('unchanged', 0)} unchanged")


def print_run(run):
    git = run["env"]["git"]
    print(f"{run['id']}  suite={run['suite']}  label={run.get('label') or '-'}")
    print(f"commit {git.get('short')} ({git.get('branch')}){' + local changes' if git.get('dirty') else ''}: "
          f"{git.get('subject')}")
    print(f"{run['env']['host']['cpu']} · {run['duration_s']}s")
    for w in run.get("warnings", []):
        print(_color(f"! {w}", "33"))
    for section in suites.SECTIONS:
        ms = [m for m in run["metrics"] if m["section"] == section]
        if not ms:
            continue
        print(f"\n{section.upper()}")
        table = []
        for m in ms:
            extra = ""
            if m.get("ci") and m["unit"] == "fraction":
                extra = f"95% CI {m['ci'][0]:.1%}–{m['ci'][1]:.1%}"
            table.append([m["label"], fmt(m["value"], m["unit"]), extra])
        print("\n".join(_table(table, ["metric", "value", ""])))


def _resolve(ref, runs):
    try:
        return store.resolve(ref, runs)
    except store.RefError as e:
        sys.exit(f"error: {e}")


def cmd_run(args):
    from . import runner

    cfg = suites.get(args.suite)
    if args.only:
        cfg["sections"] = [s for s in suites.SECTIONS if s in args.only.split(",")]
        unknown = set(args.only.split(",")) - set(suites.SECTIONS)
        if unknown:
            sys.exit(f"error: unknown section(s) {', '.join(sorted(unknown))}")
    if args.match:
        cfg["matches"] = []
        for text in args.match:
            a, sep, b = text.partition(" vs ")
            if not sep:
                sys.exit(f"error: --match expects 'A vs B', got {text!r}")
            cfg["matches"].append({"a": a.strip(), "b": b.strip(), "pairs": 100})
    if args.pairs:
        for m in cfg["matches"]:
            m["pairs"] = args.pairs
    if args.latency_agent is not None:
        cfg["latency_agents"] = args.latency_agent
    if args.mcts_agent is not None:
        cfg["mcts_agents"] = args.mcts_agent
    if args.training_games:
        cfg["training"]["learn_games"] = args.training_games
    if args.seed is not None:
        cfg["seed"] = args.seed
    cfg["workers"] = args.workers or os.cpu_count() or 1

    runs = store.load_all()
    try:
        run = runner.execute(cfg, args.suite, label=args.label, allow_stale=args.allow_stale)
    except (runner.StaleBuildError, runner.ExtensionError, SpecError) as e:
        sys.exit(f"error: {e}")
    path = store.save(run)
    print(f"\nsaved {run['id']} → {os.path.relpath(path)}")

    base = _resolve(args.compare_to, runs) if args.compare_to else store.default_base(runs, run)
    if base is None:
        print("no earlier run to compare against")
        return 0
    print()
    result = compare.compare_runs(base, run)
    print_comparison(result, show_all=False)
    return 1 if args.fail_on_regression and result["counts"].get("regressed") else 0


def cmd_list(args):
    runs = store.load_all()
    if not runs:
        print("no runs yet: python -m bench run")
        return 0
    table = []
    for r in runs[-args.n:]:
        s = store.summary(r)
        rates = [m for m in s["metrics"].values() if m["kind"] == "rate"]
        headline = ", ".join(f"{m['label'].split(':')[0]} {m['value']:.0%}" for m in rates[:3])
        git = s["git"]
        table.append([
            s["id"], s["label"] or "", s["suite"] or "", f"{git.get('short')}{'*' if git.get('dirty') else ''}",
            (git.get("subject") or "")[:40], headline,
        ])
    print("\n".join(_table(table, ["id", "label", "suite", "commit", "subject", "scores"])))
    return 0


def cmd_show(args):
    runs = store.load_all()
    run = _resolve(args.ref, runs)
    if args.json:
        print(json.dumps(run, indent=1))
    else:
        print_run(run)
    return 0


def cmd_compare(args):
    runs = store.load_all()
    head = _resolve(args.head, runs)
    if args.base:
        base = _resolve(args.base, runs)
    else:
        base = store.default_base(runs, head)
        if base is None:
            sys.exit("error: no earlier run to compare against; pass a base explicitly")
    result = compare.compare_runs(base, head)
    if args.json:
        print(json.dumps(result, indent=1))
    else:
        print_comparison(result, show_all=not args.changes)
    return 1 if args.fail_on_regression and result["counts"].get("regressed") else 0


def cmd_positions(args):
    if args.generate:
        from . import runner

        try:
            runner.check_extensions()
        except runner.ExtensionError as e:
            sys.exit(f"error: {e}")
        if os.path.exists(positions.DATA_PATH) and not args.force:
            sys.exit("error: the position suite exists; regenerating breaks comparison with "
                     "earlier runs. Pass --force to replace it.")
        suite = positions.generate(per_category=args.per_category, seed=args.seed)
        positions.save(suite)
        print(f"wrote {len(suite['positions'])} positions to {os.path.relpath(positions.DATA_PATH)}")
    suite = positions.load()
    counts = {}
    for p in suite["positions"]:
        counts[p["category"]] = counts.get(p["category"], 0) + 1
    print(f"{len(suite['positions'])} positions, fingerprint {positions.suite_fingerprint(suite)}")
    for cat, n in counts.items():
        print(f"  {cat:8} {n}")
    return 0


def build_parser():
    p = argparse.ArgumentParser(prog="python -m bench", description="Backgammon benchmark suite")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="run benchmarks and record the results")
    r.add_argument("--suite", default="quick", choices=list(suites.SUITES))
    r.add_argument("--only", help="comma-separated sections: " + ",".join(suites.SECTIONS))
    r.add_argument("--label", help="name this run, e.g. 'baseline' or 'movegen-fix'")
    r.add_argument("--match", action="append", metavar="'A vs B'",
                   help="replace the default matches (repeatable); e.g. 'nn:models/x.pt vs heuristic'")
    r.add_argument("--pairs", type=int, help="game pairs per match (each pair is 2 games)")
    r.add_argument("--latency-agent", action="append", metavar="SPEC",
                   help="replace the agents timed per decision (repeatable)")
    r.add_argument("--mcts-agent", action="append", metavar="SPEC",
                   help="replace the MCTS configurations whose iterations are counted (repeatable)")
    r.add_argument("--training-games", type=int, metavar="N",
                   help="self-play games per learning run in the training section")
    r.add_argument("--seed", type=int, help="base seed (keep the default so runs stay paired)")
    r.add_argument("--workers", type=int, help="processes for strength games (default: all cores)")
    r.add_argument("--compare-to", metavar="REF", help="run to compare with (default: previous run of this suite)")
    r.add_argument("--fail-on-regression", action="store_true", help="exit 1 if anything regressed")
    r.add_argument("--allow-stale", action="store_true", help="run even if native builds are out of date")
    r.set_defaults(func=cmd_run)

    ls = sub.add_parser("list", help="list recorded runs")
    ls.add_argument("-n", type=int, default=20)
    ls.set_defaults(func=cmd_list)

    s = sub.add_parser("show", help="show one run")
    s.add_argument("ref", nargs="?", default="latest", help="id, id prefix, label, latest or latest~N")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_show)

    c = sub.add_parser("compare", help="compare two runs")
    c.add_argument("base", nargs="?", help="default: the latest earlier run of the same suite covering the same sections")
    c.add_argument("head", nargs="?", default="latest")
    c.add_argument("--changes", action="store_true", help="only show improved/regressed/new/removed")
    c.add_argument("--json", action="store_true")
    c.add_argument("--fail-on-regression", action="store_true")
    c.set_defaults(func=cmd_compare)

    ps = sub.add_parser("positions", help="inspect or regenerate the position suite")
    ps.add_argument("--generate", action="store_true")
    ps.add_argument("--force", action="store_true")
    ps.add_argument("--per-category", type=int, default=80)
    ps.add_argument("--seed", type=int, default=1)
    ps.set_defaults(func=cmd_positions)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args) or 0
