"""Run the configured sections and assemble a run record."""
import datetime as dt
import sys
import time

from . import agents, env, positions, rules, search, speed, strength, training


BUSY_WARNING = 0.15  # fraction of all cores busy with other work
TIMED_SECTIONS = ("speed", "search", "training")


class StaleBuildError(RuntimeError):
    pass


class ExtensionError(RuntimeError):
    pass


def make_id(started, git):
    stamp = started.strftime("%Y%m%d-%H%M%S")
    sha = git.get("short") or "nogit"
    return f"{stamp}-{sha}" + ("-dirty" if git.get("dirty") else "")


def check_extensions():
    problems = env.extension_problems()
    if problems:
        raise ExtensionError(
            f"this is Python {sys.version.split()[0]} ({sys.executable}), which cannot load the "
            "native extensions:\n  " + "\n  ".join(problems)
            + "\nActivate the virtual environment first (source venv/bin/activate, or "
            ".\\venv\\Scripts\\Activate.ps1 on Windows) and run `python -m bench ...`, "
            "or rebuild the extensions with this interpreter."
        )


def execute(cfg, suite_name, label=None, allow_stale=False, log=print):
    check_extensions()
    warnings = []
    stale = env.stale_builds()
    if stale and not allow_stale:
        raise StaleBuildError(
            "native extensions are out of date, so results would measure old code:\n  "
            + "\n  ".join(stale)
            + "\nRebuild, or pass --allow-stale to benchmark anyway."
        )
    warnings += [f"stale build: {s}" for s in stale]

    specs = {m["a"] for m in cfg["matches"]} | {m["b"] for m in cfg["matches"]}
    specs |= set(cfg["latency_agents"]) | set(cfg["mcts_agents"])
    for spec in sorted(specs):
        agents.validate(spec)
    for spec in cfg["mcts_agents"]:
        if agents.parse(spec)[0] != "mcts":
            raise agents.SpecError(f"--mcts-agent needs an mcts spec, got {spec!r}")

    suite = positions.load()
    started = dt.datetime.now().astimezone()
    environment = env.collect()
    if environment["git"].get("dirty"):
        warnings.append("working tree has uncommitted changes; the commit alone does not reproduce this run")

    run = {
        "schema": 1,
        "id": make_id(started, environment["git"]),
        "label": label,
        "suite": suite_name,
        "started_at": started.isoformat(timespec="seconds"),
        "config": dict(cfg, positions=positions.suite_fingerprint(suite), position_count=len(suite["positions"])),
        "agents": {s: agents.describe(s) for s in sorted(specs)},
        "env": environment,
        "warnings": warnings,
        "timings": {},
        "metrics": [],
    }
    t0 = time.perf_counter()
    sections = {
        "rules": lambda: rules.run(suite, log),
        "speed": lambda: speed.run(suite, cfg, log),
        "search": lambda: search.run(suite, cfg, log),
        "training": lambda: training.run(cfg, log),
        "strength": lambda: strength.run(cfg, log),
    }
    busy_log = run["env"]["host"]["cpu_busy_before"] = {}
    for name in cfg["sections"]:
        if name in TIMED_SECTIONS:
            busy = busy_log[name] = env.cpu_busy()
            if busy is not None and busy > BUSY_WARNING:
                msg = (f"machine was {busy:.0%} busy before the {name} benchmarks; "
                       "their timings are likely depressed")
                run["warnings"].append(msg)
                log(f"warning: {msg}")
        log(f"[{name}]")
        s0 = time.perf_counter()
        run["metrics"].extend(sections[name]())
        run["timings"][name] = round(time.perf_counter() - s0, 2)
    run["duration_s"] = round(time.perf_counter() - t0, 2)
    run["finished_at"] = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    return run
