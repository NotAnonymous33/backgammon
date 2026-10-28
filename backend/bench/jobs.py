"""Benchmark runs started from the dashboard.

Each job runs ``python -m bench run`` in its own process group, so a long
benchmark never blocks the server and cancelling it stops every worker. A job
lives in ``results/jobs/<id>/``:

    job.json       what was asked for, and the process running it
    config.json    the complete configuration handed to the run
    progress.json  written by the run as it goes (see progress.py)
    log.txt        the run's console output

Only one job runs at a time: runs use every core, and two at once would
distort each other's timings.

Time estimates come from earlier runs that measured a section with the same
settings, and from the running section's own progress once it has started.
"""
import copy
import datetime as dt
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import threading
import time

from . import agents, store, suites

JOBS_DIR = os.path.join(store.RESULTS_DIR, "jobs")
LOG_TAIL = 40

# Settings that determine how long each section takes; earlier runs with the
# same values predict its duration.
SECTION_KEYS = {
    "rules": [],
    "speed": ["repeats", "min_time", "latency_agents", "latency_positions"],
    "search": ["mcts_agents", "search_positions", "search_repeats"],
    "training": ["training", "workers"],
    "strength": ["matches", "workers", "move_limit"],
    "skill": ["skill_agents", "skill_repeats", "skill_positions", "workers"],
    "ratings": ["ratings", "workers", "move_limit"],
}

# What a custom benchmark may change, on top of the preset it starts from.
EDITABLE = ("sections", "matches", "latency_agents", "mcts_agents", "skill_agents", "skill_repeats",
            "repeats", "latency_positions", "search_positions", "search_repeats", "workers", "seed")
RATINGS_KEYS = ("pool", "anchor", "initial_pairs", "batch_pairs", "max_pairs", "target_se", "max_games")
TRAINING_KEYS = ("learn_games", "learn_seeds", "learn_pairs", "epoch_games", "epoch_repeats", "speed_repeats")

_procs = {}  # job id -> Popen, for jobs started by this server process
_lock = threading.Lock()


class JobError(ValueError):
    pass


class Busy(RuntimeError):
    pass


# ---------------------------------------------------------------- configuration

def _int(value, name, lo=0, hi=None):
    try:
        n = int(value)
    except (TypeError, ValueError):
        raise JobError(f"{name} must be a whole number, got {value!r}")
    if n < lo:
        raise JobError(f"{name} must be at least {lo}")
    if hi is not None and n > hi:
        raise JobError(f"{name} must be at most {hi}")
    return n


def _workers(value):
    # Each worker is a spawned process holding its own networks, so an unbounded
    # count from a request could exhaust the machine's memory.
    cpus = os.cpu_count() or 1
    return _int(value or cpus, "workers", 1, cpus)


def _float(value, name, lo=0.0):
    try:
        x = float(value)
    except (TypeError, ValueError):
        raise JobError(f"{name} must be a number, got {value!r}")
    if x < lo:
        raise JobError(f"{name} must be at least {lo:g}")
    return x


def _specs(value, name):
    if not isinstance(value, list) or not all(isinstance(s, str) and s.strip() for s in value):
        raise JobError(f"{name} must be a list of agent specs")
    return [s.strip() for s in value]


def _comparable(cfg):
    return {k: v for k, v in cfg.items() if k not in ("sections", "workers")}


def build(request):
    """(suite name, complete config) for a job request; raises JobError if it is invalid."""
    kind = request.get("kind", "suite")
    if kind == "tournament":
        return "tournament", _tournament(request)
    base = request.get("suite", "quick")
    try:
        cfg = suites.get(base)
    except KeyError as e:
        raise JobError(str(e.args[0]))
    cfg["workers"] = os.cpu_count() or 1
    if kind == "suite":
        _validate(cfg)
        return base, cfg
    if kind != "custom":
        raise JobError(f"unknown job kind {kind!r}")

    preset = copy.deepcopy(cfg)
    over = request.get("config") or {}
    for key in EDITABLE:
        if key in over:
            cfg[key] = over[key]
    for key in TRAINING_KEYS:
        if key in (over.get("training") or {}):
            cfg["training"][key] = _int(over["training"][key], f"training.{key}")
    for key in RATINGS_KEYS:
        if key in (over.get("ratings") or {}):
            cfg["ratings"][key] = over["ratings"][key]
    _normalise(cfg)
    _validate(cfg)
    # Turning sections off keeps the run comparable with its preset, as --only does.
    return (base if _comparable(cfg) == _comparable(preset) else "custom"), cfg


def _tournament(request):
    pool = list(dict.fromkeys(_specs(request.get("pool"), "pool")))
    if len(pool) < 2:
        raise JobError("a tournament needs at least two agents")
    anchor = request.get("anchor") or ("heuristic" if "heuristic" in pool else pool[0])
    if anchor not in pool:
        raise JobError("the anchor must be one of the agents")
    pairs = _int(request.get("pairs", 20), "pairs per matchup", 1)
    adaptive = bool(request.get("adaptive", True))
    matchups = len(pool) * (len(pool) - 1) // 2
    cfg = suites.get("quick")
    cfg.update(
        sections=["ratings"], matches=[], latency_agents=[], mcts_agents=[], skill_agents=[],
        workers=_workers(request.get("workers")),
        ratings={
            "pool": pool, "anchor": anchor, "anchor_elo": 1500, "initial_pairs": pairs,
            "batch_pairs": pairs,
            # Without adaptive scheduling every matchup plays exactly `pairs`: a plain round robin.
            "max_pairs": _int(request.get("max_pairs", 200), "max pairs", pairs) if adaptive else pairs,
            "target_se": _float(request.get("target_se", 20.0), "target precision", 1.0) if adaptive else 1e9,
            "max_games": (_int(request.get("max_games", 4000), "game budget", 2) if adaptive
                          else 2 * pairs * matchups),
        },
    )
    if "seed" in request:
        cfg["seed"] = _int(request["seed"], "seed")
    _validate(cfg)
    return cfg


def _normalise(cfg):
    cfg["sections"] = [s for s in suites.SECTIONS if s in (cfg.get("sections") or [])]
    if not cfg["sections"]:
        raise JobError("turn on at least one section")
    for key in ("latency_agents", "mcts_agents", "skill_agents"):
        cfg[key] = _specs(cfg[key], key)
    matches = []
    for m in cfg["matches"]:
        if not isinstance(m, dict):
            raise JobError("matches must be {a, b, pairs} objects")
        a, b = _specs([m.get("a"), m.get("b")], "match agents")
        matches.append({"a": a, "b": b, "pairs": _int(m.get("pairs"), f"pairs for {a} vs {b}", 1)})
    cfg["matches"] = matches
    for key in ("skill_repeats", "repeats", "latency_positions", "search_positions", "search_repeats", "seed"):
        cfg[key] = _int(cfg[key], key, 0 if key == "seed" else 1)
    cfg["workers"] = _workers(cfg["workers"])
    rc = cfg["ratings"]
    rc["pool"] = list(dict.fromkeys(_specs(rc["pool"], "rating pool")))
    for key in ("initial_pairs", "batch_pairs", "max_pairs", "max_games"):
        rc[key] = _int(rc[key], f"ratings.{key}", 1)
    rc["target_se"] = _float(rc["target_se"], "ratings.target_se", 1.0)


def _check_web_spec(spec):
    """Refuse specs that would let a request run arbitrary code.

    ``py:`` specs import and call any class, and a checkpoint is unpickled
    when loaded, so dashboard jobs get the built-in agents and checkpoints
    under models/ only. The command line is not restricted.
    """
    kind, target, kwargs = agents.parse(spec)
    if kind == "py":
        raise JobError(f"{spec!r}: py: agents can only be benchmarked from the command line")
    models = os.path.realpath(os.path.join(agents.BACKEND_DIR, "models"))
    for path in agents.checkpoints_of(kind, target, kwargs):
        real = os.path.realpath(agents.resolve_path(str(path)))
        if os.path.commonpath([real, models]) != models:
            raise JobError(f"{spec!r}: checkpoints must be under backend/models/")


def _validate(cfg):
    """The checks runner.execute makes, done up front so mistakes surface before anything starts."""
    sec = set(cfg["sections"])
    specs = set()
    if "speed" in sec:
        specs |= set(cfg["latency_agents"])
    if "search" in sec:
        specs |= set(cfg["mcts_agents"])
        for spec in cfg["mcts_agents"]:
            if agents.parse(spec)[0] != "mcts":
                raise JobError(f"MCTS search configurations must be mcts agents, got {spec!r}")
    if "strength" in sec:
        specs |= {m["a"] for m in cfg["matches"]} | {m["b"] for m in cfg["matches"]}
        if not cfg["matches"]:
            raise JobError("the strength section needs at least one match")
    if "skill" in sec:
        specs |= set(cfg["skill_agents"])
    if "ratings" in sec:
        rc = cfg["ratings"]
        specs |= set(rc["pool"]) | {rc["anchor"]}
        if len(set(rc["pool"]) | {rc["anchor"]}) < 2:
            raise JobError("ratings need at least two agents")
    try:
        for spec in sorted(specs):
            _check_web_spec(spec)
            agents.validate(spec)
    except agents.SpecError as e:
        raise JobError(str(e))


# ---------------------------------------------------------------- estimates

def _history(runs):
    return [r for r in runs if r.get("timings") and r.get("config")]


def section_estimate(name, cfg, suite_name, runs):
    """(seconds, source) for one section: 'same settings', 'same suite' or (None, None)."""
    keys = SECTION_KEYS[name]
    same = [r["timings"][name] for r in runs if name in r["timings"]
            and all(r["config"].get(k) == cfg.get(k) for k in keys)]
    if same:
        return statistics.median(same[-3:]), "same settings"
    # Custom runs and tournaments share a name but not their settings, so only presets fall back to it.
    similar = [r["timings"][name] for r in runs if name in r["timings"] and r.get("suite") == suite_name]
    if similar and suite_name in suites.SUITES:
        return similar[-1], "same suite"
    return None, None


def estimate(cfg, suite_name, runs=None):
    runs = _history(store.load_all() if runs is None else runs)
    out = {}
    for name in cfg["sections"]:
        seconds, source = section_estimate(name, cfg, suite_name, runs)
        out[name] = {"seconds": seconds, "source": source}
    known = [e["seconds"] for e in out.values() if e["seconds"] is not None]
    return {"sections": out, "seconds": sum(known) if known else None,
            "unknown": [n for n, e in out.items() if e["seconds"] is None]}


def _live(job, prog, now, state=None):
    """Per-section elapsed and expected time, blending history with the running section's rate.

    ``state`` is the job's state when known. A cancelled or crashed run never
    updates its progress file, which then still says it is running.
    """
    running = (state or prog.get("state")) == "running"
    sections = []
    planned = job.get("estimate", {}).get("sections", {})
    index = prog.get("section_index", -1)
    for i, name in enumerate(prog["sections"]):
        hist = (planned.get(name) or {}).get("seconds")
        source = (planned.get(name) or {}).get("source")
        row = {"name": name, "estimate": hist, "source": source, "elapsed": None,
               "done": None, "total": None, "unit": None}
        if name in prog.get("timings", {}):
            row.update(state="done", elapsed=prog["timings"][name], estimate=prog["timings"][name])
        elif i == index and running:
            elapsed = now - prog["section_started"]
            done, total = prog.get("done") or 0, prog.get("total")
            row.update(state="running", elapsed=elapsed, done=done, total=total, unit=prog.get("unit"))
            frac = min(1.0, done / total) if total else 0.0
            # A rate from the first few seconds or items is mostly start-up cost.
            live = elapsed / frac if frac > 0 and (elapsed > 5 or frac > 0.1) else None
            if live is not None:
                # Lean on this run's own rate as more of the section completes.
                est = frac * live + (1 - frac) * hist if hist else live
                bounded = name == "ratings" and job.get("adaptive", True)
                if bounded:
                    # Progress counts against the game budget, so the rate gives an upper bound:
                    # adaptive scheduling often stops short of the budget.
                    est = min(est, live)
                if est < elapsed:
                    est = live
                row["source"] = "game budget (upper bound)" if bounded and not hist else "this run's progress"
            else:
                # An earlier run's time this section has already overrun says nothing about what is left.
                est = hist if hist is not None and hist > elapsed else None
            row["estimate"] = est
        else:
            row["state"] = "pending" if running else "skipped"
        sections.append(row)
    return sections


# ---------------------------------------------------------------- processes

JOB_ID = re.compile(r"[0-9]{8}-[0-9]{6}(-[0-9]+)?")


def _path(job_id, name):
    if not JOB_ID.fullmatch(job_id):
        raise KeyError(job_id)
    return os.path.join(JOBS_DIR, job_id, name)


def _job_ids():
    """Recorded jobs, newest first."""
    if not os.path.isdir(JOBS_DIR):
        return []
    ids = [d for d in os.listdir(JOBS_DIR) if JOB_ID.fullmatch(d) and os.path.exists(_path(d, "job.json"))]
    return sorted(ids, key=lambda d: (d[:15], int(d[16:] or 1)), reverse=True)


def _read(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def _write(path, doc):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=1)
    os.replace(tmp, path)


def _alive(job):
    proc = _procs.get(job["id"])
    if proc is not None:
        return proc.poll() is None
    # Started by an earlier server process (the dev server restarts on every edit),
    # so all that is left is its pid, which the system may since have reused.
    if (_read(_path(job["id"], "progress.json")) or {}).get("state") in ("finished", "failed"):
        return False
    pid = job.get("pid")
    if not isinstance(pid, int):
        return False
    if os.name == "nt":
        # On Windows os.kill(pid, 0) terminates the process rather than probing it.
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                             capture_output=True, text=True).stdout
        return f'"{pid}"' in out
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:  # where /proc exists, make sure the pid is still this job's run
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return job["id"].encode() in f.read()
    except OSError:
        return True


def start(request):
    """Validate a request and start its run. Raises JobError, or Busy if a job is running."""
    suite_name, cfg = build(request)
    label = (request.get("label") or "").strip() or None
    with _lock:
        if active():
            raise Busy("a benchmark is already running; wait for it or cancel it")
        now = dt.datetime.now().astimezone()
        job_id = base_id = now.strftime("%Y%m%d-%H%M%S")
        n = 1
        while os.path.exists(os.path.join(JOBS_DIR, job_id)):
            n += 1
            job_id = f"{base_id}-{n}"
        os.makedirs(os.path.join(JOBS_DIR, job_id))
        _write(_path(job_id, "config.json"), {"suite": suite_name, "config": cfg})
        cmd = [sys.executable, "-m", "bench", "run", "--config", _path(job_id, "config.json"),
               "--progress", _path(job_id, "progress.json")]
        if label:
            cmd += ["--label", label]
        if request.get("allow_stale"):
            cmd.append("--allow-stale")
        kwargs = {}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True  # its own process group, so cancel reaches the workers
        with open(_path(job_id, "log.txt"), "w") as log:
            proc = subprocess.Popen(cmd, cwd=agents.BACKEND_DIR, stdout=log, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, env=dict(os.environ, PYTHONUNBUFFERED="1"), **kwargs)
        _procs[job_id] = proc
        job = {
            "id": job_id, "kind": request.get("kind", "suite"), "suite": suite_name, "label": label,
            "created": now.isoformat(timespec="seconds"), "pid": proc.pid, "sections": cfg["sections"],
            "estimate": estimate(cfg, suite_name),
        }
        if job["kind"] == "tournament":
            job.update(pool=cfg["ratings"]["pool"], adaptive=bool(request.get("adaptive", True)))
        _write(_path(job_id, "job.json"), job)
    return status(job_id)


def cancel(job_id):
    job = _read(_path(job_id, "job.json"))
    if job is None:
        raise KeyError(job_id)
    if _alive(job):
        job["cancelled"] = True
        _write(_path(job_id, "job.json"), job)
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(job["pid"])], capture_output=True)
            else:
                os.killpg(job["pid"], signal.SIGTERM)
        except OSError:
            pass
        proc = _procs.get(job_id)
        if proc is not None:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if os.name != "nt":
                    os.killpg(job["pid"], signal.SIGKILL)
    return status(job_id)


def _tail(path, lines=LOG_TAIL):
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 16384))
            text = f.read().decode("utf-8", "replace")
    except OSError:
        return []
    # Workers' progress lines end in \r on some platforms; keep what a terminal would show.
    return [ln.rsplit("\r", 1)[-1] for ln in text.splitlines()[-lines:]]


def status(job_id, log=True):
    job = _read(_path(job_id, "job.json"))
    if job is None:
        raise KeyError(job_id)
    prog = _read(_path(job_id, "progress.json")) or {}
    state = prog.get("state")
    if state not in ("finished", "failed"):
        if _alive(job):
            state = "running" if prog else "starting"
        else:
            state = "cancelled" if job.get("cancelled") else "failed"
    now = time.time()
    out = dict(job, state=state, run_id=prog.get("run_id"), error=prog.get("error"))
    if state == "failed" and not out["error"]:
        out["error"] = "the run stopped unexpectedly; see its log"
    if prog.get("sections"):
        sections = _live(job, prog, now, state)
        out["sections_detail"] = sections
        out["elapsed"] = (prog["updated"] if state != "running" else now) - prog["started"]
        remaining = [max(0.0, s["estimate"] - (s["elapsed"] or 0)) for s in sections
                     if s["state"] in ("running", "pending") and s["estimate"] is not None]
        out["remaining"] = sum(remaining) if state == "running" else 0.0
        out["unknown"] = [s["name"] for s in sections if s["state"] in ("running", "pending") and s["estimate"] is None]
    if log:
        out["log"] = _tail(_path(job_id, "log.txt"))
    return out


def list_jobs(limit=10):
    return [status(i, log=False) for i in _job_ids()[:limit]]


def active():
    """The running job's id, if any."""
    for job_id in _job_ids()[:5]:
        job = _read(_path(job_id, "job.json"))
        if job and _alive(job):
            return job_id
    return None
