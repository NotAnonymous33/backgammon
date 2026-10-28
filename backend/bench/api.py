"""HTTP API for the frontend dashboard: recorded runs, and starting new ones.

Runs started here execute in their own process (see jobs.py), so a long
benchmark never blocks the server, though it does use every core.
"""
import glob
import math
import os

from flask import Blueprint, jsonify, request

from . import agents, compare, jobs, reference, store, suites

bp = Blueprint("bench", __name__, url_prefix="/api/bench")


def _json_safe(obj):
    """Replace NaN/inf, which browsers' JSON.parse rejects, with null."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def _resolve(ref, runs):
    try:
        return store.resolve(ref, runs), None
    except store.RefError as e:
        return None, (jsonify({"error": str(e)}), 404)


@bp.get("/runs")
def list_runs():
    return jsonify(_json_safe([store.summary(r) for r in store.load_all()]))


@bp.get("/runs/<ref>")
def get_run(ref):
    run, err = _resolve(ref, store.load_all())
    return err or jsonify(_json_safe(run))


@bp.get("/compare")
def compare_runs():
    runs = store.load_all()
    head, err = _resolve(request.args.get("head", "latest"), runs)
    if err:
        return err
    base_ref = request.args.get("base")
    if base_ref:
        base, err = _resolve(base_ref, runs)
        if err:
            return err
    else:
        base = store.default_base(runs, head)
        if base is None:
            return jsonify({"error": "no earlier run to compare against"}), 404
    return jsonify(_json_safe(compare.compare_runs(base, head)))


@bp.get("/options")
def options():
    """Everything the run forms need: presets, sections, checkpoints, MCTS defaults and time estimates."""
    runs = store.load_all()
    presets = {}
    for name in suites.SUITES:
        cfg = suites.get(name)
        cfg["workers"] = os.cpu_count() or 1
        presets[name] = {"config": cfg, "estimate": jobs.estimate(cfg, name, runs)}
    models = sorted(os.path.relpath(p, agents.BACKEND_DIR).replace(os.sep, "/")
                    for p in glob.glob(os.path.join(agents.BACKEND_DIR, "models", "**", "*.pt"), recursive=True))
    return jsonify(_json_safe({
        "presets": presets,
        "preset_order": list(suites.SUITES),  # jsonify sorts the presets' keys
        "sections": list(suites.SECTIONS),
        "checkpoints": models,
        "default_nn": agents.DEFAULT_NN,
        "cpu_count": os.cpu_count() or 1,
        "has_reference": os.path.exists(reference.DATA_PATH),
        # MCTSAgent2's defaults (CMCTS2.pyx), shown as placeholders.
        "mcts": {"modes": ["neural", "rollout", "heuristic"], "exploration": {"neural": 0.01, "heuristic": 0.2,
                                                                          "rollout": 0.2},
                 "sim": 50, "time": 1.0},
    }))


@bp.post("/estimate")
def estimate():
    """Expected duration of a job request, from earlier runs, without starting it."""
    try:
        suite_name, cfg = jobs.build(request.get_json(force=True) or {})
    except jobs.JobError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify(_json_safe(dict(jobs.estimate(cfg, suite_name), suite=suite_name)))


@bp.get("/jobs")
def list_jobs():
    return jsonify(_json_safe({"active": jobs.active(), "jobs": jobs.list_jobs()}))


@bp.post("/jobs")
def start_job():
    try:
        job = jobs.start(request.get_json(force=True) or {})
    except jobs.JobError as e:
        return jsonify({"error": str(e)}), 400
    except jobs.Busy as e:
        return jsonify({"error": str(e)}), 409
    return jsonify(_json_safe(job)), 202


@bp.get("/jobs/<job_id>")
def get_job(job_id):
    try:
        return jsonify(_json_safe(jobs.status(job_id)))
    except KeyError:
        return jsonify({"error": f"no job {job_id!r}"}), 404


@bp.post("/jobs/<job_id>/cancel")
def cancel_job(job_id):
    try:
        return jsonify(_json_safe(jobs.cancel(job_id)))
    except KeyError:
        return jsonify({"error": f"no job {job_id!r}"}), 404
