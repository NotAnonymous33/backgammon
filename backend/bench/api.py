"""Read-only HTTP API over recorded runs, used by the frontend dashboard.

Runs are started from the command line; the server only reads results, so a
long benchmark never competes with live games for the server process.
"""
import math

from flask import Blueprint, jsonify, request

from . import compare, store

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
