"""Agent specs: short strings that name an agent and its settings.

    random                         RandomAgent
    first                          FirstAgent
    heuristic                      HeuristicAgent with default weights
    nn                             FinalNNAgent, models/main/main_checkpoint_latest.pt
    nn:models/main/x.pt            FinalNNAgent with another checkpoint
    mcts:time=0.5                  MCTSAgent2 (keys: time, iters, mode, depth, sim, c, nn)
    mcts:iters=400                 MCTSAgent2 stopping after 400 iterations per move
    py:package.module:Class,k=v    any class with select_move(board), built with k=v

Heavy resources (neural checkpoints) are loaded once per process and shared;
agents themselves are rebuilt for every game so search trees never leak
between games.
"""
import ast
import hashlib
import importlib
import os

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_NN = "models/main/main_checkpoint_latest.pt"

KINDS = ("random", "first", "heuristic", "nn", "mcts", "py")
MCTS_KEYS = ("time", "iters", "mode", "depth", "sim", "c", "nn")
ITERATION_TIME_CAP = 600.0  # seconds; only reached if a search stalls


class SpecError(ValueError):
    pass


def _value(text):
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def parse(spec):
    """Return (kind, target, kwargs) for a spec string."""
    spec = spec.strip()
    kind, _, rest = spec.partition(":")
    kind = kind.lower()
    if kind not in KINDS:
        raise SpecError(f"unknown agent kind {kind!r} in {spec!r}; expected one of {', '.join(KINDS)}")

    target = None
    kwargs = {}
    parts = [p for p in rest.split(",") if p] if rest else []
    if kind == "py":
        if not parts or ":" not in parts[0]:
            raise SpecError(f"py agents need module:Class, got {spec!r}")
        target = parts.pop(0)
    for part in parts:
        if "=" in part:
            key, _, val = part.partition("=")
            kwargs[key.strip()] = _value(val.strip())
        elif kind == "nn" and target is None:
            target = part
        else:
            raise SpecError(f"cannot parse {part!r} in {spec!r}; use key=value")
    if kind == "nn":
        target = kwargs.pop("path", target) or DEFAULT_NN
    if kind == "mcts":
        unknown = set(kwargs) - set(MCTS_KEYS)
        if unknown:
            raise SpecError(f"unknown mcts key(s) {', '.join(sorted(unknown))} in {spec!r}; "
                            f"expected {', '.join(MCTS_KEYS)}")
    return kind, target, kwargs


def resolve_path(path):
    return path if os.path.isabs(path) else os.path.join(BACKEND_DIR, path)


_hash_cache = {}


def file_digest(path):
    path = resolve_path(path)
    if path not in _hash_cache:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        _hash_cache[path] = h.hexdigest()[:16]
    return _hash_cache[path]


def describe(spec):
    """Metadata identifying exactly what a spec ran, including checkpoint digests."""
    kind, target, kwargs = parse(spec)
    info = {"spec": spec, "kind": kind}
    checkpoints = []
    if kind == "nn":
        checkpoints.append(target)
    if kind == "mcts" and kwargs.get("mode") == "neural":
        checkpoints.append(kwargs.get("nn", DEFAULT_NN))
    if checkpoints:
        info["checkpoints"] = {p: file_digest(p) for p in checkpoints}
    return info


def validate(spec):
    """Fail fast on a typo or missing checkpoint before spawning workers."""
    kind, target, kwargs = parse(spec)
    paths = []
    if kind == "nn":
        paths.append(target)
    if kind == "mcts" and kwargs.get("mode") == "neural":
        paths.append(kwargs.get("nn", DEFAULT_NN))
    for p in paths:
        if not os.path.exists(resolve_path(p)):
            raise SpecError(f"checkpoint not found for {spec!r}: {p}")


def _load_nn(path, cache):
    key = ("nn", resolve_path(path))
    if key not in cache:
        from classes.agents.NNAgent import FinalNNAgent

        cache[key] = FinalNNAgent(checkpoint_path=resolve_path(path))
    return cache[key]


def make_agent(spec, cache):
    """Build a fresh agent for one game. ``cache`` is a per-process dict."""
    kind, target, kwargs = parse(spec)
    if kind == "random":
        from classes.agents.RandomAgent import RandomAgent

        return RandomAgent()
    if kind == "first":
        from classes.agents.FirstAgent import FirstAgent

        return FirstAgent()
    if kind == "heuristic":
        from classes.agents.HeuristicAgent import HeuristicAgent

        return HeuristicAgent(**kwargs)
    if kind == "nn":
        # FinalNNAgent holds no per-game state, so one instance is shared.
        return _load_nn(target, cache)
    if kind == "mcts":
        from classes.agents.CMCTS2 import MCTSAgent2  # type: ignore

        mode = kwargs.get("mode", "default")
        iters = int(kwargs.get("iters", 0))
        # An iteration budget makes searches reproducible and independent of
        # machine speed; time then only caps a runaway search.
        default_time = ITERATION_TIME_CAP if iters else 1.0
        args = {
            "time_budget": float(kwargs.get("time", default_time)),
            "max_iterations": iters,
            "mode": mode,
            "eval_depth": int(kwargs.get("depth", -1)),
            "simulation_depth": int(kwargs.get("sim", 50)),
            "exploration_weight": float(kwargs.get("c", 1.0)),
        }
        if mode == "neural":
            args["nn_agent"] = _load_nn(kwargs.get("nn", DEFAULT_NN), cache)
        return MCTSAgent2(**args)
    module_name, _, class_name = target.partition(":")
    cls = getattr(importlib.import_module(module_name), class_name)
    return cls(**kwargs)
