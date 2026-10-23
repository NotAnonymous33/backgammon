"""Describe the code and machine a run measured."""
import glob
import os
import platform
import socket
import subprocess
import sys
from importlib import metadata

from .agents import BACKEND_DIR

REPO_DIR = os.path.dirname(BACKEND_DIR)

# Native extensions and the sources they are built from. Benchmarking a
# stale build silently measures the old code.
NATIVE_BUILDS = [
    ("classes/board_cpp.cpp", "classes/board_cpp*.so", "python setup.py build_ext --inplace"),
    ("classes/agents/CMCTS2.pyx", "classes/agents/CMCTS2*.so", "python setup_cython.py build_ext --inplace"),
]


def _git(*args):
    try:
        out = subprocess.run(
            ["git", *args], cwd=REPO_DIR, capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def git_info():
    sha = _git("rev-parse", "HEAD")
    if sha is None:
        return {"sha": None}
    status = _git("status", "--porcelain", "--untracked-files=no") or ""
    return {
        "sha": sha,
        "short": sha[:7],
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "subject": _git("log", "-1", "--format=%s"),
        "dirty": bool(status),
        "changed_files": [line[3:] for line in status.splitlines()][:50],
    }


def _cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _version(pkg):
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def host_info():
    cpu = _cpu_model()
    cores = os.cpu_count()
    name = socket.gethostname()
    return {
        "hostname": name,
        "platform": platform.platform(),
        "cpu": cpu,
        "cpu_count": cores,
        "python": sys.version.split()[0],
        "torch": _version("torch"),
        "numpy": _version("numpy"),
        # Speed numbers are only comparable between runs with the same fingerprint.
        "fingerprint": f"{name}|{cpu}|{cores}",
    }


def cpu_busy(interval=0.5):
    """Fraction of total CPU time in use right now (Linux only, else None).

    Other work on the machine depresses speed results well beyond their noise
    floors, so runs record how busy the machine was when speed was measured.
    """
    import time

    def sample():
        with open("/proc/stat") as f:
            fields = [int(x) for x in f.readline().split()[1:]]
        idle = fields[3] + (fields[4] if len(fields) > 4 else 0)
        return idle, sum(fields)

    try:
        idle1, total1 = sample()
        time.sleep(interval)
        idle2, total2 = sample()
    except (OSError, ValueError, IndexError):
        return None
    total = total2 - total1
    return round(1 - (idle2 - idle1) / total, 3) if total > 0 else None


def _builds_for_this_python(pattern):
    """Compiled files matching pattern that this interpreter can import."""
    from importlib.machinery import EXTENSION_SUFFIXES

    stem = os.path.join(BACKEND_DIR, pattern.split("*")[0])
    return [stem + sfx for sfx in EXTENSION_SUFFIXES if os.path.exists(stem + sfx)]


def extension_problems():
    """Native extensions the running interpreter cannot load at all.

    The usual cause is running outside the virtual environment: the builds
    carry the Python version they were compiled for (e.g. cpython-311).
    """
    problems = []
    for src, pattern, cmd in NATIVE_BUILDS:
        if not os.path.exists(os.path.join(BACKEND_DIR, src)) or _builds_for_this_python(pattern):
            continue
        others = sorted(os.path.basename(p) for p in glob.glob(os.path.join(BACKEND_DIR, pattern)))
        if others:
            problems.append(f"{src} is only built for another Python: {', '.join(others)}")
        else:
            problems.append(f"{src} has never been built (cd backend/classes && {cmd})")
    return problems


def stale_builds():
    """Native extensions older than their sources, with the rebuild command."""
    stale = []
    for src, pattern, cmd in NATIVE_BUILDS:
        src_path = os.path.join(BACKEND_DIR, src)
        builds = _builds_for_this_python(pattern)
        if not os.path.exists(src_path) or not builds:
            continue  # missing builds are reported by extension_problems()
        if max(os.path.getmtime(b) for b in builds) < os.path.getmtime(src_path):
            stale.append(f"{src} is newer than its build (cd backend/classes && {cmd})")
    return stale


def collect():
    return {"git": git_info(), "host": host_info()}
