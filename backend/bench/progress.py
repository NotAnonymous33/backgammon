"""Progress of a running benchmark, written to a file the dashboard polls.

Sections report how much of their work is done with ``update(done, total,
unit)``. Without a reporter (the default, and in worker processes) the calls
do nothing, so sections stay usable on their own.
"""
import json
import os
import time

_reporter = None


class FileReporter:
    """Writes the run's state to ``path`` as JSON, at most every ``interval`` seconds."""

    def __init__(self, path, sections, interval=0.5):
        self.path = path
        self.interval = interval
        self.last_write = 0.0
        now = time.time()
        self.state = {
            "state": "running",
            "pid": os.getpid(),
            "started": now,
            "updated": now,
            "sections": list(sections),
            "section": None,
            "section_index": -1,
            "section_started": None,
            "done": 0,
            "total": None,
            "unit": None,
            "timings": {},
        }
        self.write()

    def write(self):
        self.state["updated"] = time.time()
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.state, f)
        os.replace(tmp, self.path)
        self.last_write = time.monotonic()

    def section(self, name):
        s = self.state
        if s["section"] is not None:
            s["timings"][s["section"]] = round(time.time() - s["section_started"], 2)
        s.update(section=name, section_index=s["sections"].index(name), section_started=time.time(),
                 done=0, total=None, unit=None)
        self.write()

    def update(self, done, total, unit):
        self.state.update(done=done, total=total, unit=unit)
        if done == total or time.monotonic() - self.last_write >= self.interval:
            self.write()

    def finish(self, state, **extra):
        s = self.state
        if s["section"] is not None and s["section"] not in s["timings"]:
            s["timings"][s["section"]] = round(time.time() - s["section_started"], 2)
        s.update(extra, state=state)
        self.write()


def set_reporter(reporter):
    global _reporter
    _reporter = reporter


def section(name):
    if _reporter is not None:
        _reporter.section(name)


def update(done, total, unit):
    if _reporter is not None:
        _reporter.update(done, total, unit)
