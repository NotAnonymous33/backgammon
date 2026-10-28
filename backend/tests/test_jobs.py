import json
import os
import tempfile
import time
import unittest

from bench import jobs, progress, suites


class BuildTests(unittest.TestCase):
    def test_presets_keep_their_name(self):
        name, cfg = jobs.build({"kind": "suite", "suite": "standard"})
        self.assertEqual(name, "standard")
        self.assertEqual(cfg["sections"], list(suites.SECTIONS))

    def test_turning_sections_off_stays_comparable_with_the_preset(self):
        name, cfg = jobs.build({"kind": "custom", "suite": "quick", "config": {"sections": ["ratings", "rules"]}})
        self.assertEqual(name, "quick")
        self.assertEqual(cfg["sections"], ["rules", "ratings"])  # always in suite order

    def test_changed_settings_make_a_custom_run(self):
        name, cfg = jobs.build({"kind": "custom", "suite": "quick", "config": {
            "sections": ["strength"], "matches": [{"a": "nn", "b": "mcts:iters=50,mode=rollout", "pairs": "3"}]}})
        self.assertEqual(name, "custom")
        self.assertEqual(cfg["matches"], [{"a": "nn", "b": "mcts:iters=50,mode=rollout", "pairs": 3}])

    def test_round_robin_tournament_plays_exactly_its_pairs(self):
        name, cfg = jobs.build({"kind": "tournament", "pool": ["random", "nn", "heuristic"], "pairs": 5,
                                "adaptive": False})
        rc = cfg["ratings"]
        self.assertEqual((name, cfg["sections"]), ("tournament", ["ratings"]))
        self.assertEqual((rc["anchor"], rc["initial_pairs"], rc["max_pairs"]), ("heuristic", 5, 5))
        self.assertEqual(rc["max_games"], 2 * 5 * 3)

    def test_invalid_requests_are_refused(self):
        for request in [
            {"kind": "custom", "suite": "quick", "config": {"sections": []}},
            {"kind": "tournament", "pool": ["nn"]},
            {"kind": "tournament", "pool": ["nn", "heuristic"], "anchor": "random"},
            {"kind": "custom", "suite": "quick", "config": {"sections": ["search"], "mcts_agents": ["nn"]}},
            {"kind": "suite", "suite": "nope"},
        ]:
            with self.subTest(request=request), self.assertRaises(jobs.JobError):
                jobs.build(request)

    def test_more_workers_than_cpus_are_refused(self):
        too_many = (os.cpu_count() or 1) + 1
        for request in [{"kind": "tournament", "pool": ["nn", "heuristic"], "workers": too_many},
                        {"kind": "custom", "suite": "quick", "config": {"workers": too_many}}]:
            with self.subTest(request=request), self.assertRaises(jobs.JobError):
                jobs.build(request)

    def test_specs_that_could_run_arbitrary_code_are_refused(self):
        for spec in ["py:subprocess:Popen,args='ls'", "nn:/etc/passwd", "nn:models/../app.py",
                     "mcts:iters=10,nn=../app.py", "emm:ply=2,nn=/tmp/x.pt"]:
            with self.subTest(spec=spec), self.assertRaises(jobs.JobError):
                jobs.build({"kind": "tournament", "pool": ["heuristic", spec]})


class EstimateTests(unittest.TestCase):
    def run_record(self, suite, cfg, timings):
        return {"suite": suite, "config": cfg, "timings": timings}

    def test_history_with_the_same_settings_wins(self):
        _, cfg = jobs.build({"kind": "suite", "suite": "quick"})
        other = dict(cfg, matches=[])
        runs = [self.run_record("quick", other, {"strength": 99.0}), self.run_record("x", cfg, {"strength": 10.0})]
        self.assertEqual(jobs.section_estimate("strength", cfg, "quick", runs), (10.0, "same settings"))
        self.assertEqual(jobs.section_estimate("strength", cfg, "quick", runs[:1]), (99.0, "same suite"))
        # Custom runs share a name but not their settings.
        self.assertEqual(jobs.section_estimate("strength", cfg, "custom", [self.run_record("custom", other, {"strength": 5.0})]),
                         (None, None))

    def test_running_section_blends_history_with_its_rate(self):
        now = time.time()
        job = {"estimate": {"sections": {"skill": {"seconds": 100.0, "source": "same settings"},
                                         "ratings": {"seconds": None, "source": None}}}}
        prog = {"state": "running", "sections": ["skill", "ratings"], "section_index": 0,
                "section_started": now - 30, "done": 50, "total": 100, "timings": {}}
        skill, ratings = jobs._live(job, prog, now)
        self.assertEqual(skill["state"], "running")
        self.assertAlmostEqual(skill["estimate"], 0.5 * 60 + 0.5 * 100)
        self.assertEqual((ratings["state"], ratings["estimate"]), ("pending", None))

    def test_adaptive_ratings_treat_the_budget_as_an_upper_bound(self):
        now = time.time()
        job = {"estimate": {"sections": {"ratings": {"seconds": 500.0, "source": "same settings"}}}}
        prog = {"state": "running", "sections": ["ratings"], "section_index": 0,
                "section_started": now - 20, "done": 100, "total": 200, "timings": {}}
        (row,) = jobs._live(job, prog, now)
        self.assertAlmostEqual(row["estimate"], 40.0)  # the rate to the budget, not the slower history

    def test_a_cancelled_run_stops_showing_sections_as_running(self):
        now = time.time()
        job = {"estimate": {"sections": {}}}
        # SIGTERM leaves the progress file saying the run is still going.
        prog = {"state": "running", "sections": ["strength", "skill", "ratings"], "section_index": 1,
                "section_started": now - 20, "done": 40, "total": 100, "timings": {"strength": 30.0}}
        states = [s["state"] for s in jobs._live(job, prog, now, "cancelled")]
        self.assertEqual(states, ["done", "skipped", "skipped"])


class ProcessTests(unittest.TestCase):
    def test_a_reused_pid_is_not_mistaken_for_the_job(self):
        with tempfile.TemporaryDirectory() as d:
            saved, jobs.JOBS_DIR = jobs.JOBS_DIR, d
            try:
                os.makedirs(os.path.join(d, "20261028-120000"))
                # This test's own process is alive but is not running that job.
                job = {"id": "20261028-120000", "pid": os.getpid()}
                self.assertEqual(jobs._alive(job), not os.path.exists("/proc"))
                jobs._write(jobs._path(job["id"], "progress.json"), {"state": "finished"})
                self.assertFalse(jobs._alive(job))
            finally:
                jobs.JOBS_DIR = saved


class ProgressTests(unittest.TestCase):
    def test_reporter_records_sections_and_outcome(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "progress.json")
            reporter = progress.FileReporter(path, ["rules", "speed"])
            progress.set_reporter(reporter)
            try:
                progress.section("rules")
                progress.update(3, 3, "steps")
                progress.section("speed")
                progress.update(1, 2, "steps")
                reporter.finish("finished", run_id="r1")
            finally:
                progress.set_reporter(None)
            with open(path) as f:
                state = json.load(f)
        self.assertEqual(state["state"], "finished")
        self.assertEqual(state["run_id"], "r1")
        self.assertEqual(set(state["timings"]), {"rules", "speed"})
        self.assertEqual((state["section"], state["done"], state["total"]), ("speed", 1, 2))


if __name__ == "__main__":
    unittest.main()
