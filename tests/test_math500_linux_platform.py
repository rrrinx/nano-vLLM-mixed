"""Linux sampling and real worker process-group cleanup, with stdlib only.

Run directly with Python on Linux when torch/pytest are not yet installed.
"""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from time import monotonic, sleep
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common.diagnostics import HostSampler
from experiments.math500 import run_math500 as experiment


class LinuxPlatformTests(unittest.TestCase):
    def test_proc_cpu_delta_steal_and_guest_accounting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "stat").write_text("cpu 100 0 50 800 20 0 0 30 20 0\n")
            (root / "meminfo").write_text("MemTotal: 4194304 kB\nMemAvailable: 2097152 kB\n")
            sampler = HostSampler(proc_root=root, platform_name="posix")
            first = sampler.read()
            self.assertEqual(first, {"system_available_memory_mb": 2048})
            (root / "stat").write_text("cpu 140 0 60 830 30 0 0 40 40 0\n")
            (root / "meminfo").write_text("MemTotal: 4194304 kB\nMemAvailable: 1048576 kB\n")
            second = sampler.read()
            self.assertEqual(second["system_available_memory_mb"], 1024)
            self.assertAlmostEqual(second["system_cpu_pct"], 60)
            self.assertAlmostEqual(second["system_cpu_steal_pct"], 10)

    def test_server_config_is_portable_and_balanced(self):
        config = Path(__file__).resolve().parents[1] / "experiments/math500/math500_linux_config.json"
        options = experiment.parse_options(["--config", str(config)])
        self.assertEqual(options["repeat_schedule"], "round_robin")
        self.assertEqual(options["repeats"], 12)
        self.assertEqual(options["warmup"], 1)
        self.assertEqual(options["blocks"], -1)
        self.assertEqual(options["max_model_len"], 0)
        self.assertFalse(options["diagnostics"])
        self.assertTrue(options["model"].endswith(str(Path("offline/Qwen3-0.6B"))))
        self.assertNotIn("D:", options["model"])

    @unittest.skipUnless(sys.platform == "linux", "requires real Linux /proc")
    def test_native_proc_sampler(self):
        sampler = HostSampler()
        self.assertEqual(sampler.backend, "/proc")
        self.assertGreater(sampler.read()["system_available_memory_mb"], 0)
        sample = sampler.read()
        for key in ("system_cpu_pct", "system_cpu_steal_pct"):
            if key in sample:
                self.assertGreaterEqual(sample[key], 0)
                self.assertLessEqual(sample[key], 100)

    def check_worker_cleanup(self, behavior):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pid_path, log_path = root / "descendant.pid", root / "worker.log"
            # The descendant deliberately holds the stdout pipe open, like a
            # surviving NVIDIA sampler after a native worker crash.
            script = ("import subprocess, sys, time\n"
                      "from pathlib import Path\n"
                      "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
                      f"Path({str(pid_path)!r}).write_text(str(child.pid))\n"
                      "print('case: config test=linux', flush=True)\n"
                      + ("time.sleep(30)\n" if behavior in ("timeout", "interrupt") else
                         f"sys.exit({3 if behavior == 'failed' else 0})\n"))
            original_popen = subprocess.Popen
            processes = []

            def launch(command, **kwargs):
                self.assertTrue(kwargs["start_new_session"])
                child = original_popen([sys.executable, "-u", "-c", script], **kwargs)
                processes.append(child)
                if behavior in ("timeout", "interrupt"):
                    original_wait = child.wait

                    def wait(timeout=None):
                        if timeout is None:
                            return original_wait()
                        try:
                            return original_wait(timeout=.5)
                        except subprocess.TimeoutExpired:
                            if behavior == "interrupt":
                                raise KeyboardInterrupt()
                            raise

                    child.wait = wait
                return child

            job = {"case_id": "case", "settings": {
                "worker_retries": 0, "timeout_s": 1, "warmup": 0,
                "repeats": 1, "timings": ["throughput"]}}
            try:
                with patch.object(experiment.subprocess, "Popen", launch):
                    if behavior == "ok":
                        attempts = experiment.run_worker_process(job, root / "job.json", log_path)
                        self.assertEqual(attempts[0]["exit_code"], 0)
                    else:
                        expected = {"failed": RuntimeError, "timeout": subprocess.TimeoutExpired,
                                    "interrupt": KeyboardInterrupt}[behavior]
                        with self.assertRaises(expected) as caught:
                            experiment.run_worker_process(job, root / "job.json", log_path)
                        self.assertEqual(len(caught.exception.worker_attempts), 1)
                self.assertEqual(len(processes), 1)
                self.assertIsNotNone(processes[0].returncode)
                if pid_path.exists():
                    child_stat = Path("/proc") / pid_path.read_text() / "stat"
                    deadline = monotonic() + 2
                    while child_stat.exists():
                        if child_stat.read_text().split()[2] == "Z":
                            break  # Dead, awaiting its adoptive parent's reap.
                        self.assertLess(monotonic(), deadline, "descendant is still running")
                        sleep(.02)
                self.assertIn("case: config", log_path.read_text())
            finally:
                for process in processes:
                    experiment.terminate_worker_tree(process)

    @unittest.skipUnless(sys.platform == "linux", "requires POSIX process groups")
    def test_success_removes_surviving_child_and_closes_pipe(self):
        self.check_worker_cleanup("ok")

    @unittest.skipUnless(sys.platform == "linux", "requires POSIX process groups")
    def test_failure_removes_surviving_child_and_closes_pipe(self):
        self.check_worker_cleanup("failed")

    @unittest.skipUnless(sys.platform == "linux", "requires POSIX process groups")
    def test_timeout_terminates_worker_and_descendant(self):
        self.check_worker_cleanup("timeout")

    @unittest.skipUnless(sys.platform == "linux", "requires POSIX process groups")
    def test_interrupt_terminates_worker_and_descendant(self):
        self.check_worker_cleanup("interrupt")


if __name__ == "__main__":
    unittest.main(verbosity=2)
