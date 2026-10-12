"""Optional host/NVIDIA sampling, aggregated in RAM without per-step probes."""

import csv
import ctypes
import math
import os
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from time import process_time, time


GPU_FIELDS = (
    ("clocks.sm", "gpu_sm_clock_mhz"),
    ("clocks.mem", "gpu_memory_clock_mhz"),
    ("power.draw", "gpu_power_w"),
    ("temperature.gpu", "gpu_temperature_c"),
    ("utilization.gpu", "gpu_utilization_pct"),
    ("memory.used", "gpu_memory_used_mb"),
    ("memory.free", "gpu_memory_free_mb"),
)
GPU_FLAGS = (
    ("clocks_event_reasons.sw_power_cap", "gpu_power_cap_sample_fraction"),
    ("clocks_event_reasons.sw_thermal_slowdown", "gpu_sw_thermal_sample_fraction"),
    ("clocks_event_reasons.hw_thermal_slowdown", "gpu_hw_thermal_sample_fraction"),
    ("clocks_event_reasons.hw_power_brake_slowdown", "gpu_power_brake_sample_fraction"),
    ("clocks_event_reasons.gpu_idle", "gpu_idle_sample_fraction"),
)
DIAGNOSTIC_FIELDS = (
    "process_cpu_s", "process_cpu_pct", "peak_cuda_reserved_mb",
    "cuda_reserved_start_mb", "cuda_reserved_end_mb", "cuda_free_start_mb", "cuda_free_end_mb",
    "gpu_samples", "host_samples", "gpu_sm_clock_mhz_mean", "gpu_sm_clock_mhz_min",
    "gpu_sm_clock_mhz_max", "gpu_memory_clock_mhz_mean", "gpu_memory_clock_mhz_min",
    "gpu_power_w_mean", "gpu_power_w_max", "gpu_temperature_c_max", "gpu_utilization_pct_mean",
    "gpu_memory_used_mb_max", "gpu_memory_free_mb_min", "system_cpu_pct_mean",
    "system_available_memory_mb_min", "system_cpu_steal_pct_mean", *(name for _, name in GPU_FLAGS),
)


def parse_gpu_line(line):
    """Driver timestamps associate delayed pipe reads with their original replay."""
    values = next(csv.reader([line]))
    if len(values) != 1 + len(GPU_FIELDS) + len(GPU_FLAGS):
        raise ValueError("unexpected nvidia-smi sample format")
    stamp = datetime.strptime(values[0].strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()
    sample = {}
    for (_, name), value in zip(GPU_FIELDS, values[1:]):
        try:
            number = float(value.strip())
        except ValueError:
            continue  # N/A means unavailable, never zero.
        if math.isfinite(number):
            sample[name] = number
    for (_, name), value in zip(GPU_FLAGS, values[1 + len(GPU_FIELDS):]):
        flag = value.strip().lower()
        if flag in ("active", "not active"):
            sample[name] = float(flag == "active")
    return stamp, sample


class HostSampler:
    """Global CPU busy fraction and available physical RAM; no extra packages."""

    def __init__(self, *, proc_root="/proc", platform_name=None):
        self.previous = None
        self.windows = (platform_name or os.name) == "nt"
        self.proc_root = Path(proc_root)
        self.backend = "GetSystemTimes/GlobalMemoryStatusEx" if self.windows else "/proc"
        if self.windows:
            self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            self.kernel.GetSystemTimes.argtypes = [ctypes.POINTER(ctypes.c_uint64)] * 3

            class MemoryStatus(ctypes.Structure):
                _fields_ = [("length", ctypes.c_uint32), ("load", ctypes.c_uint32)] + [
                    (name, ctypes.c_uint64) for name in
                    ("total_phys", "available_phys", "total_page", "available_page",
                     "total_virtual", "available_virtual", "available_extended")]

            self.memory_type = MemoryStatus
            self.kernel.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(MemoryStatus)]

    def read(self):
        steal = None
        if self.windows:
            idle, kernel, user = ctypes.c_uint64(), ctypes.c_uint64(), ctypes.c_uint64()
            if not self.kernel.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
                raise ctypes.WinError(ctypes.get_last_error())
            total, idle = kernel.value + user.value, idle.value
            memory = self.memory_type()
            memory.length = ctypes.sizeof(memory)
            if not self.kernel.GlobalMemoryStatusEx(ctypes.byref(memory)):
                raise ctypes.WinError(ctypes.get_last_error())
            available = memory.available_phys / 2**20
        else:
            ticks = [int(x) for x in (self.proc_root / "stat").read_text().splitlines()[0].split()[1:9]]
            total, idle = sum(ticks), ticks[3] + ticks[4]
            # guest/guest_nice are already included in user/nice. Count only
            # the first eight fields, with steal tracked independently.
            steal = ticks[7]
            memory = dict(line.split(":", 1) for line in (self.proc_root / "meminfo").read_text().splitlines())
            available = int(memory["MemAvailable"].split()[0]) / 1024
        values = {"system_available_memory_mb": available}
        if self.previous is not None:
            previous_total, previous_idle, previous_steal = self.previous
            if total > previous_total:
                values["system_cpu_pct"] = max(0., min(100.,
                    100 * (1 - (idle - previous_idle) / (total - previous_total))))
                if steal is not None:
                    values["system_cpu_steal_pct"] = max(0., min(100.,
                        100 * (steal - previous_steal) / (total - previous_total)))
        self.previous = total, idle, steal
        return values


class DiagnosticMonitor:
    def __init__(self, interval_s, torch, device):
        self.interval_s, self.torch, self.device = interval_s, torch, device
        self.lock, self.stop_event = threading.Lock(), threading.Event()
        self.phases, self.active = {}, None
        self.notes, self.threads, self.gpu_process = [], [], None
        self.host = None
        self.gpu_identifier = None

    def note(self, message):
        with self.lock:
            message = str(message).splitlines()[0][:300] if str(message) else "unknown sampling error"
            if message not in self.notes and len(self.notes) < 8:
                self.notes.append(message)

    def start(self):
        try:
            self.host = HostSampler()
            self.host.read()  # Prime CPU deltas outside every replay timer.
            self._start_thread(self._read_host)
        except (OSError, ValueError, KeyError) as exc:
            self.note(f"host sampling unavailable: {exc}")
        if self.device.type != "cuda":
            return
        try:
            # CUDA logical indices may be remapped by CUDA_VISIBLE_DEVICES.
            # Select the NVIDIA process by the physical UUID from PyTorch.
            self.gpu_identifier = str(self.torch.cuda.get_device_properties(self.device).uuid)
            if not self.gpu_identifier.startswith(("GPU-", "MIG-")):
                self.gpu_identifier = "GPU-" + self.gpu_identifier
            query = ["timestamp"] + [field for field, _ in GPU_FIELDS + GPU_FLAGS]
            command = ["nvidia-smi", "--id=" + self.gpu_identifier,
                       "--query-gpu=" + ",".join(query), "--format=csv,noheader,nounits",
                       "--loop-ms=" + str(round(self.interval_s * 1000))]
            self.gpu_process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            self._start_thread(self._read_gpu)
        except (OSError, AttributeError) as exc:
            self.note(f"GPU sampling unavailable: {exc}")

    def _start_thread(self, function):
        thread = threading.Thread(target=function, daemon=True)
        self.threads.append(thread)
        thread.start()

    def _read_host(self):
        while not self.stop_event.wait(self.interval_s):
            try:
                self.record(time(), self.host.read(), "host")
            except (OSError, ValueError, KeyError) as exc:
                self.note(f"host sampling unavailable: {exc}")
                return

    def _read_gpu(self):
        for line in self.gpu_process.stdout:
            try:
                stamp, values = parse_gpu_line(line)
                self.record(stamp, values, "gpu")
            except (ValueError, OSError):
                if line.strip():
                    self.note(f"GPU sampling: {line.strip()}")

    def record(self, stamp, values, source):
        with self.lock:
            for phase in reversed(list(self.phases.values())):
                if phase["start"] <= stamp and (phase["end"] is None or stamp < phase["end"]):
                    phase[source + "_samples"] += 1
                    for name, value in values.items():
                        # n/sum/min/max use constant storage, even for long runs.
                        stats = phase["stats"].setdefault(name, [0, 0., value, value])
                        stats[0] += 1
                        stats[1] += value
                        stats[2], stats[3] = min(stats[2], value), max(stats[3], value)
                    break

    def _cuda_memory(self):
        if self.device.type != "cuda":
            return None, None
        return (self.torch.cuda.memory_reserved(self.device) / 2**20,
                self.torch.cuda.mem_get_info(self.device)[0] / 2**20)

    def begin(self, label):
        reserved, free = self._cuda_memory()
        with self.lock:
            if self.active is not None or label in self.phases:
                raise RuntimeError("diagnostic replays must have unique nonoverlapping labels")
            self.active = label
            self.phases[label] = {"start": time(), "end": None, "cpu_start": process_time(),
                                  "gpu_samples": 0, "host_samples": 0, "stats": {},
                                  "values": {"cuda_reserved_start_mb": reserved, "cuda_free_start_mb": free}}

    def end(self, elapsed):
        cpu_end, end = process_time(), time()
        reserved, free = self._cuda_memory()
        peak = (self.torch.cuda.max_memory_reserved(self.device) / 2**20
                if self.device.type == "cuda" else None)
        with self.lock:
            phase = self.phases[self.active]
            cpu_s = cpu_end - phase["cpu_start"]
            phase["end"] = end
            phase["values"].update(process_cpu_s=cpu_s, process_cpu_pct=100 * cpu_s / elapsed,
                                   cuda_reserved_end_mb=reserved, cuda_free_end_mb=free,
                                   peak_cuda_reserved_mb=peak)
            self.active = None

    def summary(self, label):
        with self.lock:
            phase = self.phases[label]
            values = dict.fromkeys(DIAGNOSTIC_FIELDS)
            values.update(phase["values"], gpu_samples=phase["gpu_samples"], host_samples=phase["host_samples"])
            for name, (n, total, minimum, maximum) in phase["stats"].items():
                if name.endswith("_sample_fraction"):
                    values[name] = total / n
                else:
                    for suffix, value in (("mean", total / n), ("min", minimum), ("max", maximum)):
                        key = f"{name}_{suffix}"
                        if key in values:
                            values[key] = value
            return values

    def stop(self):
        self.stop_event.set()
        if self.gpu_process is not None:
            if self.gpu_process.poll() is None:
                self.gpu_process.terminate()
                try:
                    self.gpu_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.gpu_process.kill()
                    self.gpu_process.wait(timeout=5)
            for thread in self.threads:
                thread.join(timeout=5)
            self.gpu_process.stdout.close()
        else:
            for thread in self.threads:
                thread.join(timeout=5)

    def metadata(self):
        return {"enabled": True, "interval_s": self.interval_s,
                "gpu_backend": "nvidia-smi" if self.device.type == "cuda" else None,
                "gpu_identifier": self.gpu_identifier,
                "host_backend": self.host.backend if self.host else None, "notes": list(self.notes)}
