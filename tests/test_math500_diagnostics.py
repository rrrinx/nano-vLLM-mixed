"""Replay attribution, unavailable sensors, UUID selection and sampler cleanup."""

from datetime import datetime
from io import StringIO
from types import SimpleNamespace

import pytest

from experiments.common import diagnostics


def test_gpu_parser_keeps_unsupported_readings_missing_and_parses_flags():
    line = ("2026/10/06 15:40:01.123, 900, 8001, N/A, 61, 6, 7600, 300, "
            "Active, Not Active, N/A, Not Active, Active")
    stamp, sample = diagnostics.parse_gpu_line(line)
    assert stamp == datetime(2026, 10, 6, 15, 40, 1, 123000).timestamp()
    assert sample["gpu_sm_clock_mhz"] == 900
    assert sample["gpu_power_cap_sample_fraction"] == sample["gpu_idle_sample_fraction"] == 1
    assert sample["gpu_sw_thermal_sample_fraction"] == 0
    assert "gpu_power_w" not in sample and "gpu_hw_thermal_sample_fraction" not in sample
    with pytest.raises(ValueError):
        diagnostics.parse_gpu_line("driver unavailable")


def test_delayed_samples_belong_to_original_replay_and_gaps_are_excluded(monkeypatch):
    clock = SimpleNamespace(wall=100., cpu=5.)
    monkeypatch.setattr(diagnostics, "time", lambda: clock.wall)
    monkeypatch.setattr(diagnostics, "process_time", lambda: clock.cpu)
    monitor = diagnostics.DiagnosticMonitor(1., None, SimpleNamespace(type="cpu"))
    monitor.begin("throughput/1")
    clock.wall, clock.cpu = 110., 7.
    monitor.end(10.)
    clock.wall = 120.
    monitor.begin("throughput/2")
    monitor.record(105., {"gpu_sm_clock_mhz": 800, "gpu_power_cap_sample_fraction": 1}, "gpu")
    monitor.record(108., {"gpu_sm_clock_mhz": 1000, "gpu_power_cap_sample_fraction": 0}, "gpu")
    monitor.record(106., {"system_cpu_pct": 40, "system_available_memory_mb": 12000,
                         "system_cpu_steal_pct": 3}, "host")
    monitor.record(115., {"gpu_sm_clock_mhz": 200}, "gpu")
    monitor.record(122., {"gpu_sm_clock_mhz": 1500}, "gpu")
    clock.wall, clock.cpu = 125., 8.
    monitor.end(5.)
    first, second = monitor.summary("throughput/1"), monitor.summary("throughput/2")
    assert first["gpu_samples"] == 2 and first["host_samples"] == 1
    assert first["gpu_sm_clock_mhz_mean"] == 900 and first["gpu_sm_clock_mhz_min"] == 800
    assert first["gpu_sm_clock_mhz_max"] == 1000
    assert first["gpu_power_cap_sample_fraction"] == .5
    assert first["process_cpu_s"] == 2 and first["process_cpu_pct"] == 20
    assert first["system_cpu_pct_mean"] == 40 and first["system_available_memory_mb_min"] == 12000
    assert first["system_cpu_steal_pct_mean"] == 3
    assert second["gpu_samples"] == 1 and second["gpu_sm_clock_mhz_mean"] == 1500
    assert second["gpu_power_w_mean"] is None and second["cuda_free_end_mb"] is None


class FakeHost:
    backend = "test-host"

    def read(self):
        return {"system_cpu_pct": 50., "system_available_memory_mb": 10000.}


def fake_cuda(uuid="GPU-physical-test"):
    return SimpleNamespace(cuda=SimpleNamespace(
        get_device_properties=lambda device: SimpleNamespace(uuid=uuid),
        memory_reserved=lambda device: 1024 * 2**20,
        mem_get_info=lambda device: (4000 * 2**20, 8000 * 2**20),
        max_memory_reserved=lambda device: 1500 * 2**20))


def test_missing_nvidia_tool_keeps_cpu_cuda_diagnostics_and_stops_threads(monkeypatch):
    monkeypatch.setattr(diagnostics, "HostSampler", FakeHost)

    def missing(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi not installed")

    monkeypatch.setattr(diagnostics.subprocess, "Popen", missing)
    monitor = diagnostics.DiagnosticMonitor(.1, fake_cuda(), SimpleNamespace(type="cuda"))
    monitor.start()
    monitor.begin("throughput/1")
    monitor.end(1.)
    monitor.stop()
    summary = monitor.summary("throughput/1")
    assert summary["gpu_samples"] == 0 and summary["gpu_power_w_mean"] is None
    assert summary["peak_cuda_reserved_mb"] == 1500
    assert summary["cuda_reserved_start_mb"] == summary["cuda_reserved_end_mb"] == 1024
    assert summary["cuda_free_start_mb"] == summary["cuda_free_end_mb"] == 4000
    assert monitor.metadata()["gpu_identifier"] == "GPU-physical-test"
    assert "nvidia-smi not installed" in monitor.metadata()["notes"][0]
    assert all(not thread.is_alive() for thread in monitor.threads)


@pytest.mark.parametrize("raw_uuid", ["GPU-physical-test", "physical-test"])
def test_gpu_stream_uses_physical_uuid_and_is_terminated_without_file_output(monkeypatch, raw_uuid):
    monkeypatch.setattr(diagnostics, "HostSampler", FakeHost)
    epoch = datetime(2026, 10, 6, 15, 40, 0).timestamp()
    clock = SimpleNamespace(wall=epoch)
    monkeypatch.setattr(diagnostics, "time", lambda: clock.wall)
    monitor = diagnostics.DiagnosticMonitor(.1, fake_cuda(raw_uuid), SimpleNamespace(type="cuda", index=0))
    monitor.begin("throughput/1")
    stamp = datetime.fromtimestamp(epoch + .5).strftime("%Y/%m/%d %H:%M:%S.%f")
    calls, process_state = [], SimpleNamespace(exited=False, terminated=False)

    def popen(command, **kwargs):
        calls.append(command)
        assert "--id=GPU-physical-test" in command and "--loop-ms=100" in command
        assert kwargs["stdout"] == diagnostics.subprocess.PIPE
        line = f"{stamp}, 900, 8001, 20, 60, 10, 7000, 900, Not Active, Not Active, Not Active, Not Active, Active\n"
        return SimpleNamespace(stdout=StringIO(line),
                               poll=lambda: 0 if process_state.exited else None,
                               terminate=lambda: setattr(process_state, "terminated", True),
                               wait=lambda timeout: setattr(process_state, "exited", True))

    monkeypatch.setattr(diagnostics.subprocess, "Popen", popen)
    monitor.start()
    clock.wall = epoch + 1.
    monitor.end(1.)
    monitor.stop()
    assert len(calls) == 1 and process_state.terminated and process_state.exited
    assert monitor.summary("throughput/1")["gpu_samples"] == 1
    assert monitor.summary("throughput/1")["gpu_idle_sample_fraction"] == 1
    assert all(not thread.is_alive() for thread in monitor.threads)


def test_native_host_sample_has_valid_available_memory_and_cpu_delta():
    sampler = diagnostics.HostSampler()
    first, second = sampler.read(), sampler.read()
    assert first["system_available_memory_mb"] > 0
    assert second["system_available_memory_mb"] > 0
    if "system_cpu_pct" in second:
        assert 0 <= second["system_cpu_pct"] <= 100
