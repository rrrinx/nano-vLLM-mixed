import pytest

from experiments.common.measurement import assert_idle, replay
from test_math500_experiments import Clock, TimedEngine, request


def test_online_individual_output_limits():
    clock = Clock()
    engine = TimedEngine(clock)
    run = replay(
        engine, [request("a", [1], 0), request("b", [2, 3, 4], .25)],
        99, "latency", lambda: None, 10,
        request_limits={"a": 2, "b": 1}, clock=clock, wait=clock.wait,
    )
    a, b = run["requests"]
    assert a["marks"] == [1, 1.5]
    assert b["marks"] == [1.5]
    assert a["output_tokens"] == 2 and b["output_tokens"] == 1
    assert b["ttft_ms"] == 1250
    assert b["admission_delay_ms"] == 750
    assert b["tpot_ms"] is None
    assert run["summary"]["output_tok_s"] == pytest.approx(3 / 1.5)
    assert_idle(engine)


def test_online_partial_prefill_does_not_emit_token():
    clock = Clock()
    engine = TimedEngine(clock)
    run = replay(
        engine, [request("a", [1], 0), request("b", [2] * 6, .25)],
        99, "latency", lambda: None, 10,
        request_limits={"a": 3, "b": 1}, clock=clock, wait=clock.wait,
        save_steps=True,
    )
    a, b = run["requests"]
    assert a["marks"] == [1, 1.5, 2.25]
    assert b["marks"] == [2.25]
    assert b["ttft_ms"] == 2000
    assert run["summary"]["counts"]["mixed_steps"] == 2
    assert [s["kind"] for s in run["steps"]] == ["prefill", "mixed", "mixed"]
    assert_idle(engine)