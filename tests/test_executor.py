"""执行器安全逻辑测试。"""
from src.executor.devices import ActuatorBus


def test_auto_timeout_uses_requested_duration_not_global_cap():
    bus = ActuatorBus()
    bus.apply([{"device": "drip_valve", "op": "open", "duration_min": 10.0, "reason": "test"}])
    valve = bus.devices["drip_valve"]
    valve._started_monotonic -= 11 * 60

    events = bus.auto_timeout_check(max_minutes=30.0)

    assert valve.state == "closed"
    assert len(events) == 1
    assert events[0]["device"] == "drip_valve"
    assert events[0]["operation"] == "close"


def test_auto_timeout_still_enforces_global_safety_cap():
    bus = ActuatorBus()
    bus.apply([{"device": "spray_boom", "op": "on", "duration_min": 45.0, "reason": "test"}])
    boom = bus.devices["spray_boom"]
    boom._started_monotonic -= 31 * 60

    events = bus.auto_timeout_check(max_minutes=30.0)

    assert boom.state == "off"
    assert events[0]["operation"] == "off"


def test_auto_timeout_stops_device_without_requested_duration():
    bus = ActuatorBus()
    bus.apply([{"device": "drip_valve", "op": "open", "reason": "test"}])
    valve = bus.devices["drip_valve"]
    valve._started_monotonic -= 31 * 60

    events = bus.auto_timeout_check(max_minutes=30.0)

    assert valve.state == "closed"
    assert len(events) == 1
