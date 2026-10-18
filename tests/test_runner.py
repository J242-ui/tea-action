"""Agent 并发控制测试。"""
import threading
import time

from src.agent.runner import AgentRunner


def test_step_calls_are_serialized(monkeypatch):
    runner = AgentRunner(interval=1.0)
    state = {"active": 0, "max_active": 0}
    state_lock = threading.Lock()

    def fake_step_once():
        with state_lock:
            state["active"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
        time.sleep(0.05)
        with state_lock:
            state["active"] -= 1
        return {}

    monkeypatch.setattr(runner, "_step_once", fake_step_once)
    threads = [threading.Thread(target=runner.step) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert state["max_active"] == 1
