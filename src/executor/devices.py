"""执行器状态机：滴灌阀门 + 喷杆喷头（软件模拟，预留真实硬件转发）。

设备模型：
  - drip_valve（滴灌阀门）  : state=open/closed，流量 flow_rate(L/h)
  - spray_boom（喷杆喷头）  : state=on/off，喷头压力 nozzle_pressure(bar)

每条动作产生一个事件 dict：
  {kind:"actuator_event", device, operation, reason, duration_min, state, ts}

状态快照（供网页渲染）：
  {drip_valve:{state, flow_rate, since}, spray_boom:{state, nozzle_pressure, since}}
"""
from __future__ import annotations

import threading
import time
from datetime import datetime
from typing import Callable

# 设备默认参数
DRIP_FLOW_RATE_LH = 120.0      # 滴灌流量 L/h
SPRAY_PRESSURE_BAR = 3.5       # 喷杆喷头工作压力 bar


def _ts() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _now_epoch() -> float:
    return time.time()


class DripValve:
    """滴灌阀门。"""

    def __init__(self):
        self.state = "closed"       # open / closed
        self.since = _ts()
        self.flow_rate = 0.0
        self.duration_min: float | None = None
        self._started_monotonic: float | None = None

    def open(self, duration_min: float | None = None, reason: str = "") -> dict:
        if self.state == "open":
            return self.event("open", reason, duration_min)
        self.state = "open"
        self.since = _ts()
        self.flow_rate = DRIP_FLOW_RATE_LH
        self.duration_min = duration_min
        self._started_monotonic = time.monotonic()
        return self.event("open", reason, duration_min)

    def close(self, reason: str = "") -> dict:
        if self.state == "closed":
            return self.event("close", reason, None)
        self.state = "closed"
        self.since = _ts()
        self.flow_rate = 0.0
        self.duration_min = None
        self._started_monotonic = None
        return self.event("close", reason, None)

    def event(self, operation: str, reason: str, duration_min: float | None) -> dict:
        return {
            "kind": "actuator_event",
            "device": "drip_valve",
            "operation": operation,
            "reason": reason,
            "duration_min": duration_min,
            "state": self.state,
            "flow_rate": self.flow_rate,
            "since": self.since,
            "ts": _ts(),
        }

    def snapshot(self) -> dict:
        return {
            "state": self.state,
            "flow_rate": self.flow_rate,
            "since": self.since,
            "duration_min": self.duration_min,
        }


class SprayBoom:
    """喷杆喷头（自动打药/喷灌降温共用，通过 duration_min 区分作业）。"""

    def __init__(self):
        self.state = "off"          # on / off
        self.since = _ts()
        self.nozzle_pressure = 0.0
        self.duration_min: float | None = None
        self._started_monotonic: float | None = None

    def on(self, duration_min: float | None = None, reason: str = "") -> dict:
        if self.state == "on":
            return self.event("on", reason, duration_min)
        self.state = "on"
        self.since = _ts()
        self.nozzle_pressure = SPRAY_PRESSURE_BAR
        self.duration_min = duration_min
        self._started_monotonic = time.monotonic()
        return self.event("on", reason, duration_min)

    def off(self, reason: str = "") -> dict:
        if self.state == "off":
            return self.event("off", reason, None)
        self.state = "off"
        self.since = _ts()
        self.nozzle_pressure = 0.0
        self.duration_min = None
        self._started_monotonic = None
        return self.event("off", reason, None)

    def event(self, operation: str, reason: str, duration_min: float | None) -> dict:
        return {
            "kind": "actuator_event",
            "device": "spray_boom",
            "operation": operation,
            "reason": reason,
            "duration_min": duration_min,
            "state": self.state,
            "nozzle_pressure": self.nozzle_pressure,
            "since": self.since,
            "ts": _ts(),
        }

    def snapshot(self) -> dict:
        return {
            "state": self.state,
            "nozzle_pressure": self.nozzle_pressure,
            "since": self.since,
            "duration_min": self.duration_min,
        }


class ActuatorBus:
    """执行器总线：统一 apply(actions) 并广播状态变化事件。"""

    def __init__(self):
        self.devices = {"drip_valve": DripValve(), "spray_boom": SprayBoom()}
        self._listeners: list[Callable[[dict], None]] = []
        self._lock = threading.Lock()

    # ---------- 事件订阅 ----------
    def subscribe(self, listener: Callable[[dict], None]) -> None:
        self._listeners.append(listener)

    def _broadcast(self, event: dict) -> None:
        for fn in list(self._listeners):
            try:
                fn(event)
            except Exception:  # noqa: BLE001 - 单个监听器异常不影响执行器
                pass

    # ---------- 执行 ----------
    def apply(self, actions: list[dict]) -> list[dict]:
        """按 Agent 指令操作执行器。action: {device, op, duration_min?, reason?}"""
        events: list[dict] = []
        with self._lock:
            for a in actions or []:
                dev = self.devices.get(a.get("device"))
                if dev is None:
                    continue
                op = a.get("op")
                reason = a.get("reason", "")
                dur = a.get("duration_min")
                if op == "open":
                    ev = dev.open(dur, reason)
                elif op == "close":
                    ev = dev.close(reason)
                elif op == "on":
                    ev = dev.on(dur, reason)
                elif op == "off":
                    ev = dev.off(reason)
                else:
                    continue
                events.append(ev)
                self._broadcast(ev)
        return events

    def snapshot(self) -> dict:
        return {name: dev.snapshot() for name, dev in self.devices.items()}

    def reset(self) -> None:
        """全部设备复位（停止作业）。"""
        with self._lock:
            self.devices["drip_valve"].close("系统复位")
            self.devices["spray_boom"].off("系统复位")

    def auto_timeout_check(self, max_minutes: float = 30.0) -> list[dict]:
        """安全检查：达到计划时长或全局安全上限时自动停止。

        max_minutes 是兜底上限，不应当延长设备请求的较短计划时长：
        duration_min=10 时应运行约 10 分钟，而不是被 max_minutes=30 拖到 30 分钟。
        """
        events: list[dict] = []
        with self._lock:
            for name, dev in self.devices.items():
                state = dev.state
                active = (state == "open" and name == "drip_valve") or \
                         (state == "on" and name == "spray_boom")
                if not active:
                    continue
                started = getattr(dev, "_started_monotonic", None)
                if started is None:
                    started = datetime.fromisoformat(dev.since).timestamp()
                    elapsed_min = (_now_epoch() - started) / 60.0
                else:
                    elapsed_min = (time.monotonic() - started) / 60.0
                requested = float(dev.duration_min) if dev.duration_min else float(max_minutes)
                limit = min(max(requested, 0.01), float(max_minutes))
                if elapsed_min >= limit:
                    ev = dev.close(f"自动停止（已运行 {elapsed_min:.0f} 分钟）") \
                        if name == "drip_valve" else dev.off(f"自动停止（已运行 {elapsed_min:.0f} 分钟）")
                    events.append(ev)
                    self._broadcast(ev)
        return events


DEFAULT_SNAPSHOT = {
    "drip_valve": {"state": "closed", "flow_rate": 0.0, "since": "", "duration_min": None},
    "spray_boom": {"state": "off", "nozzle_pressure": 0.0, "since": "", "duration_min": None},
}

# 模块级单例（供 Agent / WS 服务 / 页面复用）
bus = ActuatorBus()
