"""Agent 主循环：后台线程定时执行「采集→推理→决策→执行→落库→广播」。

典型一轮（演示链路）：
  USB摄像头/演示帧 → OpenCV → YOLO 推理 → Agent 决策
  → 执行器状态更新 → 日志落库 → WebSocket 广播 → 网页动画展示

线程安全：start/stop/step 均可从 REST API / Streamlit 调用。
"""
from __future__ import annotations

import os
import threading
from datetime import datetime
from typing import Callable

from .. import config, db, risk
from ..executor import bus
from ..sensor import sensor
from ..vision import annotate, camera, detector
from ..weather import WeatherClient
from .llm_decision import decide

DEFAULT_GARDEN_KEY = "longjing"


class AgentRunner:
    """智能执行 Agent 主循环。"""

    def __init__(self, garden_key: str = DEFAULT_GARDEN_KEY, interval: float = 10.0):
        self.garden_key = garden_key
        self.interval = max(3.0, float(interval))
        self._thread: threading.Thread | None = None
        self._stop_evt = threading.Event()
        self._lock = threading.Lock()
        self._listeners: list[Callable[[dict], None]] = []
        self.running = False
        self.steps = 0
        self.latest: dict = {}
        self._decision_state: dict = {}

    # ---------- 事件订阅 ----------
    def subscribe(self, listener: Callable[[dict], None]) -> None:
        self._listeners.append(listener)

    def _broadcast(self, event: dict) -> None:
        for fn in list(self._listeners):
            try:
                fn(event)
            except Exception:  # noqa: BLE001
                pass

    # ---------- 生命周期 ----------
    def start(self) -> bool:
        with self._lock:
            if self.running:
                return False
            self.running = True
            self._stop_evt.clear()
            self._thread = threading.Thread(target=self._loop, daemon=True,
                                            name="tea-agent-runner")
            self._thread.start()
            return True

    def stop(self) -> bool:
        with self._lock:
            if not self.running:
                return False
            self.running = False
            self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        return True

    def _loop(self) -> None:
        while not self._stop_evt.is_set():
            try:
                self.step()
            except Exception as e:  # noqa: BLE001 - 单轮失败不中断循环
                self._broadcast({"kind": "agent_error", "message": str(e), "ts": _ts()})
            self._stop_evt.wait(self.interval)

    # ---------- 单轮决策 ----------
    def step(self) -> dict:
        """执行一轮完整决策，返回并广播最新状态快照。"""
        garden = self._garden()

        # 1) 传感器
        soil = sensor.read()
        try:
            db.log_sensor(self.garden_key, soil["type"], soil["value"], soil["unit"], soil.get("meta"))
        except Exception:  # noqa: BLE001 - 数据库不可用时降级跳过
            pass

        # 2) 视觉（帧来源：摄像头实拍 / 演示帧，由 camera.last_source 标记）
        frame = camera.read_frame()
        det_result = detector.detect(frame, source_hint=getattr(camera, "last_source", "demo_frame"))
        frame_shape = det_result.get("frame_shape") or (list(frame.shape[:2]) if frame is not None else [480, 640])

        # 3) 气象（网络不可用/无 Key 时降级演示数据）
        weather, risk_result = self._weather_and_risk(garden)

        # 4) 决策
        inputs = {
            "soil": soil,
            "detections": det_result,
            "weather": weather,
            "risk": risk_result,
            "actuators": bus.snapshot(),
        }
        decision = decide(inputs, self._decision_state)

        # 5) 执行
        events = bus.apply(decision["actions"])
        bus.auto_timeout_check()

        # 5.1) 灌溉响应：阀门开启时土壤湿度逐步回升（模拟滴灌回水），
        #      达到 IRRIGATE_STOP 后 Agent 会自动关阀 —— 形成「缺水→灌溉→回水→停灌」闭环
        if bus.snapshot()["drip_valve"]["state"] == "open":
            sensor.irrigate(0.6)

        # 6) 落库（数据库不可用时降级跳过，不影响决策链路）
        try:
            for a in decision["actions"]:
                db.log_actuator(self.garden_key, a["device"], a["op"], a.get("duration_min"), a.get("reason"))
            db.log_agent_decision(self.garden_key, decision["decision_type"], decision["summary"],
                                  inputs={"soil": soil, "detections": det_result["summary"],
                                          "weather": self._weather_light(weather)},
                                  actions=decision["actions"],
                                  overall_level=decision["overall_level"])
        except Exception:  # noqa: BLE001 - 落库失败不影响本轮执行与推送
            pass

        # 7) 标注帧（供网页展示）
        image_b64 = ""
        if frame is not None:
            annotated = annotate.draw_detections(frame, det_result)
            image_b64 = annotate.frame_to_base64(annotated)

        # 8) 状态快照 + 广播
        snapshot = {
            "ts": _ts(),
            "garden": {"key": self.garden_key, "name": garden.get("name", self.garden_key)},
            "camera": {"available": camera.available, "backend": det_result.get("backend", "sim"),
                       "source": det_result.get("source", "demo_frame")},
            "soil": soil,
            "detections": {**det_result, "image": image_b64},
            "actuator": bus.snapshot(),
            "decision": decision,
            "events": events,
            "steps": self.steps,
            "running": self.running,
            "interval": self.interval,
        }
        self.steps += 1
        snapshot["steps"] = self.steps
        with self._lock:
            self.latest = snapshot
        self._broadcast({"kind": "state_update", "data": snapshot})
        return snapshot

    # ---------- 辅助 ----------
    def _garden(self) -> dict:
        g = config.get_garden(self.garden_key)
        if g is None:
            gardens = config.load_gardens()
            g = gardens[0] if gardens else {"id": self.garden_key, "name": self.garden_key}
        return {**g, "key": self.garden_key}

    def _weather_and_risk(self, garden: dict) -> tuple[dict, dict]:
        weather: dict = {}
        risk_result: dict = {"items": [], "overall_level": "low"}
        try:
            client = WeatherClient()
            loc = garden.get("location_id") or ""
            if garden.get("lon") is not None and garden.get("lat") is not None:
                loc = f"{garden['lon']},{garden['lat']}"
            if loc:
                weather = {
                    "now": client.get_now(loc),
                    "daily": client.get_forecast(loc),
                }
                n = weather["now"].get("now", weather["now"])
                dl = weather["daily"].get("daily", [])
                if n and dl:
                    risk_result = risk.RiskEngine().assess_all(n, dl)
        except Exception:  # noqa: BLE001 - 气象不可用时决策仅依赖传感器与视觉
            pass
        return weather, risk_result

    @staticmethod
    def _weather_light(weather: dict) -> dict:
        n = (weather.get("now") or {}).get("now", {})
        dl = (weather.get("daily") or {}).get("daily", [])
        return {
            "temp": n.get("temp"), "humidity": n.get("humidity"),
            "text": n.get("text"), "windScale": n.get("windScale"),
            "precip_3d": [d.get("precip") for d in dl[:3]],
            "temp_3d": [f"{d.get('tempMin')}~{d.get('tempMax')}" for d in dl[:3]],
        }


def _ts() -> str:
    return datetime.now().isoformat(timespec="seconds")


# 模块级单例（供 WS 服务 / Streamlit 页面共享）
runner = AgentRunner(interval=float(os.getenv("AGENT_INTERVAL", "10")))
