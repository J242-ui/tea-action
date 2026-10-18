"""Agent 主循环：后台线程定时执行「采集→推理→决策→执行→落库→广播」。

典型一轮（演示链路）：
  USB摄像头/演示帧 → OpenCV → YOLO 推理 → Agent 决策
  → 执行器状态更新 → 日志落库 → WebSocket 广播 → 网页动画展示

线程安全：start/stop/step 均可从 REST API / Streamlit 调用。
"""
from __future__ import annotations

import os
import threading
import time
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

    def __init__(self, garden_key: str = DEFAULT_GARDEN_KEY, interval: float = 5.0,
                 refresh_mode: str = "interval"):
        self.garden_key = garden_key
        # refresh_mode: "interval" 自定义间隔整轮执行；"stream" 持续录像（画面连续推送，决策按间隔节流）
        self.refresh_mode = "stream" if str(refresh_mode).strip().lower() == "stream" else "interval"
        self.interval = max(0.5, float(interval))
        self._thread: threading.Thread | None = None
        self._stop_evt = threading.Event()
        self._lock = threading.Lock()
        # REST 手动单步与后台循环可能同时触发，整轮硬件/决策操作必须串行。
        self._step_lock = threading.Lock()
        self._listeners: list[Callable[[dict], None]] = []
        self.running = False
        self.steps = 0
        self.latest: dict = {}
        self.latest_frame: dict = {}  # 最近一帧画面（stream_frame 与完整轮次都会更新，供 /api/frame 与主站实时画面读取）
        self._decision_state: dict = {}
        self._last_full = 0.0  # 持续录像模式下最近一次完整决策的单调钟时间

    # ---------- 事件订阅 ----------
    def subscribe(self, listener: Callable[[dict], None]) -> None:
        self._listeners.append(listener)

    def _broadcast(self, event: dict) -> None:
        for fn in list(self._listeners):
            try:
                fn(event)
            except Exception:  # noqa: BLE001
                pass

    # ---------- 配置 ----------
    def configure(self, refresh_mode: str | None = None, interval: float | None = None) -> dict:
        """运行时更新刷新模式与间隔（线程安全）。

        refresh_mode: "interval"（自定义刷新时间） / "stream"（持续录像）
        interval: 刷新间隔秒数，最小 0.5 秒。
        """
        with self._lock:
            if refresh_mode is not None:
                self.refresh_mode = "stream" if str(refresh_mode).strip().lower() == "stream" else "interval"
            if interval is not None:
                self.interval = max(0.5, float(interval))
        return {"refresh_mode": self.refresh_mode, "interval": self.interval}

    # ---------- 生命周期 ----------
    def start(self) -> bool:
        with self._lock:
            if self.running:
                return False
            # 上一个循环线程还卡在采集/推理里未退出时拒绝再启动，
            # 避免两个线程并发抢摄像头/推理导致越卡越死
            if self._thread is not None and self._thread.is_alive():
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
            # 线程卡在不可中断的采集调用（摄像头句柄阻塞）时最多等 5 秒；
            # 即使 join 超时，stop 标记已置位，线程会在下一轮循环入口退出
            self._thread.join(timeout=5)
        return True

    def _loop(self) -> None:
        while not self._stop_evt.is_set():
            try:
                if self.refresh_mode == "stream":
                    # 持续录像：高频推流画面帧；完整决策按间隔节流。
                    # tick 间隔 0.5s（约 2fps）：视觉上已如监控般流畅；
                    # 更低（如 0.2s）会让浏览器每帧解码大图、DOM 重建，CPU 高反而更卡
                    self._stream_tick()
                    self._stop_evt.wait(0.5)
                    if time.monotonic() - self._last_full >= self.interval:
                        self._last_full = time.monotonic()
                        self.step()
                else:
                    self.step()
                    self._stop_evt.wait(self.interval)
            except Exception as e:  # noqa: BLE001 - 单轮失败不中断循环
                self._broadcast({"kind": "agent_error", "message": str(e), "ts": _ts()})
                self._stop_evt.wait(self.interval)

    def _set_latest_frame(self, det_result: dict, image_b64: str) -> None:
        """写入最近一帧画面缓存（供 /api/frame 与主站实时画面读取）。"""
        with self._lock:
            self.latest_frame = {
                "ts": _ts(),
                "source": det_result.get("source", "demo_frame"),
                "backend": det_result.get("backend", "sim"),
                "image": image_b64,
                "detections": det_result.get("detections", []),
                "summary": det_result.get("summary", {}),
            }

    def _stream_tick(self) -> None:
        """持续录像模式：只做「采集→推理→标注→推送画面帧」，不落库、不决策、不覆盖完整快照。

        前端收到 kind=stream_frame 事件后仅刷新画面区；完整快照（决策/执行器/传感器）
        仍由按间隔节流的 step() 生成并推送（kind=state_update）。
        """
        frame = camera.read_frame()
        det_result = detector.detect(frame, source_hint=getattr(camera, "last_source", "demo_frame"))
        image_b64 = ""
        if frame is not None:
            annotated = annotate.draw_detections(frame, det_result)
            image_b64 = annotate.frame_to_base64(annotated)
        self._set_latest_frame(det_result, image_b64)
        self._broadcast({
            "kind": "stream_frame",
            "data": {
                "ts": _ts(),
                "camera": {"available": camera.available, "backend": det_result.get("backend", "sim"),
                           "source": det_result.get("source", "demo_frame")},
                "detections": {**det_result, "image": image_b64},
                "steps": self.steps,
                "running": self.running,
                "interval": self.interval,
                "refresh_mode": self.refresh_mode,
            },
        })

    # ---------- 单轮决策 ----------
    def step(self) -> dict:
        """执行一轮完整决策，返回并广播最新状态快照。"""
        with self._step_lock:
            return self._step_once()

    def _step_once(self) -> dict:
        """串行执行一次完整流程（由 step() 持锁调用）。"""
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
        timeout_events = bus.auto_timeout_check()
        events.extend(timeout_events)
        for event in timeout_events:
            if event.get("device") == "spray_boom" and event.get("operation") == "off":
                # 自动停止也属于一次喷雾结束，需进入冷却期。
                self._decision_state["last_spray_end_at"] = event.get("ts")

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
        self._set_latest_frame(det_result, image_b64)

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
            "refresh_mode": self.refresh_mode,
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
# 默认 5 秒刷新；最小 0.5 秒；可通过 /api/agent/config 或「智能执行」页运行时调整
runner = AgentRunner(interval=float(os.getenv("AGENT_INTERVAL", "5")))
