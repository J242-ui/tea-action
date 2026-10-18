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
from ..vision import annotate, camera, camera_manager, detector
from ..weather import WeatherClient
from .daily_recorder import daily_recorder
from .llm_decision import decide

DEFAULT_GARDEN_KEY = "longjing"


class AgentRunner:
    """智能执行 Agent 主循环。"""

    def __init__(self, garden_key: str = DEFAULT_GARDEN_KEY, interval: float = 5.0,
                 refresh_mode: str = "interval"):
        self.garden_key = garden_key
        # refresh_mode: "interval" 自定义间隔整轮决策；"stream" 持续录像（画面连续推送，决策按间隔节流）
        self.refresh_mode = "stream" if str(refresh_mode).strip().lower() == "stream" else "interval"
        self.interval = max(0.5, float(interval))
        self._thread: threading.Thread | None = None          # 推流主循环线程
        self._step_thread: threading.Thread | None = None     # 决策线程（与推流解耦）
        self._stop_evt = threading.Event()
        self._lock = threading.Lock()
        # REST 手动单步与后台决策线程可能同时触发，整轮决策操作必须串行。
        self._step_lock = threading.Lock()
        self._listeners: list[Callable[[dict], None]] = []
        self.running = False
        self.steps = 0
        self.latest: dict = {}
        self.latest_frame: dict = {}  # 最近一帧画面（stream_frame 与完整轮次都会更新，供 /api/frame 与主站实时画面读取）
        self._decision_state: dict = {}
        # 视觉缓存：推流线程持续采集+推理，决策线程复用最新结果，两者互不阻塞
        self._cams_cache: dict | None = None

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
            # 推流主循环线程：只做「采集缓存→推理→标注→WS 推流」，永不阻塞
            self._thread = threading.Thread(target=self._loop, daemon=True,
                                            name="tea-agent-runner")
            self._thread.start()
            # 决策线程：按 interval 节流执行完整决策（LLM/天气/写库慢操作
            # 只影响本线程，绝不冻结推流画面）
            self._step_thread = threading.Thread(target=self._step_loop, daemon=True,
                                                 name="tea-agent-decider")
            self._step_thread.start()
            return True

    def stop(self) -> bool:
        with self._lock:
            if not self.running:
                return False
            self.running = False
            self._stop_evt.set()
        for t in (self._thread, self._step_thread):
            if t is not None:
                # 线程卡在不可中断的采集调用（摄像头句柄阻塞）时最多等 5 秒；
                # 即使 join 超时，stop 标记已置位，线程会在下一轮循环入口退出
                t.join(timeout=5)
        return True

    def _loop(self) -> None:
        """推流主循环：无论哪种刷新模式，画面始终按 0.5s 推流；
        完整决策由 _step_loop 按 interval 节流，二者解耦互不阻塞。"""
        while not self._stop_evt.is_set():
            try:
                self._stream_tick()
                self._stop_evt.wait(0.5)
            except Exception as e:  # noqa: BLE001 - 单轮失败不中断循环
                self._broadcast({"kind": "agent_error", "message": str(e), "ts": _ts()})
                self._stop_evt.wait(0.5)

    def _step_loop(self) -> None:
        """决策线程：按 interval 节流执行完整决策（采集缓存→决策→执行→落库→广播）。"""
        while not self._stop_evt.is_set():
            if self._stop_evt.wait(self.interval):
                break
            try:
                self.step()
            except Exception as e:  # noqa: BLE001 - 单轮失败不中断决策循环
                self._broadcast({"kind": "agent_error", "message": str(e), "ts": _ts()})

    def _set_latest_frame(self, det_result: dict, image_b64: str) -> None:
        """写入最近一帧画面缓存（供 /api/frame 与主站实时画面读取，使用主源帧）。"""
        with self._lock:
            self.latest_frame = {
                "ts": _ts(),
                "source": det_result.get("source", "demo_frame"),
                "backend": det_result.get("backend", "sim"),
                "image": image_b64,
                "detections": det_result.get("detections", []),
                "summary": det_result.get("summary", {}),
            }

    # ---------- 多源采集 ----------
    def _collect_cams(self) -> list[dict]:
        """遍历所有摄像头源：采集→推理→标注，返回每路的展示数据（不含原始帧）。

        画面以 320px 缩略图 + JPEG q60 编码：持续录像高频推流时单路 payload
        从 20-50KB 降到 5-10KB，浏览器每 0.5s 解码多路大图的 CPU 压力大幅下降。
        raw_frame 字段仅供内部（决策落盘每日帧），广播前统一剔除。
        """
        cams = []
        for item in camera_manager.read_all():
            frame = item.get("frame")
            det = detector.detect(frame, source_hint=item.get("last_source", "demo_frame"))
            image_b64 = ""
            if frame is not None:
                annotated = annotate.draw_detections(frame, det)
                image_b64 = annotate.frame_to_base64_thumb(annotated, max_w=320, quality=60)
            cams.append({
                "cam_id": item.get("cam_id"),
                "name": item.get("name", item.get("cam_id")),
                "source": item.get("source", ""),
                "kind": item.get("kind", "usb"),
                "last_source": item.get("last_source", "demo_frame"),
                "available": item.get("last_source") in ("usb_cam", "net_cam"),
                "ts": _ts(),
                "image": image_b64,
                "detections": det.get("detections", []),
                "summary": det.get("summary", {}),
                "det_source": det.get("source", "demo_frame"),
                "raw_frame": frame,
            })
        return cams

    @staticmethod
    def _strip_raw(cams: list[dict]) -> list[dict]:
        """剔除内部字段（raw_frame / frame），返回可广播的安全副本。"""
        return [{k: v for k, v in c.items() if k not in ("raw_frame", "frame")} for c in cams]

    @staticmethod
    def _merge_dets(cams: list[dict]) -> dict:
        """合并所有源的检测结果供决策使用：病害去重、总数/置信度汇总。"""
        dets = [c for c in cams if c.get("detections")]
        if not dets:
            return {"source": "demo_frame", "backend": "sim",
                    "detections": [], "summary": {"total": 0, "diseases": [], "max_conf": 0.0, "healthy": True},
                    "frame_shape": [480, 640]}
        diseases = sorted({d for c in dets for d in c.get("summary", {}).get("diseases", [])})
        total = sum(c.get("summary", {}).get("total", 0) for c in dets)
        max_conf = max([c.get("summary", {}).get("max_conf", 0.0) or 0.0 for c in dets] or [0.0])
        return {
            "source": dets[0].get("det_source", "demo_frame"),
            "backend": "ultralytics" if any(c.get("det_source") == "usb_cam" or c.get("last_source") in ("usb_cam", "net_cam") for c in dets) else "sim",
            "detections": [d for c in dets for d in c.get("detections", [])],
            "summary": {
                "total": total,
                "diseases": diseases,
                "max_conf": round(max_conf, 2),
                "healthy": not diseases,
            },
            "frame_shape": [480, 640],
        }

    def _stream_tick(self) -> None:
        """推流主循环单次 tick：采集所有源→推理→标注→更新视觉缓存→推送多路画面帧。

        完整快照（决策/执行器/传感器）由 _step_loop 按间隔生成（kind=state_update）。
        """
        cams = self._collect_cams()
        main = cams[0] if cams else None
        if main is not None:
            self._set_latest_frame(
                {"source": main["last_source"], "backend": main["det_source"],
                 "detections": main["detections"], "summary": main["summary"]},
                main["image"],
            )
        # 更新视觉缓存：决策线程复用本轮的采集+推理结果，不再重复读摄像头/推理
        with self._lock:
            self._cams_cache = {
                "cams": cams,
                "det_result": self._merge_dets(cams),
                "main_frame": cams[0]["raw_frame"] if cams else None,
                "main_image": cams[0]["image"] if cams else "",
                "ts": time.monotonic(),
            }
        self._broadcast({
            "kind": "stream_frame",
            "data": {
                "ts": _ts(),
                "cams": self._strip_raw(cams),
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

    def _current_cams(self) -> tuple[list, dict, object, str]:
        """取最新视觉缓存（采集+推理结果）；缓存缺失/过期时自采一轮。

        决策与推流解耦：推流线程持续产帧，决策线程几乎总是能直接复用
        最新缓存，不重复读摄像头/跑推理，也不因 LLM/天气慢而冻结推流。
        """
        with self._lock:
            cc = self._cams_cache
        if cc is not None and (time.monotonic() - cc["ts"]) < 3.0:
            return cc["cams"], cc["det_result"], cc.get("main_frame"), cc.get("main_image", "")
        # 缓存为空或过期（如刚启动、决策间隔很大）：自采一轮
        cams = self._collect_cams()
        det_result = self._merge_dets(cams)
        with self._lock:
            self._cams_cache = {
                "cams": cams,
                "det_result": det_result,
                "main_frame": cams[0]["raw_frame"] if cams else None,
                "main_image": cams[0]["image"] if cams else "",
                "ts": time.monotonic(),
            }
        return cams, det_result, (cams[0]["raw_frame"] if cams else None), (cams[0]["image"] if cams else "")

    def _step_once(self) -> dict:
        """串行执行一次完整决策（由 step() 持锁调用）。

        视觉部分复用推流线程的最新缓存（采集+推理已完成），
        LLM/天气/写库等慢操作只发生在本决策线程，不影响画面推流。
        """
        garden = self._garden()

        # 1) 传感器
        soil = sensor.read()
        try:
            db.log_sensor(self.garden_key, soil["type"], soil["value"], soil["unit"], soil.get("meta"))
        except Exception:  # noqa: BLE001 - 数据库不可用时降级跳过
            pass

        # 2) 视觉（复用推流缓存的多源采集+推理结果）
        cams, det_result, main_frame, image_b64 = self._current_cams()
        frame_shape = det_result.get("frame_shape") or ([480, 640] if cams else [480, 640])

        # 2.1) 土壤湿度历史（监控台趋势图数据，最近 60 条）
        soil_history = []
        try:
            soil_history = db.recent_sensor(self.garden_key, "soil_moisture", limit=60)
        except Exception:  # noqa: BLE001
            pass

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

        # 7) 标注帧（多路画面已随 cams 推送，此处缓存主源缩略帧供 /api/frame 使用）
        self._set_latest_frame(
            {"source": cams[0]["last_source"] if cams else "demo_frame",
             "backend": det_result.get("backend", "sim"),
             "detections": det_result.get("detections", []),
             "summary": det_result.get("summary", {})},
            image_b64,
        )

        # 7.1) 每日视频监测汇总（供风险预警「摄像头监测预警」每日推送：
        #      当日最新标注帧 + 病害检出计数 + 处理动作计数）
        try:
            if main_frame is not None:
                daily_recorder.save_frame(annotate.frame_to_jpeg_bytes(main_frame))
            daily_recorder.record(det_result.get("summary", {}), decision)
        except Exception:  # noqa: BLE001 - 汇总失败不影响主链路
            pass

        # 8) 状态快照 + 广播
        snapshot = {
            "ts": _ts(),
            "garden": {"key": self.garden_key, "name": garden.get("name", self.garden_key)},
            "camera": {"available": bool(cams) and cams[0]["available"],
                       "backend": det_result.get("backend", "sim"),
                       "source": det_result.get("source", "demo_frame")},
            "cams": self._strip_raw(cams),
            "soil": soil,
            "soil_history": soil_history,
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
