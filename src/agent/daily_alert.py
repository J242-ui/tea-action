"""摄像头监测预警 · 每日推送生成器。

每日一次（默认 09:00，可用环境变量 ALERT_DAILY_TIME 配置）生成「摄像头监测预警」：
  拍摄画面（当日最新标注帧）+ 病虫害推送内容 + Agent 处理情况。

链路：DailyRecorder 当日汇总 → 到点生成预警 → 落库 camera_alert_log
      → WebSocket 广播 kind=alert_push（监控台 / 风险预警页展示）。

线程：ws_server 启动时挂起 daemon 线程，每 30 秒检查一次
      「已到配置时间 && 当日尚未生成」则生成；同日只生成一次。
"""
from __future__ import annotations

import base64
import os
import threading
from datetime import datetime

from .. import db
from .daily_recorder import daily_recorder
from .runner import runner  # 复用 Agent 当前归属账号（user_id），预警按账号落库

DEFAULT_ALERT_TIME = "09:00"

# YOLO 检测类别 → 中文名（用于推送文案）
# 注意：tea-leaf 数据集的实际类别名带 " of tea" 后缀（如 "White spot of tea"），
# 两类名都映射，避免推送文案出现英文原名。
DISEASE_CN = {
    "Black rot": "黑腐病",
    "Black rot of tea": "黑腐病",
    "Brown blight": "褐枯病",
    "Brown blight of tea": "褐枯病",
    "Leaf rust": "叶锈病",
    "Leaf rust of tea": "叶锈病",
    "Red Spider": "红蜘蛛（叶螨）",
    "Red Spider of tea": "红蜘蛛（叶螨）",
    "Tea Mosquito bug infested leaf": "茶角盲蝽危害叶",
    "Healthy": "健康叶",
    "White spot": "白星病",
    "White spot of tea": "白星病",
    "disease": "茶树病害",
}


def _hm() -> str:
    return datetime.now().strftime("%H:%M")


def _disease_display(name: str) -> str:
    return DISEASE_CN.get(name, name)


def _actions_text(actions: dict) -> str:
    """把处理动作计数转成自然语言（喷药/灌溉/观察）。"""
    parts = []
    spray = int(actions.get("spray_boom:on", 0) or 0)
    irrig = int(actions.get("drip_valve:open", 0) or 0)
    if spray:
        parts.append(f"喷杆喷头开启打药 {spray} 次")
    if irrig:
        parts.append(f"滴灌阀门开启灌溉 {irrig} 次")
    if not parts:
        parts.append("系统持续观察，未执行打药/灌溉作业")
    return "；".join(parts)


def build_alert(garden_key: str) -> dict | None:
    """读取当日汇总生成预警记录；当天无任何监测轮次时返回 None。"""
    data = daily_recorder.load_today()
    date_str = data.get("date") or datetime.now().strftime("%Y-%m-%d")
    rounds = int(data.get("rounds", 0) or 0)
    if rounds == 0:
        return None
    hits: dict = data.get("disease_hits") or {}
    actions: dict = data.get("actions") or {}
    max_conf = float(data.get("max_conf", 0.0) or 0.0)

    # 拍摄画面（当日最新标注帧 → base64）
    image_b64 = ""
    fp = daily_recorder.frame_path(date_str)
    if fp is not None:
        try:
            image_b64 = base64.b64encode(fp.read_bytes()).decode("ascii")
        except Exception:  # noqa: BLE001
            image_b64 = ""

    # 病虫害推送内容
    if hits:
        top = sorted(hits.items(), key=lambda kv: -kv[1])[0]
        hit_desc = "、".join(f"{_disease_display(n)}({c}轮)" for n, c in
                             sorted(hits.items(), key=lambda kv: -kv[1]))
        if max_conf >= 0.5 or len(hits) >= 3:
            level = "high"
            title = f"🚨 茶园视频监测预警（{date_str}）：检出 {len(hits)} 类病害"
        elif len(hits) >= 1:
            level = "medium"
            title = f"⚠️ 茶园视频监测预警（{date_str}）：检出 {len(hits)} 类病害"
        content = (
            f"今日共监测 {rounds} 轮。视频画面检出病虫害 {len(hits)} 类：{hit_desc}；"
            f"最高检测置信度 {max_conf:.0%}，主要风险为「{_disease_display(top[0])}」。"
        )
    else:
        level = "low"
        title = f"✅ 茶园视频监测日报（{date_str}）：未检出病害"
        content = f"今日共监测 {rounds} 轮，视频画面未检出病虫害，茶园整体健康。"

    alert = {
        "kind": "alert_push",
        "data": {
            "garden_key": garden_key,
            "user_id": runner.user_id,
            "alert_date": date_str,
            "title": title,
            "content": content,
            "level": level,
            "actions_text": _actions_text(actions),
            "image_b64": image_b64,
            "ts": datetime.now().isoformat(timespec="seconds"),
        },
    }
    return alert


class DailyAlertScheduler:
    """每日预警调度器（daemon 线程，同日只生成一次）。"""

    def __init__(self, garden_key: str):
        self.garden_key = garden_key
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_date: str | None = None
        self._latest: dict | None = None
        self._listeners: list = []

    # ---------- 订阅 / 查询 ----------
    def subscribe(self, fn) -> None:
        self._listeners.append(fn)

    def latest(self) -> dict | None:
        """内存中的最近一条预警（服务重启后为 None，页面可回退查 DB）。"""
        return self._latest

    # ---------- 生命周期 ----------
    def start(self) -> bool:
        if self._thread is not None and self._thread.is_alive():
            return False
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="daily-alert-scheduler")
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()

    # ---------- 主循环 ----------
    def _run(self) -> None:
        while not self._stop.wait(30):
            try:
                self._maybe_generate()
            except Exception:  # noqa: BLE001 - 单次失败不中断调度
                pass

    def _maybe_generate(self) -> None:
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        if self._last_date == date_str:
            return  # 同日已生成
        alert_time = (os.getenv("ALERT_DAILY_TIME", "") or DEFAULT_ALERT_TIME).strip()
        if now.strftime("%H:%M") < alert_time:
            return  # 未到推送时间
        alert = build_alert(self.garden_key)
        if alert is None:
            return  # 当天还没有监测数据，等下一轮检查
        self._last_date = date_str
        self._latest = alert
        d = alert["data"]
        # 落库（数据库不可用时降级跳过，广播仍发送）；按 Agent 当前归属账号落库
        try:
            db.log_camera_alert(
                self.garden_key, d["alert_date"], d["title"], d["content"],
                d["level"], d["actions_text"], d.get("image_b64", ""),
                user_id=runner.user_id,
            )
        except Exception:  # noqa: BLE001
            pass
        # 广播
        for fn in list(self._listeners):
            try:
                fn(alert)
            except Exception:  # noqa: BLE001
                pass


# 模块级单例（ws_server 启动时接线并 start）
def create_scheduler(garden_key: str) -> DailyAlertScheduler:
    return DailyAlertScheduler(garden_key)
