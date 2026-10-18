"""每日视频监测汇总：将 Agent 每轮的病害检出与处理动作滚动汇总到当日文件。

用途：风险预警「摄像头监测预警」每日推送一次（拍摄画面 + 病虫害情况 + 处理情况）。
- 帧文件：data/alerts/frame_YYYY-MM-DD.jpg（当日最新一轮标注帧，整轮决策后覆盖）
- 汇总文件：data/alerts/daily_YYYY-MM-DD.json（病害检出轮次计数 / 处理动作计数 / 最大置信度）

跨服务重启保留当天数据，每日生成器到点即可产出完整日报。
"""
from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path

ALERT_DIR = Path(__file__).resolve().parents[2] / "data" / "alerts"


def today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _daily_path(date_str: str) -> Path:
    return ALERT_DIR / f"daily_{date_str}.json"


def _frame_path(date_str: str) -> Path:
    return ALERT_DIR / f"frame_{date_str}.jpg"


class DailyRecorder:
    """当日滚动记录器（线程安全）：病害检出计数 + 处理动作计数 + 最新标注帧。"""

    def __init__(self):
        self._lock = threading.Lock()

    # ---------- 帧 ----------
    def save_frame(self, image_bytes: bytes | None, date_str: str | None = None) -> Path | None:
        """保存当日最新标注帧（JPEG bytes，覆盖当日文件）。失败返回 None。"""
        if not image_bytes:
            return None
        try:
            ALERT_DIR.mkdir(parents=True, exist_ok=True)
            p = _frame_path(date_str or today_str())
            p.write_bytes(image_bytes)
            return p
        except Exception:  # noqa: BLE001 - 帧落盘失败不影响主链路
            return None

    def frame_path(self, date_str: str | None = None) -> Path | None:
        p = _frame_path(date_str or today_str())
        return p if p.exists() else None

    # ---------- 汇总 ----------
    def record(self, det_summary: dict, decision: dict) -> None:
        """每轮整轮决策后调用：累加病害检出与处理动作（线程安全）。"""
        date_str = today_str()
        with self._lock:
            data = self._load(date_str)
            data["updated_at"] = datetime.now().isoformat(timespec="seconds")
            data["rounds"] = data.get("rounds", 0) + 1
            # 病害检出计数（以合并后的 summary.diseases 为准）
            diseases = (det_summary or {}).get("diseases") or []
            for name in diseases:
                data["disease_hits"][name] = data["disease_hits"].get(name, 0) + 1
            mc = (det_summary or {}).get("max_conf") or 0.0
            if float(mc) > float(data.get("max_conf", 0.0)):
                data["max_conf"] = round(float(mc), 2)
            # 处理动作计数（device:op）
            actions = (decision or {}).get("actions") or []
            for a in actions:
                key = f"{a.get('device')}:{a.get('op')}"
                data["actions"][key] = data["actions"].get(key, 0) + 1
            self._save(date_str, data)

    def load_today(self) -> dict:
        """读取当日汇总（无文件时返回空模板，不抛异常）。"""
        return self._load(today_str())

    def load(self, date_str: str) -> dict:
        return self._load(date_str)

    # ---------- 内部 ----------
    def _load(self, date_str: str) -> dict:
        p = _daily_path(date_str)
        data: dict = {}
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001 - 损坏文件按空模板处理
                data = {}
        data.setdefault("date", date_str)
        data.setdefault("disease_hits", {})
        data.setdefault("max_conf", 0.0)
        data.setdefault("actions", {})
        data.setdefault("rounds", 0)
        return data

    def _save(self, date_str: str, data: dict) -> None:
        try:
            ALERT_DIR.mkdir(parents=True, exist_ok=True)
            _daily_path(date_str).write_text(
                json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8"
            )
        except Exception:  # noqa: BLE001 - 汇总落盘失败不影响主链路
            pass


# 模块级单例（runner / 每日预警生成器共享）
daily_recorder = DailyRecorder()
