"""智能执行页：Agent 控制台（启动/停止/单步决策 + 状态展示 + 动态网页入口）。

演示链路入口：本页控制 Agent 运行；动态动画网页由 WS 服务提供
（http://127.0.0.1:8510/），页面提供一键启动/打开入口。
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from datetime import datetime

import pandas as pd
import requests
import streamlit as st

from src import config, db, ui

WS_BASE = f"http://127.0.0.1:{os.getenv('AGENT_WS_PORT', '8510')}"

# 实时画面缓存：后台 WS 订阅线程写入（方案 A），页面 fragment 读取渲染
_FRAME_CACHE: dict = {}
_FRAME_LOCK = threading.Lock()
_WS_LISTENER: threading.Thread | None = None

LEVEL_CN = {"low": "低", "medium": "中", "high": "高"}
DECISION_CN = {
    "irrigate": "💧 自动浇水", "irrigating": "💧 灌溉中", "spray": "🚿 自动打药", "cool": "❄️ 降温喷灌",
    "stop": "⏹ 停止作业", "noop": "✔ 无操作",
}


def _fmt_time(v) -> str:
    """把 created_at 统一格式化为 'MM-DD HH:MM'。

    兼容两种来源：REST 接口返回的字符串（如 "2026-10-16T02:06:35"），
    以及直连 MySQL 时 pymysql 自动转换出的 datetime 对象。
    """
    if isinstance(v, datetime):
        return v.strftime("%m-%d %H:%M")
    s = str(v or "")
    return s[5:16] if len(s) >= 16 else s


def ws_alive() -> bool:
    try:
        r = requests.get(f"{WS_BASE}/api/health", timeout=3)
        return r.status_code == 200
    except Exception:  # noqa: BLE001
        return False


def _api(path: str, method: str = "GET", timeout: int = 30, data: dict | None = None):
    try:
        if method == "POST":
            if data is not None:
                return requests.post(f"{WS_BASE}{path}", json=data, timeout=timeout).json()
            return requests.post(f"{WS_BASE}{path}", timeout=timeout).json()
        return requests.get(f"{WS_BASE}{path}", timeout=timeout).json()
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "message": str(e)}


def _ensure_ws_listener() -> bool:
    """确保后台 WS 订阅线程运行（幂等）。

    方案 A：页面内直接订阅 /ws，收到 stream_frame / state_update 帧后写入
    _FRAME_CACHE；断线自动 2 秒重连。供「实时画面」fragment 渲染。
    """
    global _WS_LISTENER
    if _WS_LISTENER is not None and _WS_LISTENER.is_alive():
        return True
    if not ws_alive():
        return False

    def _run() -> None:
        import asyncio
        import json as _json

        import websockets
        uri = f"ws://127.0.0.1:{os.getenv('AGENT_WS_PORT', '8510')}/ws"

        async def _listen() -> None:
            async with websockets.connect(uri) as ws:
                while True:
                    try:
                        msg = _json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
                    except asyncio.TimeoutError:
                        continue
                    data = msg.get("data") or {}
                    det = data.get("detections") or {}
                    img = det.get("image")
                    if img and msg.get("kind") in ("stream_frame", "state_update"):
                        with _FRAME_LOCK:
                            _FRAME_CACHE.update({
                                "ts": data.get("ts"),
                                "image": img,
                                "source": det.get("source", "demo_frame"),
                                "backend": det.get("backend", "sim"),
                            })

        while True:
            try:
                asyncio.run(_listen())
            except Exception:  # noqa: BLE001 - 断线重连
                time.sleep(2)

    _WS_LISTENER = threading.Thread(target=_run, daemon=True, name="page-ws-frame-listener")
    _WS_LISTENER.start()
    return True


def _render_live_frame() -> None:
    """实时画面片段（fragment，约 1 秒自动刷新）。

    优先读 WebSocket 订阅缓存帧（方案 A）；缓存无帧时拉取 /api/frame 兜底（方案 B）。
    """
    with _FRAME_LOCK:
        cache = dict(_FRAME_CACHE)
    img, src, ts = None, "demo_frame", ""
    if cache.get("image"):
        img, src, ts = cache["image"], cache.get("source", "demo_frame"), cache.get("ts", "")
    else:
        f = _api("/api/frame", timeout=3)
        if f.get("image"):
            img, src, ts = f["image"], f.get("source", "demo_frame"), f.get("ts", "")
    if img:
        st.image(img, use_container_width=True)
        src_txt = "USB摄像头实拍" if src == "usb_cam" else "演示帧（摄像头不可用）"
        suffix = f" ｜ 帧时间 {str(ts)[11:19]}" if ts else ""
        st.caption(f"实时画面 · 来源：{src_txt}{suffix}")
    else:
        st.info("实时画面等待中…（Agent 未运行或无帧，可先启动 Agent）")


def _start_ws_service() -> bool:
    """用当前 venv 的 python 在后台启动 WS 服务。"""
    py = sys.executable
    try:
        subprocess.Popen(
            [py, "-m", "src.ws_server"],
            cwd=str(config.BASE_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        return True
    except Exception as e:  # noqa: BLE001
        st.error(f"启动 WS 服务失败：{e}")
        return False


def main() -> None:
    st.set_page_config(page_title="智能执行", page_icon="🤖", layout="wide")
    ui.inject_css()

    st.title("🤖 智能执行（Agent 控制台）")
    st.caption(
        "演示链路：USB摄像头/演示帧 → OpenCV → YOLO 推理 → Agent 决策 "
        "→ WebSocket 推送 → 网页动态展示（喷杆喷头 / 滴灌阀门）。"
    )

    garden = ui.select_garden()
    alive = ws_alive()

    # ---------- 顶部控制区 ----------
    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        if st.button("🚀 启动 WS 服务（动态网页）", type="primary", width="stretch",
                     disabled=alive):
            pass
        if not alive and st.button("🛠 一键启动 WS 服务", width="stretch"):
            _start_ws_service()
            st.rerun()
    with c2:
        if alive and st.button("▶️ 启动 Agent 主循环", type="primary", width="stretch"):
            r = _api("/api/agent/start", "POST")
            st.toast("Agent 已启动" if r.get("ok") else f"启动失败：{r.get('message')}")
    with c3:
        if alive and st.button("⏹ 停止 Agent", width="stretch"):
            r = _api("/api/agent/stop", "POST")
            st.toast("Agent 已停止" if r.get("ok") else f"停止失败：{r.get('message')}")
        if alive and st.button("⚡ 立即执行一步", width="stretch"):
            _api("/api/agent/step", "POST")
            st.rerun()

    if not alive:
        st.warning("动态展示服务（WS）未运行。点击上方「一键启动 WS 服务」，再打开下方动态网页即可看到完整链路。")
        st.divider()
        _render_history(garden)
        return

    status = _api("/api/status")
    snap = status.get("latest") or {}

    # ---------- 刷新设置区 ----------
    cur_mode = status.get("refresh_mode", "interval")
    cur_interval = float(status.get("interval", 5.0) or 5.0)
    st.divider()
    st.markdown("### ⏱ 刷新设置")
    st.caption(
        "「持续录像」：画面连续推送（约 0.2 秒/帧，适合观察实时画面）；「自定义刷新时间」："
        "整轮（采集→推理→决策→推送）按设定间隔执行。间隔越小 CPU 占用越高，最小 0.5 秒。"
    )
    r1, r2, r3 = st.columns([2, 2, 1])
    with r1:
        mode = st.radio("刷新模式", ["自定义刷新时间", "持续录像"], horizontal=True,
                        index=0 if cur_mode != "stream" else 1)
    with r2:
        interval = st.number_input("自定义刷新间隔（秒）", min_value=0.5, max_value=60.0,
                                   value=float(cur_interval), step=0.5)
    with r3:
        st.markdown("")
        st.markdown("")
        if st.button("✅ 应用刷新设置", type="primary"):
            cfg = _api("/api/agent/config", "POST",
                       data={"refresh_mode": "stream" if mode == "持续录像" else "interval",
                             "interval": float(interval)})
            st.toast("刷新设置已应用" if cfg.get("ok") else f"应用失败：{cfg.get('message')}")
            if cfg.get("ok"):
                st.rerun()

    t1, t2 = st.columns([1, 3])
    with t1:
        live_on = st.toggle("📹 实时画面", value=st.session_state.get("live_on", True),
                            key="live_on")
    with t2:
        st.caption("开启后本页画面实时刷新（约 1 秒/帧）：优先走 WebSocket 推送帧（A），"
                   "不可用时自动拉取 /api/frame 兜底（B）。")
    if live_on:
        _ensure_ws_listener()

    col_left, col_right = st.columns([3, 2])

    # ---------- 左：实时画面 + 检测 ----------
    with col_left:
        st.subheader("📷 视觉检测画面")
        if st.session_state.get("live_on", True):
            # 实时画面：fragment 约 1 秒自动刷新（方案 A 缓存优先，方案 B 接口兜底）
            @st.fragment(run_every=1.0)
            def _live():
                _render_live_frame()

            _live()
        else:
            img = (snap.get("detections") or {}).get("image", "")
            if img:
                src = (snap.get("detections") or {}).get("source", "demo_frame")
                backend = (snap.get("detections") or {}).get("backend", "?")
                src_cn = "USB 摄像头实拍" if src == "usb_cam" else "演示帧（摄像头不可用）"
                st.image(img, caption=f"检测帧（{backend} · 来源：{src_cn}）", width="stretch")
            else:
                st.info("暂无画面：请点击「立即执行一步」生成检测帧。")

        dets = (snap.get("detections") or {}).get("detections", [])
        if dets:
            st.markdown("**检测结果**")
            cols = st.columns(min(4, len(dets)))
            for i, d in enumerate(dets[:4]):
                color = "#2e9e5b" if d["class"] == "Healthy" else "#d64545"
                cols[i].markdown(
                    f'<div class="risk-badge" style="background:{color}">{d["class"]} {d["conf"]:.0%}</div>',
                    unsafe_allow_html=True,
                )

        hist = _api("/api/history")
        if hist.get("items"):
            st.divider()
            st.subheader("📜 Agent 决策历史")
            rows = [{
                "时间": _fmt_time(i.get("created_at")),
                "决策": DECISION_CN.get(i.get("decision_type"), i.get("decision_type")),
                "等级": LEVEL_CN.get(i.get("overall_level", "low")),
                "摘要": (i.get("summary") or "")[:80],
            } for i in hist["items"][:10]]
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    # ---------- 右：传感器 + 执行器 ----------
    with col_right:
        st.subheader("🌱 传感器与执行器")
        soil = snap.get("soil") or {}
        if soil:
            v = float(soil.get("value", 0))
            st.metric("土壤湿度", f"{v:.1f}%", "缺水<45 · 适宜60-80")
        act = snap.get("actuator") or {}
        valve = act.get("drip_valve") or {}
        boom = act.get("spray_boom") or {}
        m1, m2 = st.columns(2)
        m1.metric("滴灌阀门", "开" if valve.get("state") == "open" else "关",
                  f"{valve.get('flow_rate', 0):.0f} L/h")
        m2.metric("喷杆喷头", "喷雾中" if boom.get("state") == "on" else "停",
                  f"{boom.get('nozzle_pressure', 0):.1f} bar")

        dec = snap.get("decision") or {}
        if dec:
            st.divider()
            st.subheader("🤖 最近决策")
            st.markdown(f"**{DECISION_CN.get(dec.get('decision_type'), dec.get('decision_type'))}**"
                        f" ｜ 等级：{LEVEL_CN.get(dec.get('overall_level', 'low'))}")
            st.write(dec.get("summary", ""))
            exp = dec.get("explanation") or ""
            if exp:
                st.markdown(f"🧠 **大模型解释**：{exp}")
            for a in dec.get("actions", []):
                dev = "滴灌阀门" if a.get("device") == "drip_valve" else "喷杆喷头"
                op = "开启" if a.get("op") in ("open", "on") else "关闭"
                st.markdown(f"- **{dev} → {op}**：{a.get('reason', '')}")

        st.divider()
        st.markdown("### 🖥 动态动画网页")
        st.markdown(f"打开 [茶园智能执行动态展示]({WS_BASE}/) （WebSocket 实时推送执行器动画）")
        if st.button("🔗 在浏览器打开动态网页", width="stretch"):
            import webbrowser
            webbrowser.open(f"{WS_BASE}/")

        # Agent 状态
        st.divider()
        _mode_label = "持续录像" if status.get("refresh_mode") == "stream" \
            else f"{float(status.get('interval', 5.0) or 5.0):.1f}s 间隔"
        st.caption(f"Agent 状态：{'🟢 运行中' if status.get('running') else '⚪ 已停止'} ｜ "
                   f"已执行 {status.get('steps', 0)} 轮 ｜ 刷新：{_mode_label}")


def _render_history(garden: dict) -> None:
    """WS 服务不可用时，直接读库展示历史决策。"""
    rows = db.agent_decision_history(garden["key"], limit=10)
    if rows:
        st.subheader("📜 Agent 决策历史（数据库）")
        df = pd.DataFrame([{
            "时间": _fmt_time(r.get("created_at")),
            "决策": DECISION_CN.get(r.get("decision_type"), r.get("decision_type")),
            "等级": LEVEL_CN.get(r.get("overall_level", "low")),
            "摘要": (r.get("summary") or "")[:80],
        } for r in rows])
        st.dataframe(df, width="stretch", hide_index=True)
    else:
        st.info("暂无决策记录。启动 WS 服务后点击「立即执行一步」生成。")


if __name__ == "__main__":
    main()
