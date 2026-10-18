"""智能执行页：Agent 控制台（启动/停止/单步决策 + 状态展示 + 动态网页入口）。

演示链路入口：本页控制 Agent 运行；动态动画网页由 WS 服务提供
（http://127.0.0.1:8510/），页面提供一键启动/打开入口。
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime

import pandas as pd
import requests
import streamlit as st

from src import config, db, ui

WS_BASE = f"http://127.0.0.1:{os.getenv('AGENT_WS_PORT', '8510')}"

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


def _api(path: str, method: str = "GET", timeout: int = 5, data: dict | None = None):
    """请求 WS 服务。默认 5 秒超时：8510 异常时页面快速降级提示，不干等。

    整轮决策（/api/agent/step）可能包含首帧预热等耗时操作，调用方单独传较长超时。
    """
    try:
        if method == "POST":
            if data is not None:
                return requests.post(f"{WS_BASE}{path}", json=data, timeout=timeout).json()
            return requests.post(f"{WS_BASE}{path}", timeout=timeout).json()
        return requests.get(f"{WS_BASE}{path}", timeout=timeout).json()
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "message": str(e)}


def _ensure_ws_listener() -> bool:
    """（已弃用）主站实时画面改为每秒拉取 /api/frame，不再常驻 WS 订阅线程。

    保留旧常驻线程会在 Streamlit 会话反复刷新时累积僵尸 WS 连接，
    最终拖垮 WS 服务（clients 无限增长 → 广播变慢 → 服务崩溃）。
    实时画面统一走方案 B：/api/frame 接口（服务端缓存最近一帧）。
    """
    return ws_alive()


def _render_live_frame() -> None:
    """实时画面片段（fragment，约 1 秒自动刷新）。

    直接拉取服务端最近一帧缓存（/api/frame，由 Agent 持续录像/整轮决策更新），
    本地每秒一次 50KB 级请求开销可忽略，且无连接泄漏风险。
    """
    f = _api("/api/frame", timeout=3)
    img, src, ts = f.get("image"), f.get("source", "demo_frame"), f.get("ts", "")
    if img:
        st.image(img, width="stretch")
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
        # Popen 成功只代表进程已创建；等待健康检查通过，避免页面立即重绘时误判未启动。
        for _ in range(20):
            if ws_alive():
                return True
            time.sleep(0.25)
        st.error("WS 服务已启动但健康检查超时，请查看日志或稍后重试")
        return False
    except Exception as e:  # noqa: BLE001
        st.error(f"启动 WS 服务失败：{e}")
        return False


def main() -> None:
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
            if _start_ws_service():
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
            _api("/api/agent/step", "POST", timeout=30)
            st.rerun()

    if not alive:
        st.warning("动态展示服务（WS）未运行。点击上方「一键启动 WS 服务」，再打开下方动态网页即可看到完整链路。")
        st.divider()
        _render_history(garden)
        return

    status = _api("/api/status")
    snap = status.get("latest") or {}

    # ---------- 摄像头源管理（监控台） ----------
    st.divider()
    st.markdown("### 📹 摄像头源管理")
    st.caption("默认已注册 USB 摄像头 (usb:0)。可添加网络摄像头（RTSP/HTTP），多路画面在动态监控台网页实时显示。")
    cams = _api("/api/cameras").get("cameras", [])
    if cams:
        rows = [{
            "ID": c.get("id"),
            "名称": c.get("name", ""),
            "来源": c.get("source", ""),
            "类型": {"usb": "USB", "rtsp": "RTSP", "http": "HTTP"}.get(c.get("kind"), c.get("kind", "")),
            "状态": "🟢 可用" if c.get("available") else "⚪ 待用",
            "最近帧": {"usb_cam": "USB实拍", "net_cam": "网络实拍",
                       "demo_frame": "演示帧"}.get(c.get("last_source"), "-"),
        } for c in cams]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    else:
        st.info("暂无摄像头源")
    a1, a2 = st.columns([2, 1])
    with a1:
        src_input = st.text_input("摄像头源", placeholder="usb:1 / rtsp://user:pwd@ip:554/... / http://ip:8080/video",
                                  key="cam_src")
    with a2:
        st.markdown("")
        st.markdown("")
        if st.button("➕ 添加摄像头源", type="primary"):
            if src_input.strip():
                r = _api("/api/cameras/add", "POST", timeout=10, data={"source": src_input.strip()})
                st.toast("已添加" if r.get("ok") else f"添加失败：{r.get('message')}")
                if r.get("ok"):
                    st.rerun()
            else:
                st.toast("请填写摄像头源")
    if cams:
        b1, b2 = st.columns([2, 1])
        with b1:
            del_id = st.selectbox("选择要删除的摄像头", [c.get("id") for c in cams], key="cam_del")
        with b2:
            st.markdown("")
            st.markdown("")
            if st.button("🗑 删除摄像头源"):
                r = _api("/api/cameras/remove", "POST", timeout=10, data={"cam_id": del_id})
                st.toast("已删除" if r.get("ok") else f"删除失败：{r.get('message')}")
                if r.get("ok"):
                    st.rerun()

    # ---------- 刷新设置区 ----------
    cur_mode = status.get("refresh_mode", "interval")
    cur_interval = float(status.get("interval", 5.0) or 5.0)
    st.divider()
    st.markdown("### ⏱ 刷新设置")
    st.caption(
        "「持续录像」：画面约每 0.5 秒推送一帧（适合观察实时画面）；「自定义刷新时间」："
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
        st.caption("开启后本页画面实时刷新（约 1 秒/帧）：拉取服务端最近一帧缓存，"
                   "不占用 WebSocket 长连接（稳定、无泄漏）。")
    if live_on:
        pass  # 实时画面由下方 fragment 每秒拉取 /api/frame 实现

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
                src_cn = {"usb_cam": "USB 摄像头实拍", "net_cam": "网络摄像头实拍"}.get(
                    src, "演示帧（摄像头不可用）")
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
