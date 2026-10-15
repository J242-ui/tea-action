"""智能执行页：Agent 控制台（启动/停止/单步决策 + 状态展示 + 动态网页入口）。

演示链路入口：本页控制 Agent 运行；动态动画网页由 WS 服务提供
（http://127.0.0.1:8510/），页面提供一键启动/打开入口。
"""
from __future__ import annotations

import os
import subprocess
import sys

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


def ws_alive() -> bool:
    try:
        r = requests.get(f"{WS_BASE}/api/health", timeout=3)
        return r.status_code == 200
    except Exception:  # noqa: BLE001
        return False


def _api(path: str, method: str = "GET", timeout: int = 30):
    try:
        if method == "POST":
            return requests.post(f"{WS_BASE}{path}", timeout=timeout).json()
        return requests.get(f"{WS_BASE}{path}", timeout=timeout).json()
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "message": str(e)}


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

    col_left, col_right = st.columns([3, 2])

    # ---------- 左：实时画面 + 检测 ----------
    with col_left:
        st.subheader("📷 视觉检测画面")
        img = (snap.get("detections") or {}).get("image", "")
        if img:
            st.image(img, caption=f"检测帧（{snap.get('detections', {}).get('backend', '?')}）",
                     width="stretch")
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
                "时间": (i.get("created_at") or "")[5:16],
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
        st.caption(f"Agent 状态：{'🟢 运行中' if status.get('running') else '⚪ 已停止'} ｜ "
                   f"已执行 {status.get('steps', 0)} 轮 ｜ 间隔 {status.get('interval', 10)}s")


def _render_history(garden: dict) -> None:
    """WS 服务不可用时，直接读库展示历史决策。"""
    rows = db.agent_decision_history(garden["key"], limit=10)
    if rows:
        st.subheader("📜 Agent 决策历史（数据库）")
        df = pd.DataFrame([{
            "时间": (r.get("created_at") or "")[5:16],
            "决策": DECISION_CN.get(r.get("decision_type"), r.get("decision_type")),
            "等级": LEVEL_CN.get(r.get("overall_level", "low")),
            "摘要": (r.get("summary") or "")[:80],
        } for r in rows])
        st.dataframe(df, width="stretch", hide_index=True)
    else:
        st.info("暂无决策记录。启动 WS 服务后点击「立即执行一步」生成。")


if __name__ == "__main__":
    main()
