"""风险预警页：分级预警机制 + 气象灾害/病虫害研判 + 大模型复核。

预警分级：
  紧急处置(红) <- 高风险；提前预防(橙) <- 中风险；持续观察(黄) <- 低风险
  整园预警档位取所有风险中最高的那一档；任一新增/升级风险即触发一次预警。
"""
from __future__ import annotations

import streamlit as st

from src import db, ui
from src.auth_ui import current_user
from datetime import date as _date
from src.risk import TIER_META, TIER_ORDER, build_weather_snapshot, overall_tier, tier_action, tier_of

ui.inject_css()
garden = ui.select_garden()
data = ui.get_weather(garden)

st.title(f"🚨 风险预警 · {garden['name']}")
st.caption("分级预警机制：高风险=紧急处置，中风险=提前预防，低风险=持续观察。整园档位取最高风险。")
if data is None:
    st.stop()

now = data["now"]["now"]
daily = data["daily"]["daily"]
engine = ui.risk_engine()
overall = engine.assess_all(now, daily)

ui.overall_risk_banner(overall)

# ===== 📹 摄像头监测预警（每日推送一次：拍摄画面 + 病虫害情况 + 处理情况）=====
with st.expander("📹 摄像头监测预警（每日推送）", expanded=True):
    st.caption("基于多路摄像头实时监测：每天推送一次茶园视频监测日报（拍摄画面 + 病虫害检出 + Agent 处理情况）。")
    latest_alert = db.latest_camera_alert(garden.get("key", garden.get("name", "")))
    if latest_alert:
        _a = latest_alert
        _ac = ui.LEVEL_COLOR.get(_a.get("level", "low"), "#888")
        _al = ui.LEVEL_CN.get(_a.get("level", "low"), _a.get("level", ""))
        c_img, c_txt = st.columns([1, 2])
        with c_img:
            if _a.get("image_base64"):
                import base64 as _b64
                try:
                    st.image(_b64.b64decode(_a["image_base64"]), caption=f"拍摄画面 · {_a.get('alert_date', '')}",
                             width="stretch")
                except Exception:
                    st.caption("（画面解码失败）")
            else:
                st.caption("（当日无可用拍摄画面）")
        with c_txt:
            st.markdown(f"<span style='background:{_ac};color:#fff;border-radius:4px;padding:2px 8px;"
                        f"font-size:13px'>{_al}</span> &nbsp; **{_a.get('title', '')}**",
                        unsafe_allow_html=True)
            st.markdown(_a.get("content", ""))
            st.markdown(f"**🤖 处理情况：** {_a.get('actions_text', '')}")
            st.caption(f"推送时间 {_a.get('created_at', '')}")
    else:
        st.info("暂无摄像头监测预警：Agent 运行后每天固定时间自动推送一次（默认 09:00，可用 ALERT_DAILY_TIME 配置）。")

    _hist = db.camera_alert_history(garden.get("key", garden.get("name", "")), limit=15)
    if _hist:
        with st.expander(f"📋 摄像头监测预警历史（最近 {len(_hist)} 条）", expanded=False):
            for _r in _hist:
                _hc = ui.LEVEL_COLOR.get(_r.get("level", "low"), "#888")
                st.markdown(
                    f"<span style='color:{_hc};font-weight:700'>{ui.LEVEL_CN.get(_r.get('level','low'),'')}</span> · "
                    f"<b>{_r.get('alert_date','')}</b> · {_r.get('title','')}",
                    unsafe_allow_html=True,
                )
                st.caption(_r.get("content", ""))
                if _r.get("actions_text"):
                    st.caption("🤖 处理情况：" + _r["actions_text"])
                st.divider()
    else:
        st.caption("暂无历史预警记录。")

# ===== 智能融合研判（ML 分类 + 时序 + Bandit 择优）=====
from src import ensemble
_ens = None
_st_ens = None
try:
    _st_ens = ensemble.prepare(garden, data)
    _ens = ensemble.run_ensemble(garden, now, daily, data.get("air"))
    ensemble.set_last_arm(garden, _ens["recommended_action"]["arm"])
except Exception as _e_ens:
    _ens = None

if _ens is not None:
    with st.expander("🧠 智能融合研判（ML 分类 + 时序 + Bandit 择优）", expanded=True):
        st.caption(
            f"引擎状态：ML分类{'✅就绪' if _st_ens['ml_ready'] else '⏳待数据积累后自动训练'} · "
            f"历史数据 {_st_ens['history_days']} 天 · Bandit 已学习 {_st_ens['bandit_updates']} 次反馈"
        )
        _lv = _ens["overall_level"]
        _c = ui.LEVEL_COLOR[_lv]
        _p = _ens["proba"]
        st.markdown(
            f"**融合风险等级：** <span class='risk-badge' style='background:{_c}'>{ui.LEVEL_CN[_lv]}</span>",
            unsafe_allow_html=True,
        )
        st.caption(f"模型概率：低 {_p['low']:.0%} · 中 {_p['medium']:.0%} · 高 {_p['high']:.0%}")
        st.markdown("**未来 7 天 ML 逐日风险预测**")
        _days = " ｜ ".join(
            f"{d['fxDate'][5:]}{ui.LEVEL_CN[d['level']]}" for d in _ens["ml"]["per_day"]
        )
        st.write(_days if _ens["ml"]["per_day"] else "（暂无预报数据）")
        _tw = _ens["ts"]["window"]
        if _tw.get("n"):
            st.caption(
                f"时序窗口(近{_tw['n']}天)：均温 {_tw['temp_max_mean']}°C · 高温 {_tw['hot_days']} 天 · "
                f"降水累计 {_tw['precip_sum']}mm · 温度趋势 {_tw['temp_trend']}"
            )
        if _ens["ts"]["forecast"]:
            _f = _ens["ts"]["forecast"]
            st.caption(f"时序预测(未来{len(_f['temp_max'])}天最高温)：{' / '.join(map(str, _f['temp_max']))}°C")
        _act = _ens["recommended_action"]
        st.markdown(f"**🤖 Bandit 智能推荐处置：`{_act['arm']}`**")
        st.info(_act["advice"])
        c_fb1, c_fb2 = st.columns(2)
        if c_fb1.button("✅ 已按建议处置", key="fb_act"):
            ensemble.record_feedback(
                garden, now, _ens["proba"], _lv, adopted=True, detail="已处置",
                user_id=(current_user() or {}).get("id"),
            )
            st.success("已记录反馈，Bandit 将据此优化后续推荐")
            st.rerun()
        if c_fb2.button("😌 风险未发生 / 已解除", key="fb_low"):
            ensemble.record_feedback(
                garden, now, _ens["proba"], _lv, actual_level="low", detail="风险未发生",
                user_id=(current_user() or {}).get("id"),
            )
            st.success("已记录反馈（成真=低风险），用于评估推荐有效性")
            st.rerun()


# 当前整园预警档位
tier = overall_tier(overall)
meta = TIER_META[tier]
st.markdown(
    f"""
    <div style="border:1px solid {meta['color']};border-left:6px solid {meta['color']};
         border-radius:8px;padding:12px 16px;margin-bottom:8px">
      当前预警档位：<span style="color:{meta['color']};font-weight:700">{meta['label']}</span>
    </div>
    """,
    unsafe_allow_html=True,
)
if not overall["items"]:
    st.success("当前未检测到需要预警的风险项，茶园状态良好，可持续观察。")
else:
    st.caption(f"本次共研判出 {len(overall['items'])} 项风险，按预警档位分组如下（如需按类目浏览请切换下方标签页）。")

# ---------- 录存图像（登录用户：同茶园同天只保留一条快照，多次打开更新） ----------
_user_r = current_user()
if _user_r and garden.get("db_id") is not None:
    try:
        _lv = TIER_META[tier].get("label", tier)
        _hi = sum(1 for i in overall["items"] if i.get("level") == "high")
        _mid = sum(1 for i in overall["items"] if i.get("level") == "medium")
        db.save_analysis(
            _user_r["id"], garden.get("db_id"), str(garden.get("name", "")),
            "risk", _date.today(), overall["overall_level"],
            "整园档位：" + str(_lv) + f"（{len(overall['items'])} 项风险：高{_hi} · 中{_mid}）",
            "以上为该茶园当日风险研判快照。",
            overall["items"],
        )
    except Exception:
        pass

# ---- 分级预警处置台（三档分组） ----
with st.expander("⚙️ 分级预警处置台：按档位查看并处置风险", expanded=True):
    # 档位展示顺序：紧急处置 -> 提前预防 -> 持续观察
    group = {k: [] for k in TIER_ORDER}
    for item in overall["items"]:
        group[tier_of(item)].append(item)
    for tier_key in ("act", "prevent", "observe"):
        t_meta = TIER_META[tier_key]
        items = group[tier_key]
        st.markdown(
            f"### <span style='color:{t_meta['color']}'>■</span> {t_meta['label']}（{len(items)} 项）",
            unsafe_allow_html=True,
        )
        if not items:
            st.caption("本档位当前无风险。")
        for item in items:
            color = ui.LEVEL_COLOR[item["level"]]
            with st.container(border=True):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"**{item['name']}** — {item.get('condition','')}")
                    with st.expander("处置动作", expanded=True):
                        st.markdown(f"**{tier_action(item)}**")
                        st.caption(f"💡 建议：{item.get('advice','')}")
                with col2:
                    st.markdown(
                        f'<div class="risk-badge" style="background:{color}">{ui.LEVEL_CN[item["level"]]}</div>',
                        unsafe_allow_html=True,
                    )

# ---- 各标签页：按类目细看 ----
tab_dis, tab_pest, tab_llm = st.tabs(["🏔️ 气象灾害", "🐛 病虫害风险", "🤖 大模型研判"])

with tab_dis:
    st.subheader("气象灾害研判")
    if overall["disasters"]:
        for item in overall["disasters"]:
            ui.render_risk_item(item)
    else:
        st.success("未来几天暂无显著气象灾害风险。")

with tab_pest:
    st.subheader("茶树病虫害风险")
    st.caption("基于当前温湿度与物候特征的规则研判，供田间参考。")
    if overall["pests"]:
        for item in overall["pests"]:
            ui.render_risk_item(item)
    else:
        st.success("当前温湿度条件下，暂未识别出高发病虫害。")

with tab_llm:
    st.subheader("大模型智能研判")
    st.write(
        "基于当前实时天气与预报数据生成气象快照，调用百度千帆（伏羲/文心）大模型"
        "对气象灾害与病虫害风险进行综合研判，并给出农事建议。"
    )
    snapshot = build_weather_snapshot(garden, data["now"], daily, data["air"])
    with st.expander("查看本次研判输入的气象快照", expanded=False):
        st.code(snapshot, language="text")

    if st.button("调用大模型进行综合研判", width="stretch", type="primary"):
        from src.llm import QianfanClient
        try:
            with st.spinner("伏羲/文心大模型研判中，请稍候…"):
                result = QianfanClient().assess_with_llm(snapshot)
            st.success("研判完成")
            st.markdown(result)
        except Exception as e:  # noqa: BLE001
            st.error(f"大模型调用失败：{e}")
# ---------- 历史记录（登录用户：按日期翻页，回看当日全部风险快照） ----------
with st.expander("📋 风险预警历史记录（按天存档）"):
    if (not _user_r) or garden.get("db_id") is None:
        st.info("历史按自家茶园保存：登录并选择你的茶园后即可回看风险预警历史。")
        st.stop()
    if "risk_hist_page" not in st.session_state:
        st.session_state["risk_hist_page"] = 1
    _page = st.session_state["risk_hist_page"]
    _per = 6
    _total = db.analysis_count(_user_r["id"], "risk", garden.get("db_id"))
    if _total == 0:
        st.caption("暂无历史：打开本页完成风险研判后会自动记录（每天一条，当天多次查看自动更新）。")
    else:
        _rows = db.analysis_records(_user_r["id"], "risk", garden.get("db_id"), _page, _per)
        _pages = max(1, -(-_total // _per))
        st.caption(f"共 {_total} 天 存档 · 第 {_page}/{_pages} 页（日期倒序）")
        for _r in _rows:
            _lvl = str(_r.get("overall_level") or "low")
            # 兼容历史旧数据：早期曾把档位(act/prevent/observe)存入该字段
            if _lvl in TIER_META:
                _lvl = {"act": "high", "prevent": "medium", "observe": "low"}[_lvl]
            _col = ui.LEVEL_COLOR.get(_lvl, "#888")
            _lbl = ui.LEVEL_CN.get(_lvl, _lvl)
            st.markdown(f"<span style='color:{_col};font-weight:700'>{_lbl}</span> · "
                        f"<b>{_r['scoped_date']}</b> · {_r.get('title') or ''}", unsafe_allow_html=True)
            for _x in (_r.get("_items") or []):
                _c = ui.LEVEL_COLOR.get(_x.get("level", "low"), "#888")
                with st.container(border=True):
                    st.markdown(f"**{_x.get('name','')}** — {_x.get('condition','')} "
                                f"<span style='background:{_c};color:#fff;border-radius:4px;padding:2px 6px;font-size:12px'>"
                                f"{ui.LEVEL_CN.get(_x.get('level',''), _x.get('level',''))}</span>",
                                unsafe_allow_html=True)
                    if _x.get("advice"):
                        st.caption("💡 建议：" + _x["advice"])
            if st.button("🗑 删除当天", key=f"risk_del_{_r['id']}"):
                db.delete_analysis_record(_r["id"], _user_r["id"])
                st.rerun()
            st.divider()
        cl2, _m2, cr2 = st.columns([1, 2, 1])
        if cl2.button("← 上一页", disabled=_page <= 1, key="risk_prev",
                      on_click=lambda: st.session_state.update(risk_hist_page=max(1, _page - 1))):
            pass
        if cr2.button("下一页 →", disabled=_page >= _pages, key="risk_next",
                      on_click=lambda: st.session_state.update(risk_hist_page=min(_pages, _page + 1))):
            pass
