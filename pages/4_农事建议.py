"""农事建议页：基于天气与茶园物候生成的日常农事操作参考建议。

（原[智能推送]页转型：剥离风险告警，专注非紧急的日常农事建议，
  风险预警请查看 [风险预警] 页。）
"""
from __future__ import annotations

import streamlit as st

from src import db, ui
from src.auth_ui import current_user
from src.push import (
    CAT_CN, build_routine_cards, summarize, today_range_text,
)

ui.inject_css()
garden = ui.select_garden()
data = ui.get_weather(garden)

st.title("🌾 农事建议 · " + str(garden["name"]))
st.caption(f"{today_range_text()} · 基于实时天气与茶园物候给出的日常农事参考建议（不构成风险告警）")

# ---------- 农事类别筛选 ----------
if "farm_cats" not in st.session_state:
    st.session_state["farm_cats"] = {"water": True, "fertilize": True, "spray": True, "harvest": True}
with st.sidebar:
    st.markdown("### 🌾 建议范围")
    for key, label in CAT_CN.items():
        st.session_state["farm_cats"][key] = st.toggle(
            label, value=st.session_state["farm_cats"][key], key=f"cat_{key}"
        )

if data is None:
    st.stop()

now = data["now"]["now"]
daily = data["daily"]["daily"]

# ---------- 生成农事建议 ----------
engine = ui.risk_engine()
overall = engine.assess_all(now, daily)
cards = build_routine_cards(garden, data["now"], daily, data["air"], overall)

cats = st.session_state["farm_cats"]
filtered = [c for c in cards if cats.get(c.get("cat"), True)]

# ===== Bandit 智能推送推荐（在线择优置顶）=====
from src import ensemble
_ens_f = None
try:
    ensemble.prepare(garden, data)
    _ens_f = ensemble.run_ensemble(garden, now, daily, data.get("air"))
    ensemble.set_last_arm(garden, _ens_f["recommended_action"]["arm"])
except Exception:
    _ens_f = None

if _ens_f is not None:
    _act = _ens_f["recommended_action"]
    _lv = _ens_f["overall_level"]
    _c = ui.LEVEL_COLOR[_lv]
    with st.container(border=True):
        st.markdown(
            f"<span class='garden-title'>🤖 今日智能推荐</span> "
            f"<span class='risk-badge' style='background:{_c}'>融合风险 {ui.LEVEL_CN[_lv]}</span>",
            unsafe_allow_html=True,
        )
        st.write(f"**{_act['arm']}** —— " + _act["advice"])
        st.caption("由「ML 分类 + 时序 + Contextual Bandit」三引擎融合择优，随反馈持续优化。")
        _f1, _f2 = st.columns(2)
        if _f1.button("👍 采纳这条建议", key="farm_fb_ok"):
            ensemble.record_feedback(garden, now, _ens_f["proba"], _lv,
                                     kind="farm", adopted=True, detail="采纳智能推荐",
                                     user_id=(current_user() or {}).get("id"))
            st.success("已记录采纳，感谢反馈")
            st.rerun()
        if _f2.button("👎 不采纳", key="farm_fb_no"):
            ensemble.record_feedback(garden, now, _ens_f["proba"], _lv,
                                     kind="farm", adopted=False, detail="未采纳",
                                     user_id=(current_user() or {}).get("id"))
            st.success("已记录，将减少此类推荐")
            st.rerun()

# ---------- 摘要 ----------
summary = summarize(filtered)
col1, col2, col3 = st.columns(3)
col1.metric("今日农事建议", summary["total"])
col2.metric("会操作类别", len(summary["by_cat"]))
col3.metric("茶园", str(garden["name"]))

if not filtered:
    st.info("当前没有适用的日常农事建议。可调整左侧范围，或换一个茶园查看；若想评估是否会有风险侵袭，请前往 [风险预警] 页。")

# ---------- 记录历史（登录用户：同茶园同天只保留一条快照，多次打开仅更新） ----------
user = current_user()
from datetime import date as _today
if user and cards and garden.get("db_id") is not None:
    try:
        _cnt = len(cards)
        _cls = [CAT_CN.get(c.get("cat", ""), c.get("cat", "")) for c in cards]
        _cls = sorted(set(x for x in _cls if x))
        db.save_analysis(
            user["id"],
            garden.get("db_id"),
            str(garden.get("name", "")),
            "farm",
            _today.today(),
            "low",
            "今日农事建议 %d 项" % _cnt,
            "涉及类别：" + "/".join(_cls),
            cards,
        )
    except Exception:
        pass  # 记录失败不阻塞页面

# ---------- 渲染建议卡片 ----------
for c in filtered:
    tag = "🌱 日常建议"
    with st.container(border=True):
        head, tail = st.columns([4, 1])
        with head:
            st.markdown(
                f"<span class='garden-title'>{tag} · {CAT_CN.get(c.get('cat'), '农事')}</span> "
                f"<span class='meta'>{c['title']} · {c['time_text']}</span>",
                unsafe_allow_html=True,
            )
            st.write(c["body"])
        with tail:
            st.markdown(
                "<div class='risk-badge' style='background:#2e9e5b'>参考</div>",
                unsafe_allow_html=True,
            )
        if c.get("advice"):
            st.caption("💡 建议：" + c["advice"])

# ---------- 历史记录（登录用户：按天+分类查看，按日期翻页） ----------
if user:
    if garden.get("db_id") is None:
        st.info("历史按自家茶园保存：登录并选择你的茶园后即可回看农事建议历史。")
    else:
        with st.expander("📋 农事建议历史记录（按天存档）"):
            _opts = ["全部", *CAT_CN.values()]
            _cur = st.session_state.get("farm_hist_cat", "全部")
            _cat = st.selectbox("按类别筛选", _opts,
                                index=_opts.index(_cur) if _cur in _opts else 0,
                                key="farm_cat_sel")
            st.session_state["farm_hist_cat"] = _cat
            if "farm_hist_page" not in st.session_state:
                st.session_state["farm_hist_page"] = 1
            _page = st.session_state["farm_hist_page"]
            _per = 8
            _total = db.analysis_count(user["id"], "farm", garden.get("db_id"))
            if _total == 0:
                st.caption("暂无历史：通过本页查看农事建议后会自动记录（每天一条，当天多次查看自动更新）。")
            else:
                _rows = db.analysis_records(user["id"], "farm", garden.get("db_id"), _page, _per)
                _pages = max(1, -(-_total // _per))
                st.caption(f"共 {_total} 天 · 第 {_page}/{_pages} 页（日期倒序）")
                for _r in _rows:
                    _its = [x for x in (_r.get("_items") or [])
                            if _cat == "全部" or CAT_CN.get(x.get("cat", "")) == _cat]
                    st.markdown(f"<b>{_r['scoped_date']}</b> · {_r.get('title') or ''}",
                                unsafe_allow_html=True)
                    if not _its:
                        st.caption("（所选类别当日无记录）")
                    for _x in _its:
                        st.markdown("**%s** · %s · %s" % (CAT_CN.get(_x.get("cat", ""), _x.get("cat", "")),
                                                           _x.get("title", ""), _x.get("time_text", "")))
                        st.write(_x.get("body", ""))
                        if _x.get("advice"):
                            st.caption("💡 建议：" + _x["advice"])
                    if st.button("🗑 删除当天", key=f"farm_del_{_r['id']}"):
                        db.delete_analysis_record(_r["id"], user["id"])
                        st.rerun()
                    st.divider()
                cl, _m, cr = st.columns([1, 2, 1])
                if cl.button("← 上一页", disabled=_page <= 1, key="farm_prev",
                             on_click=lambda: st.session_state.update(farm_hist_page=max(1, st.session_state["farm_hist_page"] - 1))):
                    pass
                if cr.button("下一页 →", disabled=_page >= _pages, key="farm_next",
                             on_click=lambda: st.session_state.update(farm_hist_page=min(_pages, st.session_state["farm_hist_page"] + 1))):
                    pass

# ===== 大模型研判叠加块（天气 -> 大模型研判 -> 推送）=====
from src import risk as _risk_mod
from src.llm import QianfanClient as _QfFarm
_snap_farm = _risk_mod.build_weather_snapshot(garden, now, daily, data.get("air"))
_qf_farm = _QfFarm()
st.markdown("### 大模型农事研判")
if _qf_farm.demo:
    st.caption("提示：演示模式（未配置 QIANFAN_API_KEY），下方为占位演示；配置密钥后调用真实千帆大模型。")
if st.button("生成大模型农事研判", key="farm_llm_generate"):
    with st.spinner("正在调用大模型研判天气与农事建议……"):
        try:
            _sys_farm = "你是茶园农业专家，结合天气快照给出近几日可操作的农事建议，分条编号、简明中文。"
            _usr_farm = (
                "请研判农事操作建议（浇水/施肥/打药/采摘与物候等），"
                "聚焦日常田间作业，不要罗列灾害预警；3-6条。" + _snap_farm
            )
            _adv_farm = _qf_farm.chat(
                [{"role": "system", "content": _sys_farm},
                 {"role": "user", "content": _usr_farm}],
                temperature=0.5,
                max_tokens=1200,
            )
        except Exception as _e_farm:
            _adv_farm = "调用大模型研判失败：" + str(_e_farm)
    st.session_state["farm_llm_result"] = {
        "garden_key": garden["key"],
        "text": _adv_farm,
    }

_llm_result = st.session_state.get("farm_llm_result") or {}
if _llm_result.get("garden_key") == garden["key"] and _llm_result.get("text"):
    with st.container(border=True):
        st.caption("数据来源：实时天气 + 未来预报 · 千帆大模型研判（叠加参考，不与风险预警冲突）")
        st.write(_llm_result["text"])

# ===== 大模型研判叠加块结束 =====
