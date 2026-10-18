"""我的茶园：登录用户管理自己的茶园（增、删、改）。"""
from __future__ import annotations

import streamlit as st

from src import db, ui
from src.auth_ui import current_user, render_auth_ui

ui.inject_css()

render_auth_ui()
user = current_user()

st.title("🌱 我的茶园")

if not user:
    st.warning("请先登录，登录后即可管理自己的茶园。")
    st.stop()

st.caption(f"当前用户：{user['nickname'] or user['username']}（ID: {user['id']}）")


def _garden_to_dict(g) -> dict:
    """把数据库行转成表单需要的字段。"""
    return {
        "name": g.get("name", ""),
        "location": g.get("location") or "",
        "location_id": g.get("location_id") or "",
        "lon": g.get("lon"),
        "lat": g.get("lat"),
        "altitude_m": g.get("altitude_m"),
        "variety": g.get("variety") or "",
        "notes": g.get("notes") or "",
    }


def _collect(d: dict, prefix) -> dict:
    """从 session_state 里收集这个表单填的值。"""
    lon = st.session_state.get(f"{prefix}_lon")
    lat = st.session_state.get(f"{prefix}_lat")
    alt = st.session_state.get(f"{prefix}_alt")
    return {
        "name": st.session_state.get(f"{prefix}_name", "").strip(),
        "location": st.session_state.get(f"{prefix}_loc", "").strip(),
        "location_id": st.session_state.get(f"{prefix}_locid", "").strip(),
        "lon": float(lon) if lon not in (None, "") else d.get("lon"),
        "lat": float(lat) if lat not in (None, "") else d.get("lat"),
        "altitude_m": float(alt) if alt not in (None, "") else d.get("altitude_m"),
        "variety": st.session_state.get(f"{prefix}_variety", "").strip(),
        "notes": st.session_state.get(f"{prefix}_notes", "").strip(),
    }


def _validate(g: dict) -> str | None:
    if not g["name"]:
        return "茶园名称不能为空"
    lon, lat = g.get("lon"), g.get("lat")
    has_coords = lon is not None and lat is not None
    if not g["location_id"] and not has_coords:
        return "请填写 LocationID 或经纬度至少一项"
    if has_coords:
        if not -180 <= float(lon) <= 180 or not -90 <= float(lat) <= 90:
            return "经纬度超出有效范围（经度 -180~180，纬度 -90~90）"
        if abs(float(lon)) < 1e-6 and abs(float(lat)) < 1e-6:
            return "经纬度不能同时为 0，请填写真实坐标或改用 LocationID"
    return None


# 正在编辑的茶园 id（0 表示无）
if "edit_garden_id" not in st.session_state:
    st.session_state["edit_garden_id"] = None

# ---------- 添加茶园 ----------
with st.expander("➕ 添加新茶园", expanded=False):
    with st.form("add_garden_form"):
        c1, c2 = st.columns(2)
        name = c1.text_input("茶园名称 *", key="add_name")
        variety = c2.text_input("品种", key="add_variety")
        loc = st.text_input("所属地区（如：浙江省杭州市西湖区）", key="add_loc")
        lid = st.text_input("和风天气 LocationID（可留空，用下方经纬度）", key="add_locid")
        c3, c4, c5 = st.columns(3)
        lon = c3.number_input("经度", key="add_lon", format="%.4f", value=None, placeholder="如 120.1234")
        lat = c4.number_input("纬度", key="add_lat", format="%.4f", value=None, placeholder="如 30.1234")
        alt = c5.number_input("海拔(m)", key="add_alt", format="%.1f", value=None, placeholder="如 200")
        notes = st.text_area("备注", key="add_notes")
        submitted = st.form_submit_button("保存茶园")
    if submitted:
        g = _collect(_garden_to_dict({}), "add")
        err = _validate(g)
        if err:
            st.error(err)
        else:
            ok, msg = db.add_garden(user["id"], g)
            if ok:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)

st.divider()

# ---------- 我的茶园列表 ----------
gardens = db.user_gardens(user["id"])
if not gardens:
    st.info("你还没有茶园，使用上方表单添加第一个茶园吧。")
else:
    st.subheader(f"我的茶园（{len(gardens)}）")
    for g in gardens:
        gd = _garden_to_dict(g)
        editing = st.session_state.get("edit_garden_id") == g["id"]
        with st.container(border=True):
            c1, c2, c3 = st.columns([3, 2, 1])
            with c1:
                st.markdown(f"**{g['name']}**")
                st.caption(f"{g.get('location') or '-'} ｜ 品种 {g.get('variety') or '-'} ｜ 海拔 {g.get('altitude_m') or '-'}m")
            with c2:
                lon = g.get("lon"); lat = g.get("lat")
                st.caption(f"LocationID: {g.get('location_id') or '-'}")
                st.caption(f"经纬度: {lon if lon else '-'}, {lat if lat else '-'}")
            with c3:
                if st.button("✏️ 编辑", key=f"edit_{g['id']}"):
                    st.session_state["edit_garden_id"] = g["id"] if not editing else None
                    st.rerun()
                if st.button("🗑 删除", key=f"del_{g['id']}"):
                    ok, msg = db.delete_garden(g["id"], user["id"])
                    if ok:
                        if st.session_state.get("edit_garden_id") == g["id"]:
                            st.session_state["edit_garden_id"] = None
                        st.session_state.pop("garden_id", None)
                        st.rerun()
                    else:
                        st.error(msg)

            # ---- 编辑表单（内联） ----
            if editing:
                st.divider()
                with st.form(f"edit_form_{g['id']}"):
                    e1, e2 = st.columns(2)
                    e_name = e1.text_input("茶园名称 *", value=gd["name"], key=f"e_name_{g['id']}")
                    e_variety = e2.text_input("品种", value=gd["variety"], key=f"e_variety_{g['id']}")
                    e_loc = st.text_input("所属地区", value=gd["location"], key=f"e_loc_{g['id']}")
                    e_lid = st.text_input("LocationID", value=gd["location_id"], key=f"e_lid_{g['id']}")
                    e3, e4, e5 = st.columns(3)
                    e_lon = e3.number_input("经度", value=gd["lon"], format="%.4f", key=f"e_lon_{g['id']}")
                    e_lat = e4.number_input("纬度", value=gd["lat"], format="%.4f", key=f"e_lat_{g['id']}")
                    e_alt = e5.number_input("海拔(m)", value=gd["altitude_m"], format="%.1f", key=f"e_alt_{g['id']}")
                    e_notes = st.text_area("备注", value=gd["notes"], key=f"e_notes_{g['id']}")
                    ec1, ec2 = st.columns(2)
                    submitted = ec1.form_submit_button("💾 保存修改", type="primary")
                    canceled = ec2.form_submit_button("取消")

                if canceled:
                    st.session_state["edit_garden_id"] = None
                    st.rerun()
                if submitted:
                    ps = st.session_state
                    ng = {
                        "name": ps.get(f"e_name_{g['id']}", "").strip(),
                        "location": ps.get(f"e_loc_{g['id']}", "").strip(),
                        "location_id": ps.get(f"e_lid_{g['id']}", "").strip(),
                        "lon": float(ps.get(f"e_lon_{g['id']}")) if ps.get(f"e_lon_{g['id']}") else None,
                        "lat": float(ps.get(f"e_lat_{g['id']}")) if ps.get(f"e_lat_{g['id']}") else None,
                        "altitude_m": float(ps.get(f"e_alt_{g['id']}")) if ps.get(f"e_alt_{g['id']}") else None,
                        "variety": ps.get(f"e_variety_{g['id']}", "").strip(),
                        "notes": ps.get(f"e_notes_{g['id']}", "").strip(),
                    }
                    err = _validate(ng)
                    if err:
                        st.error(err)
                    else:
                        ok, msg = db.update_garden(g["id"], user["id"], ng)
                        if ok:
                            st.session_state["edit_garden_id"] = None
                            st.session_state.pop("garden_id", None)
                            st.success(msg)
                            st.rerun()
                        else:
                            st.error(msg)
