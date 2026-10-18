"""开发者控制台页面主体（不在 pages/ 下，避免被 Streamlit 自动列入侧边栏）。

由 pages/6_开发者控制台.py 薄壳在 DEV_MODE=true 时调用 main()。
"""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from src import backtest, config, db, ensemble, ml_risk, ui


def main() -> None:
    ui.inject_css()

    # 启动时确保扩展表（含 model_eval_log）已建
    try:
        db.init_db_ext()
    except Exception:  # noqa: BLE001
        pass

    if not config.is_dev_mode():
        st.warning("🛠 开发者模式未开启。请在项目根目录 `.env` 中设置 `DEV_MODE=true` 后重启应用。")
        st.caption("普通用户无需此页面；关闭开关后本页仅显示本条提示。")
        st.stop()

    st.title("🛠 开发者控制台")
    st.caption("仅开发者可见：模型状态、数据体检、滚动回测、评估历史、手动重训。")

    garden = ui.select_garden()

    tab_status, tab_data, tab_backtest, tab_history, tab_manual = st.tabs(
        ["📊 模型状态", "🩺 数据体检", "🧪 一键回测", "📜 评估历史", "⚙️ 手动操作"]
    )

    # -----------------------------------------------------------------------
    # Tab 1：模型状态
    # -----------------------------------------------------------------------
    with tab_status:
        st.subheader("当前模型状态")
        ml_ready = ml_risk.classifier.load()
        status = ensemble.ensure_models(garden)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ML 分类器", "就绪" if status["ml_ready"] else "未就绪")
        c2.metric("当前茶园历史天数", status["history_days"])
        c3.metric("Bandit 更新次数", status["bandit_updates"])
        c4.metric("模型文件", "存在" if ml_risk.MODEL_PATH.exists() else "缺失")

        st.markdown("**各茶园历史数据量（演示园）**")
        rows = []
        for g in config.load_gardens():
            cnt = db.weather_history_count(g["id"])
            rows.append({"茶园": g["name"], "garden_key": g["id"], "历史天数": cnt,
                         "可训练": "✅" if cnt >= 30 else "❌(<30)"})
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    # -----------------------------------------------------------------------
    # Tab 2：数据体检
    # -----------------------------------------------------------------------
    with tab_data:
        st.subheader("数据体检")
        total_hist = sum(db.weather_history_count(g["id"]) for g in config.load_gardens())
        fb = db.risk_feedback_stats(garden["key"])
        total_fb = sum(v["n"] for v in fb.values())
        labeled_fb = sum(v["labeled"] for v in fb.values())
        c1, c2, c3 = st.columns(3)
        c1.metric("weather_history 总记录", total_hist)
        c2.metric("risk_feedback 总反馈", total_fb)
        c3.metric("已标注实际等级", labeled_fb)

        st.markdown("**反馈分布（当前茶园，按预测等级）**")
        if fb:
            st.dataframe(pd.DataFrame([
                {"预测等级": k, "条数": v["n"], "采纳数": v["adopted"], "已标注": v["labeled"]}
                for k, v in fb.items()
            ]), width="stretch", hide_index=True)
        else:
            st.info("暂无风险反馈数据。用户在风险预警页点击「采纳/未发生」后会积累。")

    # -----------------------------------------------------------------------
    # Tab 3：一键回测
    # -----------------------------------------------------------------------
    with tab_backtest:
        st.subheader("时间序列滚动回测")
        st.caption("用历史段训练、未来段测试，衡量 ML 分类器的真实泛化能力（防泄漏）。")

        col_p1, col_p2, col_p3 = st.columns(3)
        min_train = col_p1.number_input("最小训练天数", min_value=14, max_value=180, value=30, step=7)
        test_win = col_p2.number_input("测试窗口（天）", min_value=3, max_value=30, value=7)
        lookback = col_p3.number_input("回溯历史（天）", min_value=30, max_value=720, value=360, step=30)

        if st.button("🚀 运行滚动回测", type="primary"):
            with st.spinner("回测中（滚动窗口训练+预测，约需几秒）…"):
                result = backtest.rolling_backtest(
                    garden["key"], garden,
                    min_train=int(min_train), test=int(test_win), days=int(lookback),
                )
            st.session_state["last_backtest"] = result

        result = st.session_state.get("last_backtest")
        if result:
            if not result.get("ok"):
                st.error(result.get("reason", "回测失败"))
            else:
                st.success(backtest.verdict(result))
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("测试样本数", result["sample_count"])
                m2.metric("准确率", f"{result['accuracy']:.1%}")
                m3.metric("宏平均 F1", f"{result['macro_f1']:.3f}")
                m4.metric("高风险召回率", f"{result['high_recall']:.1%}")

                st.markdown("**混淆矩阵（行=真实，列=预测）**")
                cm = result["confusion"]
                lv_cn = {"low": "低", "medium": "中", "high": "高"}
                cm_df = pd.DataFrame(
                    cm,
                    index=[f"真实·{lv_cn[lv]}" for lv in result["levels"]],
                    columns=[f"预测·{lv_cn[lv]}" for lv in result["levels"]],
                )
                st.dataframe(cm_df, width="stretch")

                with st.expander("查看完整分类报告"):
                    st.json(result["report"])

                if st.button("💾 保存本次评估到数据库"):
                    db.save_eval_log(
                        garden_key=garden["key"],
                        eval_type="rolling",
                        sample_count=result["sample_count"],
                        accuracy=result["accuracy"],
                        macro_f1=result["macro_f1"],
                        high_recall=result["high_recall"],
                        confusion_json=json.dumps(result["confusion"], ensure_ascii=False),
                        report_json=json.dumps(result["report"], ensure_ascii=False),
                    )
                    st.success("已保存到 model_eval_log，可在「评估历史」查看。")

    # -----------------------------------------------------------------------
    # Tab 4：评估历史
    # -----------------------------------------------------------------------
    with tab_history:
        st.subheader("历次模型评估记录")
        logs = db.eval_log_history(limit=50)
        if not logs:
            st.info("暂无评估记录。到「一键回测」运行并保存后会出现在这里。")
        else:
            df = pd.DataFrame(logs)
            df["created_at"] = pd.to_datetime(df["created_at"]).dt.strftime("%Y-%m-%d %H:%M")
            show = df[["created_at", "garden_key", "sample_count", "accuracy",
                       "macro_f1", "high_recall"]].rename(columns={
                "created_at": "评估时间", "garden_key": "茶园",
                "sample_count": "样本数", "accuracy": "准确率",
                "macro_f1": "宏F1", "high_recall": "高风险召回",
            })
            st.dataframe(show, width="stretch", hide_index=True)

            with st.expander("查看某次评估的混淆矩阵与完整报告"):
                idx = st.selectbox("选择记录", range(len(logs)),
                                   format_func=lambda i: f"{logs[i]['created_at']}  {logs[i]['garden_key']}")
                row = logs[idx]
                cm = json.loads(row["confusion_json"]) if row.get("confusion_json") else []
                if cm:
                    lv_cn = {"low": "低", "medium": "中", "high": "高"}
                    st.dataframe(pd.DataFrame(
                        cm,
                        index=[f"真实·{lv_cn[lv]}" for lv in ["low", "medium", "high"]],
                        columns=[f"预测·{lv_cn[lv]}" for lv in ["low", "medium", "high"]],
                    ), width="stretch")
                if row.get("report_json"):
                    st.json(json.loads(row["report_json"]))

    # -----------------------------------------------------------------------
    # Tab 5：手动操作
    # -----------------------------------------------------------------------
    with tab_manual:
        st.subheader("手动操作")
        if st.button("🔄 强制重训 ML 分类器", type="primary"):
            with st.spinner("重训中…"):
                status = ensemble.ensure_models(garden, force_retrain=True)
            st.success(f"重训完成：ML就绪={status['ml_ready']}，历史天数={status['history_days']}")

        if st.button("🧹 清除页面缓存"):
            st.cache_data.clear()
            st.success("缓存已清除，刷新页面生效。")

        st.divider()
        st.caption("提示：日常重训由每日 05:30 定时任务自动执行；此处用于调试或强制更新。")
