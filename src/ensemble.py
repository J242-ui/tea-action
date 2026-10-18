"""三引擎融合引擎（风险预警 + 智能推送的编排中枢）。

流程（一次"预测 → 推荐 → 反馈"闭环）：
  1. 数据打底  ensure_data()  ：无历史则生成模拟历史；并把今日预报归档
  2. 模型就绪  ensure_models()：懒加载/训练 LightGBM 分类器 + LinUCB bandit
  3. 三引擎融合 run_ensemble()：
       - 规则引擎（兜底 + 具体风险项）
       - ML 分类（未来每日风险等级概率）
       - 时序（历史窗口特征 + 未来指标趋势）
       融合成 overall_level + 概率 + 具体处置建议
  4. bandit 在线择优推荐处置动作（context=环境因子+风险概率）
  5. 反馈闭环  record_feedback()：采纳/成真 → 回写 bandit + 落库反馈样本
"""
from __future__ import annotations

from datetime import date

from . import bandit as bandit_mod
from . import config, db, ml_risk, risk, ts

ARM_ADVICE = {
    "补水灌溉": "立即补水灌溉（滴灌/喷灌为宜），行间覆盖稻草保墒。",
    "遮阳降温": "覆盖遮阳网，午后短时喷灌增湿降温，避免强光时段作业。",
    "防冻保暖": "覆盖遮阳网/防冻布，可熏烟增温，幼龄茶园浇透水保根防冻。",
    "排水防渍": "提前疏通排水沟渠，低洼茶园做好排涝防渍准备。",
    "喷药防治": "达到防治指标时选用对应药剂，轮换用药，注意安全间隔期。",
    "暂缓作业": "当前天气不适宜农事操作，暂缓修剪/施肥/采摘。",
}

_ORDER = {"low": 0, "medium": 1, "high": 2}


def _max_level(levels: list[str]) -> str:
    if not levels:
        return "low"
    return max(levels, key=lambda lv: _ORDER.get(lv, 0))


def garden_variety(garden: dict) -> str:
    return garden.get("variety") or ""


def _gkey(garden: dict) -> str:
    """兼容 {key} / {id} / 无键三种 garden 形态，避免 KeyError。"""
    return garden.get("key") or garden.get("id") or "longjing"


def _daily_rows(data: dict | None) -> list[dict]:
    """兼容和风原始响应与页面包装后的嵌套响应，返回有效逐日预报。"""
    if not isinstance(data, dict):
        return []
    rows = data.get("daily", [])
    if isinstance(rows, dict):
        rows = rows.get("daily", [])
    if not isinstance(rows, list):
        return []
    return [d for d in rows if isinstance(d, dict) and d.get("fxDate")]


# ---------------------------------------------------------------------------
# 数据打底
# ---------------------------------------------------------------------------
def ensure_data(garden: dict) -> int:
    """确保该茶园有历史气象数据：无则生成模拟历史；同时归档今日预报。返回历史条数。"""
    key = _gkey(garden)
    ml_risk.seed_history(key, garden, days=180)
    return db.weather_history_count(key)


def archive_today(garden: dict, data: dict) -> int:
    """把和风今日返回的 daily 预报归档进历史表（供后续时序/训练用）。

    每天每个茶园只归档一次：若 weather_history 已有今天记录则跳过，
    避免页面每次渲染都重复 UPSERT（降低写库与数据冗余）。返回本次写入天数。
    """
    daily = _daily_rows(data)
    if not daily:
        return 0
    key = _gkey(garden)
    today = date.today().isoformat()
    # 预报会同时写入未来多天，不能只检查“日期最大的一条”是否等于今天。
    if db.weather_history_has_date(key, today):
        return 0
    db.archive_weather(key, daily)
    return len(daily)


def prepare(garden: dict, data: dict | None = None) -> dict:
    """页面统一入口：归档今日预报 + 数据打底 + 模型懒加载。返回引擎状态。"""
    if data is not None:
        archive_today(garden, data)
    ensure_data(garden)
    return ensure_models(garden)


# ---------------------------------------------------------------------------
# 模型就绪
# ---------------------------------------------------------------------------
def ensure_models(garden: dict, force_retrain: bool = False) -> dict:
    """懒加载分类器与 bandit；样本足够且模型缺失时训练。返回引擎状态。"""
    status = {"ml_ready": False, "ts_ready": False, "history_days": 0, "bandit_updates": 0}
    history = db.weather_history_series(_gkey(garden), days=365)
    status["history_days"] = len(history)

    # 分类器
    loaded = ml_risk.classifier.load()
    if not loaded or force_retrain:
        ml_risk.classifier.train_from_history(_gkey(garden), garden, history=history, force=force_retrain)
        if not ml_risk.classifier.ready:
            ml_risk.classifier.load()
    status["ml_ready"] = ml_risk.classifier.ready

    # bandit
    bandit_mod.bandit.load()
    status["bandit_updates"] = bandit_mod.bandit.n_updates
    return status


# ---------------------------------------------------------------------------
# 三引擎融合
# ---------------------------------------------------------------------------
def run_ensemble(garden: dict, now: dict, daily: list[dict], air: dict | None = None) -> dict:
    """执行一次完整的融合研判。now/daily 来自和风返回（含 'now'/'daily' 壳）。"""
    n = now.get("now", now) if isinstance(now, dict) else now
    dl = daily.get("daily", daily) if isinstance(daily, dict) else (daily or [])
    key = _gkey(garden)

    # --- 规则引擎（兜底，产出具体风险项） ---
    engine = risk.RiskEngine()
    rule = engine.assess_all(n, dl)

    # --- ML 分类 ---
    ml = ml_risk.classifier.predict_over_daily(dl, garden)

    # --- 时序 ---
    ts_window = ts.historical_window(key, window=7)
    ts_fcst = ts.forecast(key, horizon=3)

    # --- 融合等级（取规则与 ML 中更高者，保守预警）+ 概率 ---
    overall_level = _max_level([rule["overall_level"], ml["overall_level"]])
    proba = dict(ml["proba"])
    # 若 ML 未就绪(均匀概率)，按规则等级给一个主导概率
    if not ml_risk.classifier.ready:
        dominant = overall_level if overall_level != "low" else "low"
        base = 0.5 if dominant == "high" else 0.38
        proba = {"low": 0.0, "medium": 0.0, "high": 0.0}
        proba[dominant] = base
        rest = (1.0 - base) / 2
        for lv in ("low", "medium", "high"):
            if lv != dominant:
                proba[lv] = rest

    # --- bandit 择优处置动作 ---
    ctx = bandit_mod.build_context(n, garden, proba)
    arm = bandit_mod.bandit.recommend(ctx)

    return {
        "garden_key": key,
        "overall_level": overall_level,
        "proba": proba,
        "rule": rule,
        "ml": ml,
        "ts": {"window": ts_window, "forecast": ts_fcst},
        "recommended_action": {"arm": arm, "advice": ARM_ADVICE.get(arm, "")},
    }


# ---------------------------------------------------------------------------
# 反馈闭环
# ---------------------------------------------------------------------------
def record_feedback(garden: dict, now: dict, proba: dict | None,
                    predicted_level: str, kind: str = "risk",
                    adopted: bool = False, actual_level: str | None = None,
                    detail: str | None = None, user_id: int | None = None) -> None:
    """记录一次推送反馈：回写 bandit 并落库 risk_feedback。"""
    n = now.get("now", now) if isinstance(now, dict) else now
    ctx = bandit_mod.build_context(n, garden, proba)
    # reward：采纳给高分；事后风险确实没发生(成真=low)也给高分；成真 medium/high 给低分
    reward = 0.0
    if adopted:
        reward += 0.6
    if actual_level == "low":
        reward += 0.4
    elif actual_level in ("medium", "high"):
        reward += 0.0
    else:
        reward += 0.25  # 未标注，给中性分
    arm = garden.get("_last_arm") or "暂缓作业"
    bandit_mod.bandit.update(arm, ctx, reward)
    bandit_mod.bandit.save()
    db.log_risk_feedback(
        garden_key=_gkey(garden), kind=kind, predicted_level=predicted_level,
        risk_name=arm, actual_level=actual_level, adopted=adopted,
        detail=detail, user_id=user_id,
    )


def set_last_arm(garden: dict, arm: str) -> None:
    """在页面记录本次推荐的处置动作，供反馈时回写 bandit。"""
    garden["_last_arm"] = arm
