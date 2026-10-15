"""Agent 决策引擎：融合多源输入，输出可执行指令。

输入（一次决策的完整上下文）：
  soil        : 土壤湿度读数（src.sensor）
  detections  : 视觉病害检测结果（src.vision）
  weather     : {"now": {...}, "daily": [...]}（和风数据结构）
  risk        : 规则引擎研判结果（可选，src.risk.RiskEngine().assess_all）

决策规则（带滞回与防抖，避免执行器频繁启停）：
  1. 浇水（drip_valve）：
       - 触发：土壤湿度 < IRRIGATE_DRY 且未来 2 天无明显降水(<10mm)
       - 时长  ：由 (目标湿度 - 当前湿度) 估算
       - 停止  ：湿度回升到 IRRIGATE_STOP 或出现大雨
  2. 打药（spray_boom）：
       - 触发：连续 CONSECUTIVE_SPRAY 次检测到病害且置信度 ≥ SPRAY_CONF，
               且非雨天、风力 < 4 级
       - 停止  ：连续 CONSECUTIVE_STOP 次无病害，或遇雨/大风
  3. 降温喷灌（spray_boom 短时）：最高温 ≥33°C 且湿度 <40%（复用气象规则）

输出：
  {ts, decision_type, summary, overall_level, actions:[{device,op,duration_min,reason}]}
"""
from __future__ import annotations

from datetime import datetime

# ---- 可调阈值 ----
IRRIGATE_DRY = 45.0        # 土壤湿度低于该值触发浇水
IRRIGATE_STOP = 68.0       # 高于该值停止浇水（滞回）
IRRIGATE_TARGET = 72.0     # 灌溉目标湿度
MIN_PER_PCT = 1.2          # 每差 1% 湿度约需滴灌时长(分钟)
SPRAY_CONF = 0.50          # 病害置信度阈值
CONSECUTIVE_SPRAY = 2      # 连续 N 次检出病害才打药（防抖动）
CONSECUTIVE_STOP = 3       # 连续 N 次无病害才停
RAIN_LIMIT_MM = 10.0       # 未来单日降水 ≥ 该值视为雨天
WIND_LIMIT_LEVEL = 4       # 风力 ≥ 4 级不打药
COOL_TEMP = 33.0           # 降温喷灌触发温度
COOL_HUMIDITY = 40.0       # 降温喷灌触发湿度上限


def _num(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _max_precip(daily: list[dict]) -> float:
    return max([_num(d.get("precip")) for d in (daily or [])] + [0.0])


def _max_wind(daily: list[dict], now: dict) -> float:
    scales = [_num(d.get("windScaleDay")) for d in (daily or [])]
    if now:
        scales.append(_num(now.get("windScale")))
    return max(scales + [0.0])


def _diseases(detections: dict) -> list[dict]:
    """返回置信度达标的病害检测。"""
    return [d for d in detections.get("detections", [])
            if d["class"] != "Healthy" and _num(d.get("conf")) >= SPRAY_CONF]


def _irrigate_decision(soil: dict, daily: list[dict], valve_state: str) -> list[dict]:
    """浇水决策。"""
    actions: list[dict] = []
    value = _num(soil.get("value")) if soil else 0.0
    rain = _max_precip(daily)
    if valve_state == "open":
        if value >= IRRIGATE_STOP or rain >= RAIN_LIMIT_MM:
            actions.append({"device": "drip_valve", "op": "close",
                            "reason": f"土壤湿度 {value:.0f}% 已回升/出现降水，停止滴灌"})
    else:
        if value < IRRIGATE_DRY and rain < RAIN_LIMIT_MM:
            need = max(IRRIGATE_TARGET - value, 0.0)
            duration = max(5.0, round(need * MIN_PER_PCT, 1))
            actions.append({"device": "drip_valve", "op": "open", "duration_min": duration,
                            "reason": f"土壤湿度 {value:.0f}% 低于 {IRRIGATE_DRY:.0f}% 且近期无大雨，启动滴灌约 {duration:.0f} 分钟"})
    return actions


def _spray_decision(detections: dict, daily: list[dict], now: dict,
                    boom_state: str, counters: dict) -> list[dict]:
    """打药/降温喷灌决策。counters: {"spray_hits", "spray_misses"} 跨轮次保持。"""
    actions: list[dict] = []
    diseases = _diseases(detections)
    rainy = _max_precip(daily) >= RAIN_LIMIT_MM
    windy = _max_wind(daily, now) >= WIND_LIMIT_LEVEL
    temp = _num(now.get("temp")) if now else 0.0
    hum = _num(now.get("humidity")) if now else 100.0

    if boom_state == "on":
        # 正在作业：无病害且连续 miss 达标 → 停；遇雨/大风 → 立即停
        if diseases:
            counters["spray_misses"] = 0
        else:
            counters["spray_misses"] = counters.get("spray_misses", 0) + 1
        if rainy or windy or counters.get("spray_misses", 0) >= CONSECUTIVE_STOP:
            reason = ("出现降水/大风" if (rainy or windy)
                      else f"连续 {counters['spray_misses']} 次未检出病害")
            actions.append({"device": "spray_boom", "op": "off", "reason": f"停止喷药：{reason}"})
            counters["spray_misses"] = 0
    else:
        if rainy or windy:
            counters["spray_hits"] = 0
            counters["spray_misses"] = 0
        elif diseases:
            counters["spray_hits"] = counters.get("spray_hits", 0) + 1
            counters["spray_misses"] = 0
            if counters["spray_hits"] >= CONSECUTIVE_SPRAY:
                top = max(diseases, key=lambda d: _num(d.get("conf")))
                duration = 8.0 + int(top["class"] in ("Black rot", "Brown blight")) * 4.0
                actions.append({"device": "spray_boom", "op": "on", "duration_min": duration,
                                "reason": f"连续检出病害（{top['class']} {top['conf']:.2f}），启动喷杆打药约 {duration:.0f} 分钟"})
                counters["spray_hits"] = 0
        else:
            counters["spray_hits"] = 0
            counters["spray_misses"] = counters.get("spray_misses", 0) + 1
            # 高温低湿 → 短时喷灌降温（复用气象风险规则）
            if temp >= COOL_TEMP and hum < COOL_HUMIDITY and not rainy:
                actions.append({"device": "spray_boom", "op": "on", "duration_min": 5.0,
                                "reason": f"气温 {temp:.0f}°C 且湿度 {hum:.0f}%，短时喷灌降温增湿"})
    return actions


def decide(inputs: dict, state: dict | None = None) -> dict:
    """执行一次融合决策。state 为跨轮次状态（打药连续计数等），会原地更新。"""
    # 注意：不能用 `state = state or {}` —— 空 dict 是 falsy，会丢失调用方的引用
    state = state if state is not None else {}
    soil = inputs.get("soil") or {}
    detections = inputs.get("detections") or {}
    weather = inputs.get("weather") or {}
    now = (weather.get("now") or {}).get("now") or (weather.get("now") or {})
    daily = (weather.get("daily") or {}).get("daily") or (weather.get("daily") or [])
    risk = inputs.get("risk") or {}

    # 当前执行器状态
    actuators = inputs.get("actuators") or {}
    valve_state = (actuators.get("drip_valve") or {}).get("state", "closed")
    boom_state = (actuators.get("spray_boom") or {}).get("state", "off")

    actions: list[dict] = []
    actions += _irrigate_decision(soil, daily, valve_state)
    actions += _spray_decision(detections, daily, now, boom_state,
                               state.setdefault("counters", {"spray_hits": 0, "spray_misses": 0}))

    # 决策类型与等级
    soil_v = _num(soil.get("value"))
    diseases = _diseases(detections)
    if any(a["device"] == "spray_boom" and a["op"] == "on" for a in actions):
        # 有达标病害 → 打药(high/medium)；无病害（高温低湿）→ 降温喷灌(medium)
        dtype, level = ("spray", "high" if diseases else "medium") if diseases else ("cool", "medium")
    elif any(a["device"] == "drip_valve" and a["op"] == "open" for a in actions):
        dtype, level = "irrigate", "medium"
    elif valve_state == "open":
        # 阀门已开、湿度未达停灌阈值/无大雨：持续灌溉中（无新动作）
        dtype, level = "irrigating", "medium"
    elif actions:
        dtype, level = "stop", "low"
    else:
        dtype, level = "noop", "low"

    # summary
    parts = []
    if soil:
        parts.append(f"土壤湿度 {soil_v:.0f}%")
    if valve_state == "open" and dtype == "irrigating":
        parts.append("滴灌持续进行中")
    if detections.get("summary"):
        s = detections["summary"]
        parts.append(f"检出 {s['total']} 处" + (f"（病害: {'/'.join(s['diseases'])}）" if s["diseases"] else "（无病害）"))
    summary = "；".join(parts) if parts else "暂无有效输入"

    return {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "decision_type": dtype,
        "summary": summary,
        "overall_level": level,
        "actions": actions,
    }
