"""Agent 决策分发：无 Key 时走规则引擎，有 Key 时由大模型（通义千问）复核判断并生成解释。

设计：
  1. 规则引擎（decision.py）始终先执行：维护防抖计数、阈值与天气条件判断，
     产出"候选决策"。演示（无 QWEN_API_KEY）直接采用，完全按既有防抖规则运行。
  2. 配置 QWEN_API_KEY 时：把本轮事实 + 规则候选交给通义千问（QwenClient，
     OpenAI 兼容接口），由大模型复核判断（可同意/调整时长/改为不作业），
     并输出人类可读解释。大模型输出受安全校验约束：不得新增规则引擎未建议的
     设备动作，防抖/阈值防线仍由规则引擎把关，防止误检触发误操作。
  3. 大模型调用失败/超时/输出不合法 → 自动回退规则候选 + 模板解释。
  4. 注：气象灾害研判、病虫害分析与农事问答走伏羲大模型（QianfanClient），
     与本模块（Agent 决策）互不影响。
"""
from __future__ import annotations

import json
from datetime import datetime

from . import decision as rule
from ..llm import QwenClient

VALID_TYPES = {"irrigate", "irrigating", "spray", "cool", "stop", "noop"}
DEVICE_OPS = {"drip_valve": {"open", "close"}, "spray_boom": {"on", "off"}}

SYSTEM_JUDGE = (
    "你是茶园智能执行系统的决策判断中枢。系统已根据土壤湿度、视觉病害检测与天气数据，"
    "由规则引擎（防抖/阈值）生成了本轮候选作业建议。"
    "规则引擎的底层口径（供你复核时理解候选为何如此）："
    "①浇水：土壤湿度低于 45% 且未来无大雨时开滴灌，回升到 68% 或出现降水即停；"
    "②打药：连续 2 轮检出病害且置信度 ≥0.50、非雨天、风力<4级才喷药，连续 3 轮无病害即停；"
    "③降温：气温 ≥33℃ 且空气湿度 <40% 时短时喷灌约 5 分钟；"
    "④防抖计数、降雨与风力防线已由系统跨轮维护，你无需自行推算。"
    "你的任务："
    "1. 复核候选建议是否合理：可以同意、可以调整作业时长、也可以选择本轮不作业；"
    "2. 只能从候选建议包含的设备动作中选择或放弃，不得新增候选建议之外的设备动作；"
    "3. 严格输出 JSON（不要任何其他文字）："
    '{"decision_type":"irrigate|irrigating|spray|cool|stop|noop",'
    '"actions":[{"device":"drip_valve|spray_boom","op":"open|close|on|off",'
    '"duration_min":数字或null,"reason":"中文理由"}],'
    '"explanation":"给茶园管理者的一句话解释（不超过 80 字）"}'
)


def _ctx(inputs: dict, cand: dict) -> str:
    soil = inputs.get("soil") or {}
    det = inputs.get("detections") or {}
    det_sum = det.get("summary") or {}
    weather = inputs.get("weather") or {}
    now = (weather.get("now") or {}).get("now") or (weather.get("now") or {})
    daily = (weather.get("daily") or {}).get("daily") or (weather.get("daily") or [])
    prec = [d.get("precip") for d in daily[:3]]
    act = inputs.get("actuators") or {}
    valve = (act.get("drip_valve") or {}).get("state", "closed")
    boom = (act.get("spray_boom") or {}).get("state", "off")
    diseases = "、".join(det_sum.get("diseases", [])) or "无"
    return (
        f"土壤湿度 {soil.get('value') if soil else '无'}%；"
        f"检出 {det_sum.get('total', 0)} 处（{diseases}）；"
        f"气温 {now.get('temp') or '无'}℃，湿度 {now.get('humidity') or '无'}%，"
        f"天气 {now.get('text') or '无'}，未来3日降水 {prec}，风力 {now.get('windScale') or '无'}级；"
        f"执行器：滴灌阀 {valve}，喷杆 {boom}"
    )


def _rebuild(actions: list[dict], has_disease: bool = False,
             valve_state: str = "closed") -> tuple[str, str]:
    """根据动作列表重算决策类型与等级（与规则引擎口径一致）。"""
    if any(a["device"] == "spray_boom" and a["op"] == "on" for a in actions):
        # 有达标病害 → 打药；无病害（高温低湿）→ 降温喷灌
        return ("spray", "high") if has_disease else ("cool", "medium")
    if any(a["device"] == "drip_valve" and a["op"] == "open" for a in actions):
        return "irrigate", "medium"
    if not actions and valve_state == "open":
        return "irrigating", "medium"  # 阀门保持开启、持续灌溉中
    if actions:
        return "stop", "low"
    return "noop", "low"


def _sanitize_actions(llm_actions: list | None, cand_actions: list[dict]) -> list[dict]:
    """校验大模型动作：仅允许规则候选已有的 device+op 组合，修正时长范围。"""
    allowed = {(a["device"], a["op"]) for a in cand_actions}
    out: list[dict] = []
    for a in llm_actions or []:
        if not isinstance(a, dict):
            continue
        dev = a.get("device")
        op = a.get("op")
        if (dev, op) not in allowed:
            continue  # 大模型不得发明候选之外的动作
        reason = str(a.get("reason") or "")[:120]
        dur = a.get("duration_min")
        try:
            d = float(dur) if dur is not None else None
        except (TypeError, ValueError):
            d = None
        if d is not None:
            d = max(1.0, min(120.0, d))
        out.append({"device": dev, "op": op, "duration_min": d, "reason": reason})
    return out


def _llm_judge(client: QwenClient, inputs: dict, cand: dict) -> dict | None:
    """调用大模型复核判断；任何失败/非法输出返回 None（回退规则候选）。"""
    ctx = _ctx(inputs, cand)
    user_msg = (
        f"【本轮事实】{ctx}\n"
        f"【规则引擎候选】{json.dumps(cand, ensure_ascii=False)}\n"
        "请复核并输出最终决策 JSON。"
    )
    try:
        text = client.chat(
            [{"role": "system", "content": SYSTEM_JUDGE},
             {"role": "user", "content": user_msg}],
            temperature=0.2, max_tokens=600, timeout=10,
        )
    except Exception:  # noqa: BLE001
        return None
    # 提取 JSON（容忍模型输出带 ```json 围栏或前后杂字）
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start:end + 1])
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None

    cand_actions = cand.get("actions") or []
    actions = _sanitize_actions(obj.get("actions"), cand_actions)
    has_disease = "病害:" in (cand.get("summary") or "")
    valve_state = (inputs.get("actuators") or {}).get("drip_valve", {}).get("state", "closed")
    dtype, level = _rebuild(actions, has_disease, valve_state)
    explanation = str(obj.get("explanation") or "").strip()[:200]
    if not explanation:
        explanation = client._demo_explain(cand, ctx)

    return {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "decision_type": dtype,
        "summary": cand.get("summary", ""),
        "overall_level": level,
        "actions": actions,
        "explanation": explanation,
    }


def decide(inputs: dict, state: dict | None = None) -> dict:
    """决策入口：规则引擎先算（防抖/阈值），配置千问 Key 时由大模型复核判断。"""
    cand = rule.decide(inputs, state)          # 规则候选（含防抖计数更新）
    client = QwenClient()
    if client.demo:                            # 无 QWEN_API_KEY：完全按规则判断
        cand["explanation"] = client._demo_explain(cand, _ctx(inputs, cand))
        return cand
    llm_dec = _llm_judge(client, inputs, cand)  # 有 Key：千问大模型判断
    if llm_dec is not None:
        return llm_dec
    cand["explanation"] = client._demo_explain(cand, _ctx(inputs, cand))
    return cand
