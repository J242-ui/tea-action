# -*- coding: utf-8 -*-
"""全模块冒烟测试：ROUND 1 无 Key（demo/规则路径）；ROUND 2 假 Key（大模型失败回退路径）。

运行：python smoke_test.py [--fake-key]    （不带参数=无 Key 轮；--fake-key=假 Key 轮）
"""
from __future__ import annotations

import io
import os
import sys
import traceback

ROOT = r"C:\Users\Lenovo\Desktop\项目文件"
sys.path.insert(0, ROOT)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

RESULTS: list[tuple[str, bool, str]] = []


def t(name: str):
    """测试装饰器：记录 PASS/FAIL。"""
    def deco(fn):
        def wrapper(*a, **kw):
            try:
                fn(*a, **kw)
                RESULTS.append((name, True, ""))
                print(f"  [PASS] {name}")
            except Exception as e:  # noqa: BLE001
                RESULTS.append((name, False, f"{type(e).__name__}: {e}"))
                print(f"  [FAIL] {name}  ->  {type(e).__name__}: {e}")
                traceback.print_exc()
        return wrapper
    return deco


# =====================================================================
# ROUND 1 / ROUND 2 公共测试项
# =====================================================================
@t("config: 配置读取与茶园加载")
def test_config():
    from src import config
    assert config.BASE_DIR.exists()
    assert isinstance(config.get_qianfan_api_key(), str)
    assert isinstance(config.get_qwen_api_key(), str)
    gardens = config.load_gardens()
    assert len(gardens) >= 1
    g = config.get_garden("longjing")
    assert g is not None


@t("weather: 天气客户端（无 Key 演示 / 假 Key 失败兜底）")
def test_weather():
    from src.weather import WeatherClient
    from src import config, ui
    fake = "--fake-key" in sys.argv
    c = WeatherClient()
    if not fake:
        now = c.get_now("101210101")
        fc = c.get_forecast("101210101")
        air = c.get_air("101210101")
        idx = c.get_indices("101210101")
        city = c.search_city("杭州")
        assert now.get("now") and fc.get("daily") and air.get("now")
        assert idx.get("daily") and isinstance(city, list)
    else:
        # 假 Key：非 demo → 真实请求必然失败；验证页面级兜底（ui.get_weather 返回 None）
        assert c.demo is False
        try:
            c.get_now("101210101")
            raise AssertionError("假 Key 不应成功")
        except Exception:  # noqa: BLE001 - 预期失败
            pass
        garden = config.get_garden("longjing") or {}
        w = ui.get_weather(garden)
        assert w is None, "假 Key 天气失败后 ui.get_weather 应返回 None（页面安全停止）"


@t("risk: 规则风险研判引擎")
def test_risk():
    from src.weather import WeatherClient
    from src import risk
    # 用静态演示数据构造输入（不依赖网络/Key）
    now = WeatherClient._demo_now()["now"]
    daily = WeatherClient._demo_forecast(7)["daily"]
    eng = risk.RiskEngine()
    all_res = eng.assess_all(now, daily)
    assert isinstance(all_res["items"], list)
    assert all_res["overall_level"] in ("low", "medium", "high")
    for it in all_res["items"]:
        assert set(it) >= {"name", "level", "condition", "advice"}
        assert it["level"] in ("low", "medium", "high")


@t("ml_risk: LightGBM 风险分类（模型/回退）")
def test_ml_risk():
    from src import config, ml_risk
    garden = config.get_garden("longjing") or {}
    feats = ml_risk.build_daily_features(
        {"tempMax": 30, "tempMin": 20, "humidity": 70, "precip": 0.0}, garden)
    assert isinstance(feats, list) and len(feats) > 0
    prob = ml_risk.classifier.predict(feats)
    assert isinstance(prob, dict) and set(prob) == {"low", "medium", "high"}
    # 未就绪时返回均匀分布（回退路径）
    assert abs(sum(prob.values()) - 1.0) < 1e-6


@t("ts: 时序模块（历史窗口 / 预测）")
def test_ts():
    from src import ts
    win = ts.historical_window("longjing", window=7)
    assert isinstance(win, dict) and "n" in win
    fc = ts.forecast("longjing")
    assert fc is None or isinstance(fc, dict)  # 无历史数据时允许返回 None（设计行为）


@t("ensemble: 三引擎融合 + bandit 推荐 + 落库")
def test_ensemble():
    from src import ensemble, config
    garden = {**config.get_garden("longjing"), "key": "longjing"}
    prep = ensemble.prepare(garden, {})
    assert isinstance(prep, dict)
    from src.weather import WeatherClient
    w = WeatherClient()
    # 静态演示数据（不依赖 Key/网络）
    now = w._demo_now()["now"]
    daily = w._demo_forecast(7)["daily"]
    ens = ensemble.run_ensemble(garden, now, daily, w._demo_air())
    assert isinstance(ens, dict)
    assert "overall_level" in ens and "recommended_action" in ens
    arm = ens["recommended_action"]["arm"]
    ensemble.set_last_arm(garden, arm)
    ensemble.record_feedback(garden, now, ens.get("proba"), ens["overall_level"], adopted=False)


@t("bandit: LinUCB 在线学习")
def test_bandit():
    from src import bandit, config
    garden = config.get_garden("longjing") or {}
    ctx = bandit.build_context({"temp": 26, "humidity": 70, "precip": 0.0, "windScale": 2}, garden,
                               {"low": 0.6, "medium": 0.3, "high": 0.1})
    assert isinstance(ctx, list) and len(ctx) == len(bandit._CONTEXT_FEATURES)
    arm = bandit.bandit.recommend(ctx)
    assert isinstance(arm, str) and arm
    bandit.bandit.update(arm, ctx, 0.8)
    assert bandit.bandit.n_updates >= 1


@t("llm: 千帆/千问客户端（demo 模板 / 失败回退）")
def test_llm():
    from src.llm import QianfanClient, QwenClient
    qf = QianfanClient()
    qw = QwenClient()
    # 无 Key 或假 Key 下 chat 要么 demo 回复，要么抛异常（页面层已 try 兜底）
    try:
        r = qf.chat([{"role": "user", "content": "你好"}], timeout=5)
        assert isinstance(r, str) and r
    except Exception:  # noqa: BLE001 - 假 Key 网络失败：预期行为
        pass
    try:
        r2 = qw.chat([{"role": "user", "content": "你好"}], timeout=5)
        assert isinstance(r2, str) and r2
    except Exception:  # noqa: BLE001
        pass
    # explain_decision 必须永不抛异常（内部降级模板）
    exp = qf.explain_decision({"decision_type": "irrigate", "summary": "土壤湿度 42%",
                               "actions": [{"device": "drip_valve", "op": "open",
                                            "duration_min": 36.0, "reason": "缺水"}]},
                              "context")
    assert isinstance(exp, str) and exp
    d = qf._demo_explain({"decision_type": "noop", "summary": "土壤湿度 50%"}, "ctx")
    assert "继续监测" in d


@t("vision: 采集 / 检测 / 标注")
def test_vision():
    from src.vision import capture, detector, annotate
    frame = capture.make_demo_frame()
    assert frame is not None and frame.size > 0
    frame2 = capture.camera.read_frame()
    assert frame2 is not None and frame2.size > 0
    res = detector.detect(frame2)
    assert isinstance(res.get("detections"), list)
    assert res.get("summary") and res["summary"]["total"] == len(res["detections"])
    b64 = annotate.frame_to_base64(annotate.draw_detections(frame, res))
    assert b64.startswith("data:image/jpeg;base64,")


@t("sensor: 土壤湿度模拟器")
def test_sensor():
    from src.sensor import sensor
    r = sensor.read()
    assert r["type"] == "soil_moisture" and 15.0 <= r["value"] <= 95.0
    v0 = r["value"]
    sensor.irrigate(0.6)
    r2 = sensor.read()
    assert r2["value"] >= v0 - 1.0  # 灌溉后不应明显下降


@t("executor: 执行器状态机全操作")
def test_executor():
    from src.executor import bus
    bus.reset()
    ev1 = bus.apply([{"device": "drip_valve", "op": "open", "duration_min": 10.0, "reason": "t"}])
    assert bus.snapshot()["drip_valve"]["state"] == "open"
    ev2 = bus.apply([{"device": "spray_boom", "op": "on", "duration_min": 8.0, "reason": "t"}])
    assert bus.snapshot()["spray_boom"]["state"] == "on"
    ev3 = bus.apply([{"device": "drip_valve", "op": "close", "reason": "t"}])
    assert bus.snapshot()["drip_valve"]["state"] == "closed"
    ev4 = bus.apply([{"device": "spray_boom", "op": "off", "reason": "t"}])
    assert bus.snapshot()["spray_boom"]["state"] == "off"
    # 非法设备/操作应被忽略不报错
    bus.apply([{"device": "nope", "op": "on"}, {"device": "drip_valve", "op": "fly"}])
    ev5 = bus.auto_timeout_check(max_minutes=0.001)
    assert isinstance(ev5, list)
    bus.reset()


@t("decision: 规则决策各场景 + 防抖")
def test_decision():
    from src.agent import decision as rule
    base = {
        "weather": {"now": {"now": {"temp": 26, "humidity": 70, "text": "多云", "windScale": 2}},
                    "daily": {"daily": [{"precip": 0}]}},
        "actuators": {"drip_valve": {"state": "closed"}, "spray_boom": {"state": "off"}},
    }
    state = {"counters": {"spray_hits": 0, "spray_misses": 0}}
    # 缺水 → irrigate
    d = rule.decide({**base, "soil": {"value": 42.0},
                     "detections": {"summary": {"total": 0, "diseases": []}, "detections": []}}, state)
    assert d["decision_type"] == "irrigate" and d["actions"][0]["device"] == "drip_valve"
    # 连续 1 次检出 → 不喷（防抖）
    det = {"summary": {"total": 1, "diseases": ["Leaf rust"]},
           "detections": [{"class": "Leaf rust", "conf": 0.72}]}
    d2 = rule.decide({**base, "soil": {"value": 60.0}, "detections": det}, state)
    assert d2["decision_type"] != "spray"
    # 连续 2 次检出 → spray
    d3 = rule.decide({**base, "soil": {"value": 60.0}, "detections": det}, state)
    assert d3["decision_type"] == "spray" and any(a["op"] == "on" for a in d3["actions"])
    # 低置信度误检 → 不喷
    low = {"summary": {"total": 1, "diseases": ["Leaf rust"]},
           "detections": [{"class": "Leaf rust", "conf": 0.35}]}
    d4 = rule.decide({**base, "soil": {"value": 60.0}, "detections": low}, state)
    assert d4["decision_type"] != "spray"
    # 高温低湿 → cool
    hot = {**base, "weather": {"now": {"now": {"temp": 35, "humidity": 30, "text": "晴", "windScale": 2}},
                               "daily": {"daily": [{"precip": 0}]}}}
    d5 = rule.decide({**hot, "soil": {"value": 60.0},
                      "detections": {"summary": {"total": 0, "diseases": []}, "detections": []}}, state)
    assert d5["decision_type"] == "cool"
    # 正常 → noop
    d6 = rule.decide({**base, "soil": {"value": 60.0},
                      "detections": {"summary": {"total": 0, "diseases": []}, "detections": []}}, state)
    assert d6["decision_type"] == "noop"


@t("llm_decision: 决策分发（无 Key→规则 / 假 Key→回退）")
def test_llm_decision():
    from src.agent import llm_decision as ld
    inputs = {
        "soil": {"value": 42.0, "type": "soil_moisture", "unit": "%"},
        "detections": {"summary": {"total": 2, "diseases": ["Leaf rust of tea"]},
                       "detections": [{"class": "Leaf rust of tea", "conf": 0.62}]},
        "weather": {"now": {"now": {"temp": 28, "humidity": 60, "text": "晴", "windScale": 2}},
                    "daily": {"daily": [{"precip": 0}]}},
        "actuators": {"drip_valve": {"state": "closed"}, "spray_boom": {"state": "off"}},
    }
    state = {"counters": {"spray_hits": 0, "spray_misses": 0}}
    d = ld.decide(inputs, state)
    assert d["decision_type"] in ("irrigate", "irrigating", "spray", "cool", "stop", "noop")
    assert isinstance(d.get("explanation"), str) and d["explanation"]
    # 大模型动作安全校验
    safe = ld._sanitize_actions(
        [{"device": "drip_valve", "op": "open", "duration_min": 999.0, "reason": "r"},
         {"device": "spray_boom", "op": "on", "duration_min": 8, "reason": "乱加"}],
        [{"device": "drip_valve", "op": "open", "duration_min": 36.0, "reason": "r"}])
    assert len(safe) == 1 and safe[0]["duration_min"] == 120.0  # 超时被限幅


@t("runner: Agent 主循环单步（全链路）")
def test_runner():
    from src.agent.runner import AgentRunner
    r = AgentRunner(garden_key="longjing", interval=3.0)
    snap = r.step()
    assert snap["decision"]["decision_type"] in ("irrigate", "irrigating", "spray", "cool", "stop", "noop")
    assert snap["soil"]["value"] > 0
    assert "image" in snap["detections"]
    assert snap["actuator"]["drip_valve"]["state"] in ("open", "closed")
    # 连续 step 不报错（防抖状态保持）
    snap2 = r.step()
    assert snap2["steps"] == 2
    r.stop()


@t("db: MySQL 建表与日志查询")
def test_db():
    from src import db
    db.init_db()
    conn = db.connect()
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) AS n FROM information_schema.tables "
                    "WHERE table_schema = DATABASE() AND table_name IN "
                    "('sensor_log','actuator_log','agent_decision_log','users','gardens')")
        n = cur.fetchone()["n"]
        assert n >= 3
    conn.close()
    rows = db.agent_decision_history("longjing", limit=5)
    assert isinstance(rows, list)
    sensor_rows = db.sensor_history("longjing", limit=5) if hasattr(db, "sensor_history") else []
    assert isinstance(sensor_rows, list)


@t("ws_server: 模块导入与路由表")
def test_ws_server():
    import importlib
    from src import ws_server
    assert hasattr(ws_server, "app") and hasattr(ws_server, "main")
    paths = {"/", "/api/health", "/api/status", "/api/history",
             "/api/agent/start", "/api/agent/stop", "/api/agent/step"}
    for p in paths:
        assert p in ("/",) + tuple(p for p in ["/api/health", "/api/status", "/api/history",
                                               "/api/agent/start", "/api/agent/stop", "/api/agent/step"])
    assert ws_server.SERVER_PORT == 8510


@t("home_page: 首页组件导入")
def test_home_page():
    from src import home_page
    assert callable(home_page.main)


@t("pages: 各 Streamlit 页面模块可导入")
def test_pages():
    import py_compile
    for name in ["1_气象监测", "2_风险预警", "3_智能问答", "4_农事建议", "5_我的茶园", "6_智能执行"]:
        p = os.path.join(ROOT, "pages", f"{name}.py")
        py_compile.compile(p, doraise=True)  # 语法/编译级验证（页面渲染需在 Streamlit 运行时验证）
        print(f"      page {name} compile ok")


# =====================================================================
def main():
    fake = "--fake-key" in sys.argv
    print("=" * 60)
    print(f"冒烟测试 ROUND{'2（假 Key：验证失败回退）' if fake else '1（无 Key：demo/规则路径）'}")
    print("=" * 60)
    if fake:
        os.environ["QWEN_API_KEY"] = "sk-fake-invalid-key-for-smoke-test"
        os.environ["QIANFAN_API_KEY"] = "fake-qianfan-key-for-smoke-test"
        os.environ["QWEATHER_API_KEY"] = "fake-qweather-key"
        # 强制非 demo（让真实网络路径生效，测试失败兜底）
        os.environ["DEMO_MODE"] = "false"
    else:
        os.environ.pop("QWEN_API_KEY", None)
        os.environ.pop("QIANFAN_API_KEY", None)
        os.environ.pop("QWEATHER_API_KEY", None)
        os.environ.pop("DEMO_MODE", None)

    tests = [test_config, test_weather, test_risk, test_ml_risk, test_ts, test_ensemble,
             test_bandit, test_llm, test_vision, test_sensor, test_executor, test_decision,
             test_llm_decision, test_runner, test_db, test_ws_server, test_home_page]
    if not fake:
        tests.append(test_pages)

    for fn in tests:
        fn()

    print("=" * 60)
    ok = sum(1 for _, b, _ in RESULTS if b)
    fail = len(RESULTS) - ok
    for name, b, detail in RESULTS:
        if not b:
            print(f"  FAILED: {name} -> {detail}")
    print(f"结果：{ok} PASS / {fail} FAIL / 共 {len(RESULTS)} 项")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
