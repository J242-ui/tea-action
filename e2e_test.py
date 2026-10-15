# -*- coding: utf-8 -*-
"""最终端到端验证：启动 → 连续 step 8 轮 → 检查 irrigating/决策解释/历史/停止。"""
import io, json, sys, time, urllib.request
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
B = "http://127.0.0.1:8510"


def req(path, method="GET"):
    r = urllib.request.Request(B + path, method=method)
    return json.loads(urllib.request.urlopen(r, timeout=15).read().decode("utf-8"))


ok = fail = 0
def check(name, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  [PASS] {name}")
    else:
        fail += 1
        print(f"  [FAIL] {name} -> {detail}")


assert req("/api/health").get("ok")
req("/api/agent/start", "POST")
types_seen = set()
for i in range(8):
    snap = req("/api/agent/step", "POST").get("snapshot", {})
    dec = snap.get("decision", {})
    types_seen.add(dec.get("decision_type"))
    assert dec.get("decision_type") in ("irrigate", "irrigating", "spray", "cool", "stop", "noop"), dec
    assert isinstance(dec.get("explanation"), str) and dec["explanation"], dec
    time.sleep(1.5)
d = req("/api/status")
lat = d.get("latest") or {}
dec = lat.get("decision", {})
check("连续 8 轮决策无异常", len(types_seen) >= 1, f"types={types_seen}")
check("latest 决策含解释", dec.get("explanation") and len(dec["explanation"]) > 5, str(dec)[:150])
valve = lat.get("actuator", {}).get("drip_valve", {}).get("state", "closed")
soil_v = lat.get("soil", {}).get("value")
check("执行器/传感器数据存在", valve in ("open", "closed") and isinstance(soil_v, (int, float)), f"valve={valve} soil={soil_v}")
h = req("/api/history")
check("历史记录", isinstance(h.get("items"), list) and len(h["items"]) >= 1, f"n={len(h.get('items', []))}")
req("/api/agent/stop", "POST")
check("停止", req("/api/status").get("running") is False, "running 未停")
print(f"seen types: {sorted(types_seen)}")
print("=" * 56)
print(f"端到端结果：{ok} PASS / {fail} FAIL")
sys.exit(1 if fail else 0)
