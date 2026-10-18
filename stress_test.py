"""质量保障：账号并发/WS 并发/降级/延迟压力冒烟（临时脚本，不入库）。"""
import json
import sys
import threading
import time
import urllib.request

BASE = "http://127.0.0.1:8510"
results = []
def _check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")

def _get(path, timeout=15):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))

def _post(path, body, timeout=30):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))

# 1) 启动 Agent（admin 账号）
st = _post("/api/agent/start", {"user_id": 2, "refresh_mode": "interval", "interval": 5})
_check("start_admin", st.get("ok") is True and st.get("user_id") == 2, str(st))
time.sleep(3)

# 2) 账号视图并发切换（模拟监控台快速切换用户，同时 Agent 决策线程在跑）
errs = []
def _switch():
    try:
        for uid in ("0", "2", "0", "2"):
            _get(f"/api/cameras?user_id={uid}")
    except Exception as e:
        errs.append(str(e))
threads = [threading.Thread(target=_switch) for _ in range(8)]
for t in threads: t.start()
for t in threads: t.join()
_check("账号视图并发切换(8线程x4次)", not errs, "errs=" + ";".join(errs[:3]))

# 3) 并发 GET /api/history /api/status（事件循环不被 DB 阻塞）
errs2 = []
def _read_api():
    try:
        for _ in range(6):
            _get("/api/status", timeout=5)
            _get("/api/history?user_id=2", timeout=5)
    except Exception as e:
        errs2.append(str(e))
threads = [threading.Thread(target=_read_api) for _ in range(6)]
t0 = time.monotonic()
for t in threads: t.start()
for t in threads: t.join()
elapsed = time.monotonic() - t0
_check("并发REST读接口(6线程x12次)", not errs2 and elapsed < 20, f"elapsed={elapsed:.1f}s errs={len(errs2)}")

# 4) 不可达网络源添加/删除（降级不卡）
t0 = time.monotonic()
add = _post("/api/cameras/add", {"user_id": 2, "source": "http://127.0.0.1:9/demo", "name": "不可达源"})
t_add = time.monotonic() - t0
_check("添加不可达网络源不卡", add.get("ok") is True and t_add < 10, f"t={t_add:.1f}s")
cams = _get("/api/cameras?user_id=2")
_check("含不可达源列表正常", cams.get("ok") is True and len(cams.get("cameras", [])) >= 1)
rm = _post("/api/cameras/remove", {"user_id": 2, "cam_id": add.get("camera", {}).get("id", "")})
_check("删除不可达源", rm.get("ok") is True)

# 5) 多 WS 客户端并发接收广播
import socket
def _ws_conn(n, out):
    try:
        s = socket.create_connection(("127.0.0.1", 8510), timeout=10)
        s.sendall(b"GET /ws HTTP/1.1\r\nHost: 127.0.0.1:8510\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Version: 13\r\n\r\n")
        got = 0
        s.settimeout(8)
        buf = b""
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            try:
                buf += s.recv(4096)
                got += buf.count(b"state_update")
                buf = b""
            except socket.timeout:
                break
        out[n] = got
        s.close()
    except Exception as e:
        out[n] = f"ERR:{e}"
outs = {}
threads = [threading.Thread(target=_ws_conn, args=(i, outs)) for i in range(8)]
for t in threads: t.start()
for t in threads: t.join()
ok_ws = all(isinstance(outs.get(i), int) and outs[i] > 0 for i in range(8))
_check("8个WS客户端并发接收广播", ok_ws, str(outs))

# 6) 决策连续运行步数递增
steps_before = _get("/api/status").get("steps", 0)
time.sleep(6)
steps_after = _get("/api/status").get("steps", 0)
_check("Agent决策持续递增", steps_after > steps_before, f"{steps_before}->{steps_after}")

# 7) 反复 stop/start 三次（线程干净退出/重启）
ok_cycle = True
for i in range(3):
    _post("/api/agent/stop", {})
    time.sleep(0.5)
    r = _post("/api/agent/start", {"user_id": 2, "interval": 5})
    if not r.get("ok"):
        ok_cycle = False
        break
    time.sleep(1)
_check("反复stop/start x3", ok_cycle)

# 8) 响应延迟抽样（健康检查应 < 200ms）
lats = []
for _ in range(20):
    t0 = time.monotonic()
    _get("/api/health", timeout=3)
    lats.append((time.monotonic() - t0) * 1000)
_check("health延迟中位数<100ms", sorted(lats)[10] < 100, f"med={sorted(lats)[10]:.0f}ms max={max(lats):.0f}ms")

_post("/api/agent/stop", {})
print("=" * 40)
print(f"结果：{sum(1 for _, ok, _ in results if ok)} PASS / {sum(1 for _, ok, _ in results if not ok)} FAIL")
sys.exit(0 if all(ok for _, ok, _ in results) else 1)
