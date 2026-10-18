"""茶园智能执行 WebSocket 状态服务（零新依赖：uvicorn + websockets 手写 ASGI）。

职责（演示链路的后半段）：
  Agent 决策 → 执行器状态 → WebSocket 推送 → 网页动态展示

- REST（http 同端口）：
  GET  /                网页（web/index.html）
  GET  /api/status      服务与 Agent 状态、最近状态快照
  POST /api/agent/start 启动 Agent 主循环
  POST /api/agent/stop  停止 Agent
  POST /api/agent/step  手动触发一轮决策
  GET  /api/history     决策历史（agent_decision_log）
  GET  /api/health      健康检查
- WS  /ws               实时推送 state_update 事件

运行：python -m src.ws_server   （默认 127.0.0.1:8510）
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from urllib.parse import unquote

from . import db
from .agent.daily_alert import DailyAlertScheduler
from .agent.runner import runner
from .vision.capture import camera_manager

BASE_DIR = Path(__file__).resolve().parent.parent
WEB_INDEX = BASE_DIR / "web" / "index.html"

SERVER_HOST = os.getenv("AGENT_WS_HOST", "127.0.0.1")
SERVER_PORT = int(os.getenv("AGENT_WS_PORT", "8510"))

# 首页文件内容缓存：避免每次 GET / 都重新读盘（约 33KB）
try:
    WEB_INDEX_BYTES = WEB_INDEX.read_bytes()
except Exception:  # noqa: BLE001 - 文件缺失时回退空页，接口不受影响
    WEB_INDEX_BYTES = b"<!doctype html><meta charset=utf-8><title>tea-agent-ws</title>"


def _json_body(data: dict, status: int = 200) -> bytes:
    return json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")


def _query_params(scope) -> dict:
    """解析 URL 查询参数（GET 请求的 user_id 等），值做 URL 解码。"""
    qs = (scope.get("query_string") or b"").decode("utf-8")
    out: dict = {}
    for kv in qs.split("&"):
        if not kv:
            continue
        k, _, v = kv.partition("=")
        if k:
            out[unquote(k)] = unquote(v)
    return out


def _uid(v) -> int:
    """把 user_id 参数安全转 int（游客默认 0）。"""
    try:
        return int(str(v or "").strip() or 0)
    except Exception:  # noqa: BLE001
        return 0


class TeaAgentASGI:
    """手写 ASGI 应用：HTTP REST + WebSocket 推送，无框架依赖。"""

    def __init__(self):
        self.clients: list = []
        self.loop: asyncio.AbstractEventLoop | None = None
        self._reaper_task: asyncio.Task | None = None
        self._runner_subscribed = False
        self.alert_scheduler = DailyAlertScheduler(runner.garden_key)

    # ---------- ASGI 入口 ----------
    async def __call__(self, scope, receive, send):
        if scope["type"] == "lifespan":
            await self._lifespan(receive, send)
        elif scope["type"] == "http":
            await self._http(scope, receive, send)
        elif scope["type"] == "websocket":
            await self._websocket(scope, receive, send)

    async def _lifespan(self, receive, send):
        while True:
            msg = await receive()
            if msg["type"] == "lifespan.startup":
                self.loop = asyncio.get_running_loop()
                # 确保扩展表（camera_alert_log 等）存在：8510 独立运行时也要建表
                try:
                    db.init_db_ext()
                except Exception:  # noqa: BLE001 - 数据库不可用时降级，后续写库也会降级
                    pass
                if not self._runner_subscribed:
                    runner.subscribe(self._broadcast)
                    self._runner_subscribed = True
                # 每日摄像头监测预警调度器：到点生成一次并广播 kind=alert_push
                self.alert_scheduler.subscribe(self._broadcast)
                self.alert_scheduler.start()
                # 僵尸连接清理任务：无心跳超过 60 秒的 WS 客户端直接踢出，
                # 防止主站页面反复刷新/会话残留导致 clients 无限累积、广播变慢。
                # 必须保存任务引用，否则会被事件循环垃圾回收导致清理永不运行
                self._reaper_task = asyncio.create_task(self._reap_stale_clients())
                await send({"type": "lifespan.startup.complete"})
            elif msg["type"] == "lifespan.shutdown":
                if self._reaper_task is not None:
                    self._reaper_task.cancel()
                self.alert_scheduler.stop()
                runner.stop()
                await send({"type": "lifespan.shutdown.complete"})
                return

    async def _reap_stale_clients(self) -> None:
        """周期扫描：剔除 last_seen 超过 60 秒的僵尸 WS 连接。"""
        while True:
            await asyncio.sleep(15)
            now = time.monotonic()
            for ws in list(self.clients):
                if now - ws.get("last_seen", now) > 60:
                    try:
                        await ws["send"]({"type": "websocket.close", "code": 4000})
                    except Exception:  # noqa: BLE001
                        pass
                    try:
                        self.clients.remove(ws)
                    except ValueError:
                        pass

    # ---------- 跨线程广播 ----------
    def _broadcast(self, event: dict) -> None:
        """runner 后台线程 → uvicorn 事件循环 → 所有 WS 客户端。"""
        if self.loop is None or not self.clients:
            return
        payload = json.dumps(event, ensure_ascii=False, default=str)

        async def _send_all():
            # 并发发送：每个客户端独立限时，慢/死连接只拖累自己，
            # 不阻塞其他客户端的消息投递（旧的串行方案会被慢客户端拖住整批广播）。
            async def _one(ws):
                try:
                    await asyncio.wait_for(
                        ws["send"]({"type": "websocket.send", "text": payload}), timeout=0.5)
                    ws["last_seen"] = time.monotonic()
                except Exception:  # noqa: BLE001
                    try:
                        self.clients.remove(ws)
                    except ValueError:
                        pass

            if self.clients:
                await asyncio.gather(*[_one(ws) for ws in list(self.clients)],
                                     return_exceptions=True)

        try:
            asyncio.run_coroutine_threadsafe(_send_all(), self.loop)
        except Exception:  # noqa: BLE001
            pass

    # ---------- HTTP ----------
    async def _http(self, scope, receive, send):
        path = scope["path"]
        method = scope.get("method", "GET")
        status = 200
        ctype = "application/json; charset=utf-8"
        body = b'{"ok":false,"message":"not found"}'

        # POST 请求读取 JSON body（供 start/config 传入刷新参数）
        payload: dict = {}
        if method == "POST":
            body_raw = b""
            while True:
                msg = await receive()
                if msg["type"] == "http.request":
                    body_raw += msg.get("body", b"")
                    if not msg.get("more_body"):
                        break
            if body_raw:
                try:
                    parsed = json.loads(body_raw.decode("utf-8"))
                    payload = parsed if isinstance(parsed, dict) else {}
                except Exception:  # noqa: BLE001 - body 非 JSON 时按空参数处理
                    payload = {}

        try:
            if method == "GET" and path in ("/", "/index.html"):
                ctype = "text/html; charset=utf-8"
                body = WEB_INDEX_BYTES
            elif method == "GET" and path == "/api/health":
                body = _json_body({"ok": True, "service": "tea-agent-ws"})
            elif method == "GET" and path == "/api/status":
                body = _json_body({
                    "running": runner.running,
                    "steps": runner.steps,
                    "interval": runner.interval,
                    "refresh_mode": runner.refresh_mode,
                    "garden_key": runner.garden_key,
                    "user_id": runner.user_id,
                    "latest": runner.latest,
                })
            elif method == "GET" and path == "/api/frame":
                # 最近一帧画面（持续录像帧或完整轮次帧），供主站实时画面拉取
                body = _json_body(runner.latest_frame)
            elif method == "GET" and path == "/api/history":
                # 决策历史按账号过滤（query ?user_id=，默认游客 0）
                uid = _uid(_query_params(scope).get("user_id"))
                items = await asyncio.to_thread(
                    db.agent_decision_history, runner.garden_key, 20, uid)
                body = _json_body({"items": items})
            elif method == "GET" and path == "/api/alerts/latest":
                # 最近一条摄像头监测预警（内存优先，服务重启后回退查库；按账号过滤）
                uid = _uid(_query_params(scope).get("user_id"))
                latest = self.alert_scheduler.latest()
                if latest is None or (uid and latest.get("data", {}).get("user_id") not in (None, uid)):
                    latest = await asyncio.to_thread(db.latest_camera_alert, runner.garden_key, uid)
                body = _json_body({"ok": True, "alert": latest})
            elif method == "GET" and path == "/api/alerts/history":
                uid = _uid(_query_params(scope).get("user_id"))
                items = await asyncio.to_thread(db.camera_alert_history, runner.garden_key, 30, uid)
                body = _json_body({"ok": True, "items": items})
            elif method == "POST" and path == "/api/agent/start":
                # 支持 body: {"user_id": 账号(0=游客), "garden_key": 茶园, "refresh_mode": ..., "interval": ...}
                uid = _uid(payload.get("user_id"))
                if runner.running:
                    body = _json_body({"ok": False,
                                       "message": "Agent 正在运行（归属账号 user_id=%s），请先停止" % runner.user_id,
                                       **runner.context()}, 409)
                else:
                    # set_context 内含数据库查询（取账号茶园），camera_manager.set_user
                    # 内含数据库查询（取账号摄像头源）——都放到线程池，避免阻塞事件循环
                    ctx = await asyncio.to_thread(runner.set_context, uid, payload.get("garden_key"))
                    if not ctx.get("ok"):
                        body = _json_body({"ok": False, **ctx}, 400)
                    else:
                        # 摄像头源视图切换到该账号（游客=默认源；登录=其配置源）
                        try:
                            await asyncio.to_thread(camera_manager.set_user, uid)
                        except Exception:  # noqa: BLE001 - 摄像头层异常不影响 Agent 启动
                            pass
                        runner.configure(refresh_mode=payload.get("refresh_mode"),
                                         interval=payload.get("interval"))
                        started = runner.start()
                        body = _json_body({"ok": started, "running": runner.running,
                                           "user_id": runner.user_id,
                                           "garden_key": runner.garden_key,
                                           "refresh_mode": runner.refresh_mode,
                                           "interval": runner.interval})
            elif method == "POST" and path == "/api/agent/config":
                # 运行时修改刷新模式与间隔（不改变运行/停止状态；user_id 校验不越权切换账号）
                uid = _uid(payload.get("user_id"))
                if uid and runner.user_id and uid != runner.user_id and runner.running:
                    body = _json_body({"ok": False, "message": "Agent 当前归属其他账号，不能修改"}, 403)
                else:
                    cfg = runner.configure(refresh_mode=payload.get("refresh_mode"),
                                           interval=payload.get("interval"))
                    body = _json_body({"ok": True, **cfg})
            elif method == "POST" and path == "/api/agent/stop":
                body = _json_body({"ok": runner.stop(), "running": runner.running})
            elif method == "POST" and path == "/api/agent/step":
                # 整轮决策（摄像头/推理/写库）可能耗时数秒，必须在线程池执行，
                # 否则会阻塞 uvicorn 事件循环，导致网页/WS/其他接口全部卡住
                snapshot = await asyncio.to_thread(runner.step)
                body = _json_body({"ok": True, "snapshot": snapshot})
            elif method == "GET" and path == "/api/cameras":
                # 摄像头源列表（按账号：?user_id=，游客 0 显示默认源）
                uid = _uid(_query_params(scope).get("user_id"))
                try:
                    await asyncio.to_thread(camera_manager.set_user, uid)
                except Exception:  # noqa: BLE001
                    pass
                cams = await asyncio.to_thread(camera_manager.list)
                body = _json_body({"ok": True, "user_id": uid, "cameras": cams})
            elif method == "POST" and path == "/api/cameras/add":
                # 添加摄像头源：{"user_id": 账号, "source": "usb:1" / "rtsp://…" / "http://…", "name": "可选"}
                uid = _uid(payload.get("user_id"))
                try:
                    await asyncio.to_thread(camera_manager.set_user, uid, True)
                except Exception:  # noqa: BLE001
                    pass
                cam = await asyncio.to_thread(camera_manager.add,
                                              payload.get("source", ""), payload.get("name", ""), uid)
                if cam is None:
                    body = _json_body({"ok": False, "message": "无效的摄像头源，支持 usb:N / rtsp:// / http(s)://"}, 400)
                else:
                    # 重载视图使内存与数据库一致：有配置只显示配置源，无配置回退默认源
                    try:
                        await asyncio.to_thread(camera_manager.set_user, uid, True)
                    except Exception:  # noqa: BLE001
                        pass
                    body = _json_body({"ok": True, "camera": cam})
            elif method == "POST" and path == "/api/cameras/remove":
                # 删除摄像头源：{"user_id": 账号, "cam_id": "cam2"}
                uid = _uid(payload.get("user_id"))
                try:
                    await asyncio.to_thread(camera_manager.set_user, uid, True)
                except Exception:  # noqa: BLE001
                    pass
                removed = await asyncio.to_thread(camera_manager.remove, payload.get("cam_id", ""), uid)
                if not removed:
                    body = _json_body({"ok": False, "message": "摄像头不存在"}, 404)
                else:
                    try:
                        await asyncio.to_thread(camera_manager.set_user, uid, True)
                    except Exception:  # noqa: BLE001
                        pass
                    body = _json_body({"ok": True})
            else:
                status = 404
                body = _json_body({"ok": False, "message": "not found"}, 404)
        except Exception as e:  # noqa: BLE001
            status = 500
            body = _json_body({"ok": False, "message": str(e)}, 500)

        await send({"type": "http.response.start",
                    "status": status,
                    "headers": [(b"content-type", ctype.encode()),
                                (b"content-length", str(len(body)).encode()),
                                (b"cache-control", b"no-store")]})
        await send({"type": "http.response.body", "body": body})

    # ---------- WebSocket ----------
    async def _websocket(self, scope, receive, send):
        await send({"type": "websocket.accept"})
        conn = {"send": send, "last_seen": time.monotonic()}
        self.clients.append(conn)
        try:
            if runner.latest:
                await send({"type": "websocket.send",
                            "text": json.dumps({"kind": "state_update", "data": runner.latest},
                                               ensure_ascii=False, default=str)})
            while True:
                msg = await receive()
                if msg["type"] == "websocket.disconnect":
                    break
                # 客户端心跳/消息：更新存活时间，供僵尸连接清理任务判定
                conn["last_seen"] = time.monotonic()
        except Exception:  # noqa: BLE001
            pass
        finally:
            try:
                self.clients.remove(conn)
            except ValueError:
                pass


app = TeaAgentASGI()


def main() -> None:
    import uvicorn
    print("=" * 56)
    print("  茶园智能执行服务（Tea Garden Agent WS）")
    print(f"  动态展示网页 : http://{SERVER_HOST}:{SERVER_PORT}/")
    print(f"  WebSocket   : ws://{SERVER_HOST}:{SERVER_PORT}/ws")
    print(f"  状态API     : http://{SERVER_HOST}:{SERVER_PORT}/api/status")
    print("=" * 56)
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT, log_level="info")


if __name__ == "__main__":
    main()
