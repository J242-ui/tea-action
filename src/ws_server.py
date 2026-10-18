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

from . import db
from .agent.runner import runner

BASE_DIR = Path(__file__).resolve().parent.parent
WEB_INDEX = BASE_DIR / "web" / "index.html"

SERVER_HOST = os.getenv("AGENT_WS_HOST", "127.0.0.1")
SERVER_PORT = int(os.getenv("AGENT_WS_PORT", "8510"))


def _json_body(data: dict, status: int = 200) -> bytes:
    return json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")


class TeaAgentASGI:
    """手写 ASGI 应用：HTTP REST + WebSocket 推送，无框架依赖。"""

    def __init__(self):
        self.clients: list = []
        self.loop: asyncio.AbstractEventLoop | None = None
        self._reaper_task: asyncio.Task | None = None
        self._runner_subscribed = False

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
                if not self._runner_subscribed:
                    runner.subscribe(self._broadcast)
                    self._runner_subscribed = True
                # 僵尸连接清理任务：无心跳超过 60 秒的 WS 客户端直接踢出，
                # 防止主站页面反复刷新/会话残留导致 clients 无限累积、广播变慢。
                # 必须保存任务引用，否则会被事件循环垃圾回收导致清理永不运行
                self._reaper_task = asyncio.create_task(self._reap_stale_clients())
                await send({"type": "lifespan.startup.complete"})
            elif msg["type"] == "lifespan.shutdown":
                if self._reaper_task is not None:
                    self._reaper_task.cancel()
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
            for ws in list(self.clients):
                try:
                    # 限时发送：慢/死连接不阻塞事件循环（取消后该连接视为失效移除）
                    await asyncio.wait_for(
                        ws["send"]({"type": "websocket.send", "text": payload}), timeout=0.5)
                    ws["last_seen"] = time.monotonic()
                except Exception:  # noqa: BLE001
                    try:
                        self.clients.remove(ws)
                    except ValueError:
                        pass

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
                body = WEB_INDEX.read_bytes()
            elif method == "GET" and path == "/api/health":
                body = _json_body({"ok": True, "service": "tea-agent-ws"})
            elif method == "GET" and path == "/api/status":
                body = _json_body({
                    "running": runner.running,
                    "steps": runner.steps,
                    "interval": runner.interval,
                    "refresh_mode": runner.refresh_mode,
                    "garden_key": runner.garden_key,
                    "latest": runner.latest,
                })
            elif method == "GET" and path == "/api/frame":
                # 最近一帧画面（持续录像帧或完整轮次帧），供主站实时画面拉取
                body = _json_body(runner.latest_frame)
            elif method == "GET" and path == "/api/history":
                body = _json_body({"items": db.agent_decision_history(runner.garden_key, limit=20)})
            elif method == "POST" and path == "/api/agent/start":
                # 支持可选 body: {"refresh_mode": "stream"|"interval", "interval": 秒}
                runner.configure(refresh_mode=payload.get("refresh_mode"),
                                 interval=payload.get("interval"))
                body = _json_body({"ok": runner.start(), "running": runner.running,
                                   "refresh_mode": runner.refresh_mode,
                                   "interval": runner.interval})
            elif method == "POST" and path == "/api/agent/config":
                # 运行时修改刷新模式与间隔（不改变运行/停止状态）
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
