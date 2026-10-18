"""摄像头采集与演示帧生成。

CameraCapture：
  - 支持多类画面源（source）：
      usb:N        USB 摄像头（index 可配置，默认后端 MSMF→DSHOW→ANY 探测）
      rtsp://...   网络摄像头 RTSP 流（海康/大华/小米等，需 FFMPEG 后端）
      http(s)://... MJPEG 流 / 快照（手机 IP Webcam 等）
  - 任何源不可用/读到黑帧时，自动降级返回 make_demo_frame() 合成帧

CameraManager：
  多摄像头源管理器（监控台）：默认注册一个 usb:0 主源，
  支持运行时 add / remove / list，每个源独立采集与检测。

make_demo_frame：
  用纯 numpy 绘制「茶园叶片」演示帧（若干叶片 + 病斑），
  与 SimDetector 的检测坐标一一对应，保证无模型时演示链路依然完整。
"""
from __future__ import annotations

import os
import re
import threading
import time

import numpy as np

# 演示帧里的"叶片"布局（中心坐标），与 SimDetector 检测框共用，保证标注对齐
DEMO_LEAVES = [
    {"cx": 170, "cy": 150, "rx": 66, "ry": 40, "angle": -18, "lesion": "brown"},   # -> Brown blight
    {"cx": 390, "cy": 120, "rx": 60, "ry": 36, "angle": 12,  "lesion": "rust"},    # -> Leaf rust
    {"cx": 535, "cy": 300, "rx": 70, "ry": 42, "angle": -6,  "lesion": "none"},    # -> Healthy
    {"cx": 200, "cy": 350, "rx": 58, "ry": 34, "angle": 24,  "lesion": "spider"},  # -> Red Spider
]

_DEMO_W, _DEMO_H = 640, 480


def _rotated_ellipse(cx: float, cy: float, rx: float, ry: float, angle_deg: float,
                     n: int = 48) -> tuple[np.ndarray, np.ndarray]:
    """返回椭圆轮廓的 x/y 坐标数组（便于 numpy 填充）。"""
    t = np.linspace(0, 2 * np.pi, n)
    x0 = rx * np.cos(t)
    y0 = ry * np.sin(t)
    rad = np.deg2rad(angle_deg)
    x = cx + x0 * np.cos(rad) - y0 * np.sin(rad)
    y = cy + x0 * np.sin(rad) + y0 * np.cos(rad)
    return x, y


def make_demo_frame(width: int = _DEMO_W, height: int = _DEMO_H, seed: int | None = None) -> np.ndarray:
    """生成茶园演示帧（BGR，uint8）。"""
    rng = np.random.default_rng(seed)
    # 茶园背景：深绿渐变 + 轻微噪点
    grad = np.linspace(55, 38, height, dtype=np.float32)[:, None] * np.ones((1, width), dtype=np.float32)
    frame = np.stack([grad * 0.55, grad * 0.95, grad * 0.72], axis=-1).astype(np.uint8)
    frame = np.clip(frame.astype(np.int16) + rng.integers(-4, 5, size=frame.shape, dtype=np.int16), 0, 255).astype(np.uint8)

    # 地表土壤色块
    soil_y = int(height * 0.78)
    frame[soil_y:, :, 0] = np.clip(frame[soil_y:, :, 0].astype(np.int16) + 30, 0, 255).astype(np.uint8)
    frame[soil_y:, :, 1] = np.clip(frame[soil_y:, :, 1].astype(np.int16) + 18, 0, 255).astype(np.uint8)

    # 滴灌管道（横向一条，点缀阀门）
    frame[soil_y + 18, 60:580, 0] = 90
    frame[soil_y + 18, 60:580, 1] = 60
    frame[soil_y + 18, 60:580, 2] = 60

    # 叶片
    for leaf in DEMO_LEAVES:
        x, y = _rotated_ellipse(leaf["cx"], leaf["cy"], leaf["rx"], leaf["ry"], leaf["angle"])
        pts = np.clip(np.stack([x, y], axis=1).astype(np.int32), 0, width - 1)
        # 叶片底色（深浅不一）
        base_g = int(90 + (leaf["cx"] % 5) * 12)
        cv_pts = pts.reshape(-1, 1, 2)
        # 用 cv2 填充椭圆更干净；没有 cv2 时用多边形填充
        _fill_poly(frame, cv_pts, (30, base_g, 45))
        # 叶片主脉
        x0, y0 = int(leaf["cx"] - leaf["rx"] * 0.7), int(leaf["cy"] - leaf["ry"] * 0.7)
        x1, y1 = int(leaf["cx"] + leaf["rx"] * 0.7), int(leaf["cy"] + leaf["ry"] * 0.7)
        _line(frame, x0, y0, x1, y1, (25, 60, 35), 2)
        # 病斑
        lesion = leaf["lesion"]
        if lesion == "brown":      # 褐枯病：棕色斑块
            for _ in range(6):
                px = int(leaf["cx"] + rng.integers(-40, 40))
                py = int(leaf["cy"] + rng.integers(-20, 20))
                _fill_circle(frame, px, py, int(rng.integers(4, 10)), (25, 55, 110))
        elif lesion == "rust":     # 叶锈病：黄橙色孢子堆
            for _ in range(8):
                px = int(leaf["cx"] + rng.integers(-38, 38))
                py = int(leaf["cy"] + rng.integers(-18, 18))
                _fill_circle(frame, px, py, int(rng.integers(2, 6)), (30, 130, 200))
        elif lesion == "spider":   # 红蜘蛛：细小红点
            for _ in range(12):
                px = int(leaf["cx"] + rng.integers(-44, 44))
                py = int(leaf["cy"] + rng.integers(-22, 22))
                _fill_circle(frame, px, py, 2, (40, 40, 200))
    return frame


def _fill_poly(frame, pts, color):
    try:
        import cv2
        cv2.fillPoly(frame, [pts], color)
        return
    except Exception:  # noqa: BLE001
        pass
    # 纯 numpy 兜底：逐行填充
    h = frame.shape[0]
    for y in range(h):
        xs = pts[:, 0, 0][(pts[:, 0, 1] == y)]
        if xs.size:
            frame[y, int(xs.min()):int(xs.max()) + 1] = color


def _fill_circle(frame, cx, cy, r, color):
    try:
        import cv2
        cv2.circle(frame, (int(cx), int(cy)), int(r), color, -1)
        return
    except Exception:  # noqa: BLE001
        pass
    h, w = frame.shape[:2]
    y0, y1 = max(0, cy - r), min(h - 1, cy + r)
    for y in range(y0, y1 + 1):
        dx = int((r ** 2 - (y - cy) ** 2) ** 0.5) if r ** 2 >= (y - cy) ** 2 else 0
        x0, x1 = max(0, cx - dx), min(w - 1, cx + dx)
        frame[y, x0:x1 + 1] = color


def _line(frame, x0, y0, x1, y1, color, thickness):
    try:
        import cv2
        cv2.line(frame, (x0, y0), (x1, y1), color, thickness)
        return
    except Exception:  # noqa: BLE001
        pass
    n = max(abs(x1 - x0), abs(y1 - y0), 1)
    for i in range(n + 1):
        x = int(round(x0 + (x1 - x0) * i / n))
        y = int(round(y0 + (y1 - y0) * i / n))
        if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
            frame[y, x] = color


def _run_with_timeout(fn, timeout: float, default):
    """在子线程执行 fn 并限时等待；超时返回 default（fn 可能仍阻塞，线程为 daemon 无害）。"""
    holder: dict = {}

    def _w():
        try:
            holder["v"] = fn()
        except Exception as e:  # noqa: BLE001
            holder["err"] = str(e)

    t = threading.Thread(target=_w, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        return default
    if "err" in holder:
        return default
    return holder.get("v", default)


class CameraCapture:
    """摄像头采集（支持 USB / RTSP / HTTP 多类源，带自动降级）。

    帧来源标记（last_source）：
      - "usb_cam"    ：画面来自 USB 摄像头实拍
      - "net_cam"    ：画面来自网络视频流（RTSP / HTTP）
      - "demo_frame" ：源不可用/读到无效（黑）帧，降级为演示合成帧
    """

    _BLACK_FRAME_MEAN = 8.0  # 平均亮度低于该值视为无效（黑）帧

    def __init__(self, index: int = 0, fallback: bool = True, source: str | None = None,
                 name: str | None = None):
        # source: None / "usb:N" → USB 摄像头；"rtsp://…"/"http(s)://…" → 网络视频流
        src = (source or os.getenv("CAMERA_SOURCE", "") or "").strip()
        if not src or src.lower().startswith("usb:"):
            src = f"usb:{index}"
        self.source = src
        self.is_network = src.lower().startswith(("rtsp://", "http://", "https://"))
        self.index = index
        self.fallback = fallback
        self.name = name or (self.source if self.is_network else f"USB 摄像头 #{index}")
        self._cap = None
        self._lock = threading.Lock()
        # 实例级摄像头 IO 锁：同源任意时刻只允许一个原生调用（构造/试读/read/release）执行。
        # 拿锁方式为非阻塞 acquire(timeout)：若上一个调用仍卡在原生层（MSMF 坏句柄），
        # 本轮立即放弃而非排队等待 —— 这从根本上杜绝并发构造/释放导致的
        # Windows access violation 崩溃，也绝不出现锁等待死锁；
        # 卡住的调用结束后锁自动释放，采集自动恢复（期间该源降级为演示帧，系统保持流畅）。
        self._io_lock = threading.Lock()
        self.last_source = "demo_frame"
        # 快速降级：确认所有后端都打不开摄像头后，一段时间内不再反复重建句柄
        #（cv2.VideoCapture 在无设备/占用时可能阻塞数秒，反复重试会把 Agent 每轮都拖慢）
        self._fail_fast_until = 0.0
        # 惰性打开：初始化时不抢占摄像头（8510 服务导入、Agent 未启动时摄像头保持空闲），
        # 由后台采集线程首次循环时才尝试打开
        # ---- 后台采集线程（每源一个，主线程永不阻塞在摄像头 IO 上）----
        self._frame_lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._bg_stop = threading.Event()
        self._bg_thread: threading.Thread | None = None
        self._read_busy = False   # 上次限时 read 超时后标记：不重复创建超时线程
        # 不在此处启动后台线程：模块级单例（camera / camera_manager）导入时
        # 若立即启动会抢摄像头/起线程（pytest 导入即并发构造导致原生崩溃）。
        # 由 read_frame() 首次调用时惰性启动（_ensure_bg）。

    def _start_bg(self) -> None:
        """启动本源的常驻后台采集线程（daemon）：持续刷新最新帧缓存。

        主线程（Agent 主循环/推流/REST）只读缓存，摄像头任何 IO 卡顿都只影响
        后台线程自身，绝不拖累 Agent —— 这是根治「摄像头假成功/句柄漂移导致
        越跑越卡、Agent 启动不了/停不掉」的关键。
        """
        if self._bg_thread is not None and self._bg_thread.is_alive():
            return
        self._bg_stop.clear()
        self._bg_thread = threading.Thread(target=self._bg_loop, daemon=True,
                                           name=f"cam-bg-{self.index}-{self.source[:24]}")
        self._bg_thread.start()

    def _bg_loop(self) -> None:
        """后台采集循环：打开→读→缓存最新有效帧；失败走快速降级休眠。

        任何异常都不抛出（daemon 线程异常即退出）；摄像头恢复由降级窗口
        到期后的重试自动接回。窗口内主线程读旧缓存或演示帧，系统始终流畅。
        """
        while not self._bg_stop.is_set():
            try:
                self._ensure_open()
                if time.monotonic() < self._fail_fast_until:
                    # 已确认不可用：窗口内不重试（窗口长短按源类型：网络 10s / USB 60s）
                    self._bg_stop.wait(2.0)
                    continue
                frame = self._read_valid_frame()
                if frame is not None:
                    with self._frame_lock:
                        self._latest_frame = frame
                        self.last_source = self._src_tag()
                    self._bg_stop.wait(0.05)   # 正常：贴近摄像头帧率循环
                    continue
                # 黑帧/读取失败 → 重建句柄再试一次（占用恢复、驱动瞬态）。
                # 注意：不能调 self.release()（它会 join 当前线程自己造成自杀退出），
                # 只内联摘除并释放句柄，再打开新句柄，全程串行。
                with self._lock:
                    cap = self._cap
                    self._cap = None
                    self._read_busy = False  # 新句柄即将建立：允许重新创建限时读线程
                self._release_cap(cap)
                self._open()
                frame = self._read_valid_frame()
                if frame is not None:
                    with self._frame_lock:
                        self._latest_frame = frame
                        self.last_source = self._src_tag()
                    continue
                # 打开成功但持续读不到有效帧（句柄异常但 isOpened=True），
                # 或打开失败 → 快速降级窗口，避免高频重试堆积 IO/线程
                self._fail_fast_until = time.monotonic() + (10.0 if self.is_network else 120.0)
                self._bg_stop.wait(2.0)
            except Exception:  # noqa: BLE001 - 单轮失败不退出循环，降级窗口后重试
                self._bg_stop.wait(2.0)

    @property
    def kind(self) -> str:
        """源类型：usb / rtsp / http。"""
        if self.is_network:
            return "rtsp" if self.source.lower().startswith("rtsp") else "http"
        return "usb"

    # 进程级互斥说明见 _io_lock（实例级非阻塞锁，注释在 __init__）。
    def _read_with_lock(self, cap) -> np.ndarray | None:
        """持锁读一帧有效画面；拿不到锁（上一调用仍卡住）立即返回 None。"""
        if not self._io_lock.acquire(timeout=1.0):
            return None
        try:
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0 and float(frame.mean()) >= self._BLACK_FRAME_MEAN:
                return frame
            return None
        except Exception:  # noqa: BLE001
            return None
        finally:
            try:
                self._io_lock.release()
            except Exception:  # noqa: BLE001
                pass

    def _try_open_backend(self, backend, holder: dict | None = None) -> object | None:
        """尝试用指定后端打开并验证能读到一帧有效画面；返回可用句柄或 None。

        仅构造成功不算数——MSMF 在设备被占用时 isOpened()=True 但 read() 无限阻塞，
        必须在限时线程内完成「构造→试读→非黑」验证，坏后端直接弃用换下一个。
        holder 用于在超时后让调用方拿到尚未返回的句柄并主动释放。
        整个操作持 _io_lock：同一源同时最多一个原生调用，杜绝并发构造/释放崩溃；
        持锁期间失败路径内联释放句柄（避免可重入锁死锁）。
        """
        import cv2
        if not self._io_lock.acquire(timeout=1.0):
            return None
        try:
            cap = None
            try:
                cap = cv2.VideoCapture(self.index, backend)
            except Exception:  # noqa: BLE001
                try:
                    cap = cv2.VideoCapture(self.index)
                except Exception:  # noqa: BLE001
                    return None
            if holder is not None:
                holder["cap"] = cap
            if cap is None or not cap.isOpened():
                try:
                    if cap is not None:
                        cap.release()
                except Exception:  # noqa: BLE001
                    pass
                return None
            # 试读验证：读不到有效帧（黑帧/阻塞）的后端立即弃用并释放
            try:
                ok, frame = cap.read()
                if ok and frame is not None and frame.size > 0 and float(frame.mean()) >= self._BLACK_FRAME_MEAN:
                    return cap
            except Exception:  # noqa: BLE001
                pass
            try:
                cap.release()
            except Exception:  # noqa: BLE001
                pass
            return None
        finally:
            try:
                self._io_lock.release()
            except Exception:  # noqa: BLE001
                pass

    def _release_cap(self, cap) -> None:
        """限时释放句柄（持 _io_lock），避免坏驱动在 release() 中再次阻塞调用线程。"""
        if cap is None:
            return

        def _do():
            if not self._io_lock.acquire(timeout=1.0):
                return
            try:
                cap.release()
            except Exception:  # noqa: BLE001
                pass
            finally:
                try:
                    self._io_lock.release()
                except Exception:  # noqa: BLE001
                    pass

        _run_with_timeout(_do, 1.0, None)

    def _open(self) -> None:
        """按源类型打开摄像头：USB 走后端探测，网络流走 FFMPEG。"""
        if self.is_network:
            self._open_network()
        else:
            self._open_usb()

    def _open_network(self) -> None:
        """打开 RTSP / HTTP 网络视频流（FFMPEG 后端，限时 5 秒 + 试读验证）。

        RTSP/HTTP 源在不可达时构造可能长时间阻塞，必须限时；
        断线后由 read_frame() 走 10 秒快速重连窗口自动恢复。
        """
        import cv2
        holder: dict = {}

        def _do():
            if not self._io_lock.acquire(timeout=1.0):
                return None
            try:
                cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
                holder["cap"] = cap
                return cap
            finally:
                try:
                    self._io_lock.release()
                except Exception:  # noqa: BLE001
                    pass

        cap = _run_with_timeout(_do, 5.0, None)
        if cap is None or not cap.isOpened():
            self._release_cap(holder.get("cap"))
            self._fail_fast_until = time.monotonic() + 10.0
            return
        # 试读验证：本地局域网读帧是毫秒级，读不到有效帧即弃用（限时线程 + 实例锁）
        frame = _run_with_timeout(lambda: self._read_with_lock(cap), 1.5, None)
        if frame is not None:
            self._cap = cap
            self._read_busy = False  # 新句柄：允许重新创建限时读线程
            self._fail_fast_until = 0.0
            return
        self._release_cap(cap)
        self._fail_fast_until = time.monotonic() + 10.0

    def _open_usb(self) -> None:
        """按「原生后端优先」顺序打开摄像头：MSMF(Windows) → DSHOW → 默认。

        所有后端共享 4 秒总预算，每个后端在限时线程内完成「构造→试读验证」
        （句柄异常时 cv2.VideoCapture/read 可能无限阻塞，超时即弃用该后端）。
        成功打开后清除快速降级标记；全部失败则进入快速降级。
        """
        try:
            import cv2
            backends = []
            # DSHOW 优先：DirectShow 在摄像头占用/无设备时构造失败快速返回、
            # 无 MSMF 的异步 OnReadSample 回调（反复构造/试读在坏环境下会触发
            # OpenCV 原生 heap corruption 崩溃）。MSMF 仅作兜底。
            for name in ("CAP_DSHOW", "CAP_MSMF"):
                if hasattr(cv2, name):
                    backends.append(getattr(cv2, name))
            backends.append(cv2.CAP_ANY)
            deadline = time.monotonic() + 4.0
            for backend in backends:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                holder: dict = {}
                cap = _run_with_timeout(
                    lambda b=backend, h=holder: self._try_open_backend(b, h),
                    remaining,
                    None,
                )
                if cap is not None:
                    self._cap = cap
                    self._read_busy = False  # 新句柄：允许重新创建限时读线程
                    self._fail_fast_until = 0.0
                    return
                # 超时后工作线程可能仍卡在 read()，主动释放它创建的句柄。
                self._release_cap(holder.get("cap"))
        except Exception:  # noqa: BLE001
            pass
        self._cap = None
        # 进入快速降级窗口（60 秒内不再重试，之后自动再尝试一次，便于中途插入摄像头）
        self._fail_fast_until = time.monotonic() + 60.0

    def _ensure_open(self) -> None:
        """惰性打开：无句柄且不在快速降级窗口内时尝试打开摄像头。"""
        if self._cap is None and time.monotonic() >= self._fail_fast_until:
            self._open()

    @property
    def available(self) -> bool:
        with self._frame_lock:
            return self._latest_frame is not None

    def _read_valid_frame(self) -> np.ndarray | None:
        """尝试从当前摄像头句柄读一帧有效（非黑）画面。

        cap.read() 在句柄失效时（Windows MSMF 常见：grab 失败/设备被占用）
        可能无限阻塞且不抛异常，因此放入子线程并设 1.5 秒超时；
        超时判定句柄失效，由后台线程走重建/快速降级，避免 Agent 主循环被拖死。

        超时后置 _read_busy=True：被释放的坏句柄上残留的阻塞 read 线程
        不再被新调用叠加（同源同一时刻最多一个残留线程），直到重建成功新句柄。
        """
        if self._cap is None or self._read_busy:
            return None
        cap = self._cap
        holder: dict = {}

        def _do_read():
            holder["frame"] = self._read_with_lock(cap)  # 持实例锁读帧；拿不到锁返回 None

        t = threading.Thread(target=_do_read, daemon=True)
        t.start()
        t.join(timeout=1.5)
        if t.is_alive():
            # 不能把仍在阻塞的句柄留给下一轮继续 read()：先摘除并异步释放，
            # 后续由后台线程重新打开摄像头；同时标记忙碌，不重复创建超时线程。
            # （卡住的 read 线程持锁期间，本源的后续原生调用会自动跳过 → 仅该源降级）
            self._read_busy = True
            self._cap = None
            self._release_cap(cap)
            return None  # read 阻塞超时 → 上层重建句柄或快速降级
        self._read_busy = False
        return holder.get("frame")

    def _src_tag(self) -> str:
        """当前源的实拍标记：usb_cam / net_cam。"""
        return "net_cam" if self.is_network else "usb_cam"

    def _ensure_bg(self) -> None:
        """确保本源的常驻后台采集线程已启动（未启动或已退出则启动）。"""
        if self._bg_thread is not None and self._bg_thread.is_alive():
            return
        self._start_bg()

    def read_frame(self) -> np.ndarray | None:
        """读取最近一帧 BGR 图像（主线程：只读缓存，毫秒级返回，永不阻塞）。

        画面由后台采集线程持续刷新：摄像头可用时返回实拍帧；
        不可用/尚无帧时返回演示合成帧（fallback=True）或 None。
        任何摄像头 IO 卡顿只发生在后台线程，不影响 Agent 主循环与推流。
        """
        self._ensure_bg()  # 惰性启动后台采集（构造不启动，首次读取时启动）
        with self._frame_lock:
            frame = self._latest_frame
            if frame is not None:
                self.last_source = self._src_tag()
                return frame
        # 后台线程还在初始化（或摄像头不可用）：降级演示帧
        if self.fallback:
            self.last_source = "demo_frame"
            return make_demo_frame()
        self.last_source = "none"
        return None

    def release(self) -> None:
        """停止后台采集线程并释放摄像头句柄。

        先停线程再释放句柄：避免后台采集线程与释放动作并发触碰同一设备
        （并发构造/释放是 MSMF access violation 崩溃的根因）。
        """
        self._bg_stop.set()
        # 等待后台线程退出（正常 2s 内；卡在原生调用时最多等 5s，
        # join 超时也不强杀——卡住的线程由实例锁隔离，只会让该源降级）
        if self._bg_thread is not None and self._bg_thread.is_alive():
            self._bg_thread.join(timeout=5)
        with self._lock:
            if self._cap is not None:
                cap = self._cap
                self._cap = None  # 先解除引用，避免阻塞期间其他逻辑复用坏句柄
                self._release_cap(cap)
        with self._frame_lock:
            self._latest_frame = None
        self._bg_stop.clear()  # 允许下次 _start_bg() 重启采集


# 模块级单例（惰性创建，避免无摄像头环境启动报错）
_camera_instance: CameraCapture | None = None
_camera_lock = threading.Lock()


def _get_camera() -> CameraCapture:
    global _camera_instance
    with _camera_lock:
        if _camera_instance is None:
            _camera_instance = CameraCapture(index=0, source=os.getenv("CAMERA_SOURCE"))
        return _camera_instance


camera = _get_camera()


class CameraManager:
    """多摄像头源管理器（监控台）。

    源配置按账号（user_id）隔离，存储在数据库 camera_sources 表：
      - 游客 user_id=0：无配置时使用环境变量 CAMERA_SOURCE（默认 usb:0）作为默认主源
      - 登录用户：加载自己在数据库中配置的源列表
    采集实例按 source 全局池化：同一 source（如 usb:0）全系统只打开一个
    CameraCapture 实例，多账号引用共享 —— 避免两个账号同时配置同一个
    物理摄像头时并发构造/释放导致的原生崩溃或相互抢占。
    """

    def __init__(self):
        self._cams: dict[str, CameraCapture] = {}
        self._meta: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._seq = 0
        self._user_id = 0
        self._loaded = False
        # 全局采集池：source -> CameraCapture（物理设备只开一个实例，多账号共享）
        self._pool: dict[str, CameraCapture] = {}
        # source -> 引用该采集器的 user_id 集合（池实例何时可释放）
        self._refs: dict[str, set[int]] = {}

    # ---------- 内部 ----------
    def _next_id(self) -> str:
        self._seq += 1
        return f"cam{self._seq}"

    def _pool_get(self, source: str, name: str = "") -> CameraCapture:
        """从全局池取采集实例（不存在则创建）。物理设备同一 source 只开一个。"""
        inst = self._pool.get(source)
        if inst is None:
            inst = CameraCapture(source=source, name=name or None)
            self._pool[source] = inst
        return inst

    def _default_source(self) -> str:
        src = (os.getenv("CAMERA_SOURCE", "") or "").strip()
        return src or "usb:0"

    def _load_user_sources(self, user_id: int) -> None:
        """加载某账号的源配置到当前视图；无配置/数据库不可用时用默认主源。"""
        from .. import db
        try:
            sources = db.user_cameras(user_id)
        except Exception:  # noqa: BLE001
            sources = []
        with self._lock:
            self._cams.clear()
            self._meta.clear()
            self._seq = 0
            if not sources:
                # 游客/无配置：默认主源（固定 cam1，便于前端默认展示）
                src = self._default_source()
                cam_key = "cam1"
                inst = self._pool_get(src, "默认摄像头")
                self._refs.setdefault(src, set()).add(int(user_id or 0))
                self._cams[cam_key] = inst
                self._meta[cam_key] = {
                    "id": cam_key, "name": inst.name, "source": src,
                    "kind": inst.kind, "enabled": True,
                    "added_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                self._seq = 1
                return
            for s in sources:
                cam_key = s.get("cam_key") or self._next_id()
                src = s.get("source") or ""
                if not src.lower().startswith(("usb:", "rtsp://", "http://", "https://")):
                    continue
                inst = self._pool_get(src, s.get("name") or "")
                self._refs.setdefault(src, set()).add(int(user_id or 0))
                self._cams[cam_key] = inst
                self._meta[cam_key] = {
                    "id": cam_key, "name": s.get("name") or inst.name, "source": src,
                    "kind": inst.kind, "enabled": bool(s.get("enabled", 1)),
                    "added_at": str(s.get("created_at") or time.strftime("%Y-%m-%d %H:%M:%S")),
                }
                # 保证后续新加的 cam_key 不与已有冲突
                m = re.match(r"cam(\d+)", cam_key)
                if m:
                    self._seq = max(self._seq, int(m.group(1)))

    def set_user(self, user_id: int, force: bool = False) -> None:
        """切换当前账号视图（游客 0 / 登录用户 id）。切换只换视图，不释放池实例。

        force=True 时强制从数据库重载该账号的源配置（用于增删源后刷新视图，
        避免内存残留默认源导致与数据库不一致）。
        """
        with self._lock:
            if not force and self._loaded and self._user_id == int(user_id or 0):
                return
            self._user_id = int(user_id or 0)
            self._loaded = True
        self._load_user_sources(self._user_id)

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.set_user(0)

    # ---------- 查询 ----------
    def list(self) -> list[dict]:
        """返回当前账号所有源的元信息 + 实时可用状态（不含画面帧）。"""
        self._ensure_loaded()
        with self._lock:
            out = []
            for cam_id, inst in self._cams.items():
                meta = dict(self._meta[cam_id])
                meta.update({
                    "available": inst.available,
                    "last_source": inst.last_source,
                })
                out.append(meta)
            return out

    def get(self, cam_id: str) -> CameraCapture | None:
        self._ensure_loaded()
        with self._lock:
            return self._cams.get(cam_id)

    def ids(self) -> list[str]:
        self._ensure_loaded()
        with self._lock:
            return list(self._cams.keys())

    # ---------- 增删 ----------
    def add(self, source: str, name: str = "", user_id: int | None = None) -> dict | None:
        """为某账号添加一个摄像头源（usb:N / rtsp://… / http(s)://…），同时写入数据库。

        校验通过返回元信息；源实例从全局池取（同 source 只开一个物理句柄）。
        """
        src = (source or "").strip()
        if not src:
            return None
        if not src.lower().startswith(("usb:", "rtsp://", "http://", "https://")):
            return None
        self._ensure_loaded()
        uid = self._user_id if user_id is None else int(user_id or 0)
        with self._lock:
            cam_key = self._next_id()
            inst = self._pool_get(src, name.strip() or None)
            self._refs.setdefault(src, set()).add(uid)
            self._cams[cam_key] = inst
            self._meta[cam_key] = {
                "id": cam_key, "name": name.strip() or inst.name, "source": src,
                "kind": inst.kind, "enabled": True,
                "added_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            meta = dict(self._meta[cam_key])
        try:
            from .. import db
            db.add_camera(uid, cam_key, meta["name"], src)
        except Exception:  # noqa: BLE001 - 数据库不可用时仅内存生效
            pass
        return meta

    def remove(self, cam_id: str, user_id: int | None = None) -> bool:
        """删除某账号的一个源：从当前视图移除，并从数据库删除；
        池中无其他账号引用时释放物理句柄。"""
        self._ensure_loaded()
        uid = self._user_id if user_id is None else int(user_id or 0)
        with self._lock:
            inst = self._cams.pop(cam_id, None)
            self._meta.pop(cam_id, None)
            if inst is None:
                return False
            src = inst.source
            refs = self._refs.get(src, set())
            refs.discard(uid)
            pool_gone = False
            if not refs:
                self._refs.pop(src, None)
                if self._pool.pop(src, None) is not None:
                    pool_gone = True
        try:
            from .. import db
            db.remove_camera(uid, cam_id)
        except Exception:  # noqa: BLE001
            pass
        if pool_gone:
            inst.release()
        return True

    # ---------- 采集 ----------
    def read_all(self) -> list[dict]:
        """依次读取当前账号所有启用源的一帧（由 runner 调用，返回帧 + 来源标记）。

        先在锁内取「cam_id + 实例 + 元信息」快照，再在锁外逐个读帧：
        set_user 切账号（clear/重建 _cams）与 read_all 并发时不会触发
        RuntimeError: dictionary changed size during iteration。
        """
        self._ensure_loaded()
        with self._lock:
            snapshot = [(cam_id, inst, meta) for cam_id, inst, meta in
                        ((cid, self._cams[cid], self._meta.get(cid, {})) for cid in list(self._cams.keys()))]
        out = []
        for cam_id, inst, meta in snapshot:
            try:
                frame = inst.read_frame()
                out.append({
                    "cam_id": cam_id,
                    "name": meta.get("name", cam_id),
                    "source": meta.get("source", inst.source),
                    "kind": inst.kind,
                    "last_source": inst.last_source,
                    "frame": frame,
                })
            except Exception:  # noqa: BLE001 - 单路失败不影响其他路
                out.append({
                    "cam_id": cam_id,
                    "name": meta.get("name", cam_id),
                    "source": meta.get("source", inst.source),
                    "kind": inst.kind,
                    "last_source": "demo_frame",
                    "frame": None,
                })
        return out


camera_manager = CameraManager()
