"""摄像头采集与演示帧生成。

CameraCapture：
  - 优先尝试 OpenCV(cv2) 打开 USB 摄像头（index 可配置）
  - cv2 未安装或摄像头不可用时，自动降级返回 make_demo_frame() 合成帧

make_demo_frame：
  用纯 numpy 绘制「茶园叶片」演示帧（若干叶片 + 病斑），
  与 SimDetector 的检测坐标一一对应，保证无模型时演示链路依然完整。
"""
from __future__ import annotations

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
    """USB 摄像头采集（带自动降级）。

    帧来源标记：
      - last_source == "usb_cam"    ：本轮画面来自摄像头实拍
      - last_source == "demo_frame" ：摄像头不可用/读到无效（黑）帧，降级为演示合成帧
    """

    _BLACK_FRAME_MEAN = 8.0  # 平均亮度低于该值视为无效（黑）帧

    def __init__(self, index: int = 0, fallback: bool = True):
        self.index = index
        self.fallback = fallback
        self._cap = None
        self._lock = threading.Lock()
        self.last_source = "demo_frame"
        # 快速降级：确认所有后端都打不开摄像头后，一段时间内不再反复重建句柄
        #（cv2.VideoCapture 在无设备/占用时可能阻塞数秒，反复重试会把 Agent 每轮都拖慢）
        self._fail_fast_until = 0.0
        # 惰性打开：初始化时不抢占摄像头（8510 服务导入、Agent 未启动时摄像头保持空闲），
        # 首次 read_frame() 才尝试打开，避免服务启动就占用摄像头

    def _try_open_backend(self, backend, holder: dict | None = None) -> object | None:
        """尝试用指定后端打开并验证能读到一帧有效画面；返回可用句柄或 None。

        仅构造成功不算数——MSMF 在设备被占用时 isOpened()=True 但 read() 无限阻塞，
        必须在限时线程内完成「构造→试读→非黑」验证，坏后端直接弃用换下一个。
        holder 用于在超时后让调用方拿到尚未返回的句柄并主动释放。
        """
        import cv2
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
            self._release_cap(cap)
            return None
        # 试读验证：读不到有效帧（黑帧/阻塞）的后端立即弃用并释放
        try:
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0 and float(frame.mean()) >= self._BLACK_FRAME_MEAN:
                return cap
        except Exception:  # noqa: BLE001
            pass
        self._release_cap(cap)
        return None

    @staticmethod
    def _release_cap(cap) -> None:
        """限时释放句柄，避免坏驱动在 release() 中再次阻塞调用线程。"""
        if cap is None:
            return
        _run_with_timeout(lambda: cap.release(), 1.0, None)

    def _open(self) -> None:
        """按「原生后端优先」顺序打开摄像头：MSMF(Windows) → DSHOW → 默认。

        所有后端共享 4 秒总预算，每个后端在限时线程内完成「构造→试读验证」
        （句柄异常时 cv2.VideoCapture/read 可能无限阻塞，超时即弃用该后端）。
        成功打开后清除快速降级标记；全部失败则进入快速降级。
        """
        try:
            import cv2
            backends = []
            for name in ("CAP_MSMF", "CAP_DSHOW"):
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
        return self._cap is not None

    def _read_valid_frame(self) -> np.ndarray | None:
        """尝试从当前摄像头句柄读一帧有效（非黑）画面。

        cap.read() 在句柄失效时（Windows MSMF 常见：grab 失败/设备被占用）
        可能无限阻塞且不抛异常，因此放入子线程并设 1.5 秒超时；
        超时判定句柄失效，由调用方走重建/快速降级，避免 Agent 主循环被拖死。
        """
        if self._cap is None:
            return None
        cap = self._cap
        holder: dict = {}

        def _do_read():
            try:
                ok, frame = cap.read()
                holder["res"] = (ok, frame)
            except Exception as e:  # noqa: BLE001
                holder["err"] = str(e)

        t = threading.Thread(target=_do_read, daemon=True)
        t.start()
        t.join(timeout=1.5)
        if t.is_alive():
            # 不能把仍在阻塞的句柄留给下一轮继续 read()：先摘除并异步释放，
            # 后续由 read_frame() 重新打开摄像头。
            self._cap = None
            self._release_cap(cap)
            return None  # read 阻塞超时 → 上层重建句柄或快速降级
        if "err" in holder:
            return None
        ok, frame = holder.get("res", (False, None))
        if (ok and frame is not None and frame.size > 0
                and float(frame.mean()) >= self._BLACK_FRAME_MEAN):
            return frame
        return None

    def read_frame(self) -> np.ndarray | None:
        """读取一帧 BGR 图像。

        惰性打开：首次调用才尝试初始化摄像头。
        摄像头不可用，或读到无效（纯黑）帧时：
          1) 快速降级窗口内不重试，直接返回演示合成帧（避免无设备时每轮阻塞数秒）
          2) 窗口外（或句柄曾打开成功）先重新打开摄像头再试一次
        帧来源记录在 last_source：usb_cam / demo_frame。
        """
        with self._lock:
            self._ensure_open()
            if time.monotonic() < self._fail_fast_until:
                # 已确认摄像头不可用：不重试，直接降级演示帧
                if self.fallback:
                    self.last_source = "demo_frame"
                    return make_demo_frame()
                self.last_source = "none"
                return None
            frame = self._read_valid_frame()
            if frame is not None:
                self.last_source = "usb_cam"
                return frame
            # 黑帧/读取失败 → 重建句柄再试一次（可解决占用恢复、驱动瞬态问题）
            self.release()
            self._open()
            frame = self._read_valid_frame()
            if frame is not None:
                self.last_source = "usb_cam"
                return frame
            # 打开成功但持续读不到有效帧（MSMF 常见：句柄异常但 isOpened=True）
            # 或打开失败 → 进入快速降级窗口，避免每轮都浪费数秒重试
            self._fail_fast_until = time.monotonic() + 60.0
        if self.fallback:
            self.last_source = "demo_frame"
            return make_demo_frame()
        self.last_source = "none"
        return None

    def release(self) -> None:
        with self._lock:
            if self._cap is not None:
                cap = self._cap
                self._cap = None  # 先解除引用，避免阻塞期间其他逻辑复用坏句柄
                self._release_cap(cap)


# 模块级单例（惰性创建，避免无摄像头环境启动报错）
_camera_instance: CameraCapture | None = None
_camera_lock = threading.Lock()


def _get_camera() -> CameraCapture:
    global _camera_instance
    with _camera_lock:
        if _camera_instance is None:
            _camera_instance = CameraCapture(index=0)
        return _camera_instance


camera = _get_camera()
