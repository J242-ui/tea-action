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


class CameraCapture:
    """USB 摄像头采集（带自动降级）。"""

    def __init__(self, index: int = 0, fallback: bool = True):
        self.index = index
        self.fallback = fallback
        self._cap = None
        self._lock = threading.Lock()
        self._open()

    def _open(self) -> None:
        try:
            import cv2
            cap = cv2.VideoCapture(self.index)
            if cap.isOpened():
                self._cap = cap
        except Exception:  # noqa: BLE001
            self._cap = None

    @property
    def available(self) -> bool:
        return self._cap is not None

    def read_frame(self) -> np.ndarray | None:
        """读取一帧 BGR 图像。摄像头不可用时返回演示合成帧（fallback 开启时）。"""
        with self._lock:
            if self._cap is not None:
                try:
                    import cv2
                    ok, frame = self._cap.read()
                    if ok and frame is not None and frame.size > 0:
                        return frame
                except Exception:  # noqa: BLE001
                    pass
        if self.fallback:
            return make_demo_frame()
        return None

    def release(self) -> None:
        with self._lock:
            if self._cap is not None:
                try:
                    self._cap.release()
                except Exception:  # noqa: BLE001
                    pass
                self._cap = None


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
