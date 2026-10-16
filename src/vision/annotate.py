"""检测结果标注：在帧上画框 + 类别标签，并支持转 base64 JPEG 供网页显示。"""
from __future__ import annotations

import base64

import numpy as np

# 类别 → 颜色（BGR）
_CLASS_COLORS = {
    "Black rot": (40, 40, 40),
    "Brown blight": (25, 55, 110),
    "Leaf rust": (30, 130, 200),
    "Red Spider": (40, 40, 200),
    "Tea Mosquito bug": (80, 140, 200),
    "Healthy": (60, 180, 80),
    "White spot": (90, 90, 200),
    "disease": (50, 50, 150),
}
_DEFAULT_COLOR = (0, 120, 255)


def _color(cls: str) -> tuple[int, int, int]:
    return _CLASS_COLORS.get(cls, _DEFAULT_COLOR)


def draw_detections(frame: np.ndarray, result: dict) -> np.ndarray:
    """在帧上绘制检测框与标签，返回标注后的 BGR 帧。"""
    import copy
    out = copy.deepcopy(frame)
    for det in result.get("detections", []):
        cls = det["class"]
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        color = _color(cls)
        _rectangle(out, x1, y1, x2, y2, color, 2)
        label = f"{cls} {det['conf']:.2f}"
        _put_label(out, x1, y1, label, color)
    return out


def frame_to_base64(frame: np.ndarray, quality: int = 70) -> str:
    """BGR 帧 → data URL（供网页 <img> 直接显示）。

    优先 cv2 编码 JPEG；cv2 未安装时用纯 numpy+zlib 的 PNG 编码兜底，
    保证无 OpenCV 环境（依赖安装中）也能在网页展示画面。
    quality 默认 70：持续录像高频推流时显著减小 payload 与浏览器解码开销；
    静态检测帧（state_update）可传更高质量。
    """
    try:
        import cv2
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if ok:
            return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
    except Exception:  # noqa: BLE001
        pass
    return _png_b64(frame)


def _png_b64(frame: np.ndarray) -> str:
    """纯 Python PNG 编码（无依赖），BGR → RGB。"""
    try:
        import struct
        import zlib
        h, w = frame.shape[:2]
        rgb = np.ascontiguousarray(frame[:, :, ::-1])
        scanlines = [b"\x00" + rgb[y].tobytes() for y in range(h)]
        raw = b"".join(scanlines)
        compressed = zlib.compress(raw, 6)

        def _chunk(tag: bytes, data: bytes) -> bytes:
            body = tag + data
            return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

        png = (
            b"\x89PNG\r\n\x1a\n"
            + _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + _chunk(b"IDAT", compressed)
            + _chunk(b"IEND", b"")
        )
        return "data:image/png;base64," + base64.b64encode(png).decode("ascii")
    except Exception:  # noqa: BLE001
        return ""


def _rectangle(img, x1, y1, x2, y2, color, thickness):
    try:
        import cv2
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        return
    except Exception:  # noqa: BLE001
        pass
    # 纯 numpy 兜底：画矩形边框
    h, w = img.shape[:2]
    x1, x2 = max(0, x1), min(w - 1, x2)
    y1, y2 = max(0, y1), min(h - 1, y2)
    for t in range(max(1, thickness)):
        for x in range(x1 + t, x2 - t + 1):
            for y in (y1 + t, y2 - t):
                if 0 <= y < h and 0 <= x < w:
                    img[y, x] = color
        for y in range(y1 + t, y2 - t + 1):
            for x in (x1 + t, x2 - t):
                if 0 <= y < h and 0 <= x < w:
                    img[y, x] = color


def _put_label(img, x1, y1, text, color):
    """在框左上角画标签底+文字。无 cv2 时跳过文字（演示可接受）。"""
    try:
        import cv2
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y0 = max(0, y1 - th - 8)
        cv2.rectangle(img, (x1, y0), (x1 + tw + 6, y0 + th + 6), color, -1)
        cv2.putText(img, text, (x1 + 3, y0 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    except Exception:  # noqa: BLE001
        pass
