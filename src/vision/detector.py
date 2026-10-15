"""茶叶病害检测器：Ultralytics YOLO 推理 + 模拟检测降级。

真实模式（backend=ultralytics）：
  - 加载 data/models/tea_yolo.pt（由 training/train_yolo.py 训练产出）
  - 对输入帧跑 model.predict，把结果统一为 {class, conf, bbox}

模拟模式（backend=sim）：
  - ultralytics 未安装 / 模型缺失 / 训练未完成时自动降级
  - 在 capture.make_demo_frame 的演示帧上按叶片坐标生成确定性检测框，
    保证「采集→推理→决策→推送→网页展示」全链路无模型也可演示

类别顺序与 figshare Tea-Leaf Disease and Pest Detection Dataset v1.0 一致
（训练时以数据集 data.yaml 的实际顺序为准，模型自带 names 优先）。
"""
from __future__ import annotations

import os

# Windows 下 numpy/torch/opencv 的 OpenMP 运行时冲突防护（OMP: Error #15），
# 必须在 import torch/ultralytics 之前设置
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from datetime import datetime

import numpy as np

from .capture import DEMO_LEAVES, make_demo_frame

# 与 figshare v1.0 数据集一致的类别（训练脚本会按 data.yaml 校准）
CLASS_NAMES = [
    "Black rot", "Brown blight", "Leaf rust", "Red Spider",
    "Tea Mosquito bug", "Healthy", "White spot", "disease",
]

# 模拟模式下各叶片的检出类别（与合成帧病斑对应）
_SIM_LEAF_CLASS = {
    "brown": "Brown blight",
    "rust": "Leaf rust",
    "spider": "Red Spider",
    "none": "Healthy",
}

# 模拟置信度区间
_SIM_CONF = {"Brown blight": (0.82, 0.93), "Leaf rust": (0.71, 0.86),
             "Red Spider": (0.66, 0.82), "Healthy": (0.88, 0.95)}


def _box_from_leaf(leaf: dict, rng) -> list[float]:
    """按叶片中心/尺寸生成检测框 [x1, y1, x2, y2]。"""
    dx, dy = int(rng.integers(-6, 7)), int(rng.integers(-5, 6))
    x1 = int(leaf["cx"] - leaf["rx"] + dx)
    y1 = int(leaf["cy"] - leaf["ry"] + dy)
    x2 = int(leaf["cx"] + leaf["rx"] + dx)
    y2 = int(leaf["cy"] + leaf["ry"] + dy)
    return [max(0, x1), max(0, y1), x2, y2]


class SimDetector:
    """模拟检测器：在演示帧上生成确定性、动态的检测结果。"""

    def __init__(self, seed: int | None = None):
        self.seed = seed
        self._count = 0

    def detect(self, frame: np.ndarray | None = None) -> dict:
        self._count += 1
        rng = np.random.default_rng((self.seed or 0) + self._count // 3)
        # 每 2 次轮换一次检出组合（让网页展示有动态变化，并模拟「打药→防效→复发」闭环）：
        #   0: 全部 4 片叶（含 3 种病害）
        #   1: 3 片叶（2 种病害 + 1 健康）
        #   2/3: 仅健康叶（模拟药剂起效 / 视野内无病害，持续 4 轮）→ 触发 Agent 停止打药
        cycle = (self._count // 2) % 4
        if cycle == 0:
            selected = DEMO_LEAVES
        elif cycle == 1:
            selected = DEMO_LEAVES[0:1] + DEMO_LEAVES[2:4]
        else:
            selected = [DEMO_LEAVES[2]]  # 仅 Healthy

        detections = []
        for leaf in selected:
            cls = _SIM_LEAF_CLASS.get(leaf["lesion"], "disease")
            lo, hi = _SIM_CONF.get(cls, (0.7, 0.9))
            conf = round(float(rng.uniform(lo, hi)), 2)
            detections.append({
                "class": cls,
                "conf": conf,
                "bbox": _box_from_leaf(leaf, rng),
            })
        return build_detection_result(detections, source="demo_frame", backend="sim",
                                      model="sim-detector", frame=frame)


def build_detection_result(detections: list[dict], source: str, backend: str,
                           model: str, frame: np.ndarray | None = None) -> dict:
    """把检测列表统一封装为结构化结果。"""
    diseases = [d["class"] for d in detections if d["class"] != "Healthy"]
    confs = [d["conf"] for d in detections] or [0.0]
    return {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "source": source,
        "backend": backend,
        "model": model,
        "detections": detections,
        "summary": {
            "total": len(detections),
            "diseases": sorted(set(diseases)),
            "max_conf": round(max(confs), 2),
            "healthy": not diseases,
        },
        "frame_shape": list(frame.shape[:2]) if frame is not None else None,
    }


class TeaDiseaseDetector:
    """茶叶病害 YOLO 检测器（自动降级）。"""

    def __init__(self, model_path: str | None = None):
        from .. import config
        self.model_path = model_path or str(config.DATA_DIR / "models" / "tea_yolo.pt")
        self.model = None
        self.backend = "sim"
        self._sim = SimDetector()   # 复用一个实例，让演示帧轮换逻辑连续计数
        self._try_load()

    def _try_load(self) -> None:
        try:
            from ultralytics import YOLO
            import os
            if os.path.exists(self.model_path):
                self.model = YOLO(self.model_path)
                self.backend = "ultralytics"
                return
        except Exception:  # noqa: BLE001 - 依赖或模型缺失时降级为模拟
            pass
        self.model = None
        self.backend = "sim"

    @property
    def ready(self) -> bool:
        return self.model is not None

    def reload(self) -> bool:
        """模型训练完成后热加载。"""
        self._try_load()
        return self.ready

    def detect(self, frame: np.ndarray | None = None) -> dict:
        """对一帧图像做病害检测，返回统一结构。模型不可用时降级为模拟检测。"""
        if frame is None:
            frame = make_demo_frame()
        if self.model is not None:
            try:
                return self._detect_real(frame)
            except Exception:  # noqa: BLE001 - 推理异常降级
                pass
        return self._sim.detect(frame)

    def _detect_real(self, frame: np.ndarray) -> dict:
        results = self.model.predict(frame, conf=0.25, iou=0.5, verbose=False)
        detections = []
        names = self.model.names if hasattr(self.model, "names") else CLASS_NAMES
        r = results[0]
        if r.boxes is not None and len(r.boxes):
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                conf = round(float(box.conf[0].item()), 2)
                x1, y1, x2, y2 = [round(float(v), 1) for v in box.xyxy[0].tolist()]
                detections.append({
                    "class": names.get(cls_id, f"class_{cls_id}"),
                    "conf": conf,
                    "bbox": [x1, y1, x2, y2],
                })
        return build_detection_result(detections, source="usb_cam", backend="ultralytics",
                                      model=self.model_path, frame=frame)


# 模块级单例
detector = TeaDiseaseDetector()
