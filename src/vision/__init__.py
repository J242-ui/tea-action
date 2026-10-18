"""视觉模块：USB 摄像头采集 + 茶叶病害 CNN(YOLO) 推理 + 标注。

- 真实链路：OpenCV 读 USB 摄像头帧 → YOLO 检测病害 → 画框标注
- 降级链路（无摄像头/无模型/依赖未装时自动降级，保证演示可跑）：
  合成茶园演示帧 → 模拟检测器生成确定性病害框
"""
from .capture import CameraCapture, CameraManager, make_demo_frame, camera, camera_manager
from .detector import CLASS_NAMES, TeaDiseaseDetector, SimDetector, detector
from .annotate import draw_detections, frame_to_base64

__all__ = [
    "CameraCapture", "CameraManager", "make_demo_frame", "camera", "camera_manager",
    "CLASS_NAMES", "TeaDiseaseDetector", "SimDetector", "detector",
    "draw_detections", "frame_to_base64",
]
