"""土壤湿度传感器（软件模拟 + 真实串口接口预留）。

茶树土壤适宜体积含水量约 60%~80%：
  - < 45% 需要灌溉（Agent 决策触发点）
  - > 85% 过湿（停止灌溉）

模拟器行为（贴合自然规律，便于演示决策闭环）：
  - 读数随时间随机游走，昼夜节律：白天气温高蒸发快（湿度缓慢下降），夜间趋于平稳
  - 灌溉事件注入：调用 irrigate() 后湿度按注入量上升，随后缓慢回落
  - 默认从「偏干」状态起步（约 40%），保证演示开始后能触发 Agent 浇水决策
"""
from __future__ import annotations

import random
from datetime import datetime

# 茶树土壤湿度参考阈值（%）
SOIL_DRY = 45.0      # 低于该值视为缺水，Agent 触发灌溉
SOIL_WET = 85.0      # 高于该值视为过湿，停止灌溉
SOIL_OPTIMAL_LOW = 60.0
SOIL_OPTIMAL_HIGH = 80.0


class SoilMoistureSimulator:
    """软件模拟土壤湿度传感器。"""

    def __init__(self, seed: int | None = None, base: float = 42.0):
        self.rng = random.Random(seed)
        self.value = base
        self._irrigation_budget = 0.0   # 上次浇水的剩余影响（%）
        self._read_count = 0

    def _hour_factor(self, hour: int) -> float:
        """昼夜节律：白天(10~16时)蒸发强，早晚次之，夜间最弱。返回蒸发系数。"""
        if 10 <= hour <= 16:
            return 1.0
        if 7 <= hour <= 19:
            return 0.6
        return 0.25

    def read(self) -> dict:
        """读取一次土壤湿度，返回结构化读数。"""
        self._read_count += 1
        now = datetime.now()
        # 1) 灌溉影响衰减（每次衰减约 6% 的剩余量）
        self._irrigation_budget *= 0.94
        # 2) 自然蒸发（随机游走，向适宜区间中心缓慢回归）
        drift = (SOIL_OPTIMAL_LOW - self.value) * 0.004 * self._hour_factor(now.hour)
        noise = self.rng.uniform(-0.35, 0.35)
        self.value += drift + noise - self._irrigation_budget * 0.02
        self.value = max(15.0, min(95.0, self.value))
        return {
            "type": "soil_moisture",
            "value": round(self.value, 1),
            "unit": "%",
            "status": "ok",
            "source": "sim",
            "ts": now.isoformat(timespec="seconds"),
            "meta": {
                "dry_threshold": SOIL_DRY,
                "optimal": [SOIL_OPTIMAL_LOW, SOIL_OPTIMAL_HIGH],
            },
        }

    def irrigate(self, amount: float = 8.0) -> None:
        """灌溉事件：湿度快速上升（模拟滴灌阀门开启后的土壤回水）。"""
        self._irrigation_budget += amount
        self.value = min(90.0, self.value + amount * 0.8)

    def reset(self, base: float = 42.0) -> None:
        self.value = base
        self._irrigation_budget = 0.0


class SerialSoilSensor:
    """真实串口土壤湿度传感器（预留接口）。

    硬件接线说明（以常见电容式土壤湿度传感器 + ESP32/Arduino 为例）：
      - 传感器 AO 接 ESP32 ADC 引脚，ESP32 通过串口(USB)上报 JSON: {"moisture": 62.3}
      - 本类用 pyserial 读取并换算为体积含水量百分比
    未接线时调用 read() 会抛出 NotImplementedError，上层自动降级为模拟器。
    """

    def __init__(self, port: str = "COM3", baud: int = 115200, timeout: float = 2.0):
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self._ser = None

    def read(self) -> dict:
        raise NotImplementedError(
            "真实土壤湿度传感器驱动未接线：请将传感器接入串口后，"
            "实现 SerialSoilSensor.read() 的 pyserial 读取逻辑，或继续使用模拟器。"
        )


# 模块级单例（供 Agent / 页面复用）
sensor = SoilMoistureSimulator(seed=2026, base=42.0)
