"""传感器模块：土壤湿度等环境传感器数据采集。

- 软件模拟（默认）：贴合茶树需水规律的随机游走 + 浇水响应，用于演示闭环
- 真实硬件（预留）：串口驱动接口，接线后实现读取即可无缝替换
"""
from .soil import SoilMoistureSimulator, SerialSoilSensor, sensor

__all__ = ["SoilMoistureSimulator", "SerialSoilSensor", "sensor"]
