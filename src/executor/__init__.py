"""执行器模块：滴灌阀门、喷杆喷头等田间设备的软件状态机。

- 状态机：每个执行器维护 state / since / 预计时长，支持 open|close、on|off 操作
- 事件订阅：每次状态变化发出结构化事件，供 WebSocket 服务广播到网页
- 真实硬件（预留）：把 apply() 里的事件转发到 ESP32/继电器即可驱动真实设备
"""
from .devices import (
    ActuatorBus,
    DripValve,
    SprayBoom,
    bus,
    DEFAULT_SNAPSHOT,
)

__all__ = ["ActuatorBus", "DripValve", "SprayBoom", "bus", "DEFAULT_SNAPSHOT"]
