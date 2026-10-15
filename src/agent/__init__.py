"""Agent 模块：融合 视觉病害检测 + 土壤湿度 + 气象风险 的智能决策。

- decision.py：决策引擎（规则阈值 + 滞回防抖），输出结构化执行指令
- runner.py  ：Agent 主循环（后台线程：采集→推理→决策→执行→落库→广播）
"""
from .decision import (
    decide,
    IRRIGATE_DRY,
    IRRIGATE_STOP,
    SPRAY_CONF,
)
from .runner import AgentRunner

__all__ = ["decide", "IRRIGATE_DRY", "IRRIGATE_STOP", "SPRAY_CONF", "AgentRunner"]
