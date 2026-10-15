"""茶园气象服务系统 - 配置管理。"""
import json
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

load_dotenv(BASE_DIR / ".env")


def _env_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


def get_weather_api_key() -> str:
    return os.getenv("QWEATHER_API_KEY", "").strip()


def get_qianfan_api_key() -> str:
    return os.getenv("QIANFAN_API_KEY", "").strip()


def get_qianfan_base_url() -> str:
    """大模型 OpenAI 兼容接口地址。

    默认百度千帆 v2（配合 ernie-4.0-8k 等文心模型）；
    使用 DeepSeek 等其它 OpenAI 兼容服务时在 .env 覆盖为对应地址。
    """
    return os.getenv("QIANFAN_BASE_URL", "https://qianfan.baidubce.com/v2").strip()


def get_qianfan_model() -> str:
    return os.getenv("QIANFAN_MODEL", "ernie-4.0-8k").strip()


def get_qwen_api_key() -> str:
    """通义千问（DashScope）API Key：驱动「智能执行 Agent」的决策判断。"""
    return os.getenv("QWEN_API_KEY", "").strip()


def get_qwen_base_url() -> str:
    """千问 OpenAI 兼容接口地址（阿里云百炼 DashScope）。"""
    return os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1").strip()


def get_qwen_model() -> str:
    return os.getenv("QWEN_MODEL", "qwen-plus").strip()


def is_demo_mode() -> bool:
    return _env_bool("DEMO_MODE")


def is_dev_mode() -> bool:
    """开发者模式开关：开启后显示「开发者控制台」页面。"""
    return _env_bool("DEV_MODE", False)


def load_gardens() -> list[dict]:
    """从 data/tea_gardens.json 读取茶园配置。"""
    path = DATA_DIR / "tea_gardens.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)["gardens"]


def get_garden(garden_id: str) -> dict | None:
    for g in load_gardens():
        if g["id"] == garden_id:
            return g
    return None
