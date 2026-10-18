"""和风天气 (QWeather) API 客户端。

提供实时天气、7 天预报、空气质量、生活指数的获取。
未配置 API Key 或开启演示模式时，返回内置演示数据以便体验界面。

性能：真实 Key 模式带 10 分钟 TTL 缓存（进程内）——Agent 每 5s 一轮决策
不会每轮都打两次网络请求，避免慢网络把决策循环拖住；缓存过期后自动刷新。
"""
from __future__ import annotations

import threading
import time

import requests

from . import config

BASE_URL = "https://devapi.qweather.com"
GEO_URL = "https://geoapi.qweather.com"

# 真实请求超时（秒）：网络慢时快速失败，不拖住 Agent 决策线程
REQUEST_TIMEOUT = 5
# 天气数据缓存时长（秒）：实时数据 10 分钟内变化极小，足够支撑高频决策
CACHE_TTL = 600


class WeatherError(RuntimeError):
    pass


class WeatherClient:
    def __init__(self, api_key: str | None = None, demo: bool | None = None):
        self.api_key = api_key if api_key is not None else config.get_weather_api_key()
        self.demo = demo if demo is not None else (not self.api_key or config.is_demo_mode())
        # 进程内 TTL 缓存：{"key": (expire_ts, data)}，跨实例共享（Agent 每轮 new 客户端）
        self._cache: dict = {}
        self._cache_lock = threading.Lock()

    # ---------- 基础请求 ----------
    def _get(self, base: str, path: str, params: dict) -> dict:
        params = {**params, "key": self.api_key}
        resp = requests.get(f"{base}{path}", params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != "200":
            raise WeatherError(f"和风天气接口返回异常: code={data.get('code')}")
        return data

    def _cached(self, key: str, fn) -> dict:
        """读缓存；未命中/过期时调用 fn 并写缓存。缓存写失败/读取异常一律放行。"""
        with self._cache_lock:
            hit = self._cache.get(key)
            if hit is not None and hit[0] > time.monotonic():
                return hit[1]
        data = fn()
        with self._cache_lock:
            self._cache[key] = (time.monotonic() + CACHE_TTL, data)
        return data

    # ---------- 数据接口 ----------
    def get_now(self, location: str) -> dict:
        """实时天气。location 可为 LocationID 或 经度,纬度。"""
        if self.demo:
            return self._demo_now()
        return self._cached(f"now:{location}",
                            lambda: self._get(BASE_URL, "/v7/weather/now", {"location": location}))

    def get_forecast(self, location: str, days: int = 7) -> dict:
        """未来 days 天预报（和风天气支持 3/7/10/15 天）。"""
        if self.demo:
            return self._demo_forecast(days)
        return self._cached(f"forecast:{location}:{days}",
                            lambda: self._get(BASE_URL, f"/v7/weather/{days}d", {"location": location}))

    def get_air(self, location: str) -> dict:
        """实时空气质量。"""
        if self.demo:
            return self._demo_air()
        return self._get(BASE_URL, "/v7/air/now", {"location": location})

    def get_indices(self, location: str, types: str = "1,2,3,4,5,6,7,8,9") -> dict:
        """生活指数。types 含义见和风天气文档（1=运动 2=洗车 3=穿衣 4=钓鱼 5=紫外线 ...）。"""
        if self.demo:
            return self._demo_indices(types)
        return self._get(BASE_URL, "/v7/indices/1d", {"location": location, "type": types})

    def search_city(self, keyword: str) -> list[dict]:
        """按名称搜索城市，返回 LocationID 列表（用于自定义茶园定位）。"""
        if self.demo:
            return [
                {
                    "id": "101210101",
                    "name": f"{keyword}(演示)",
                    "adm1": "浙江省",
                    "adm2": "杭州市",
                    "lat": "30.25",
                    "lon": "120.1",
                }
            ]
        data = self._get(GEO_URL, "/geo/v2/city/lookup", {"location": keyword})
        return data.get("location", [])

    # ---------- 演示数据 ----------
    @staticmethod
    def _demo_now() -> dict:
        return {
            "code": "200",
            "updateTime": "2026-08-30T10:00+08:00",
            "now": {
                "obsTime": "2026-08-30T10:00+08:00",
                "temp": "26",
                "feelsLike": "27",
                "icon": "101",
                "text": "多云",
                "wind360": "135",
                "windDir": "东南风",
                "windScale": "2",
                "windSpeed": "10",
                "humidity": "72",
                "precip": "0.0",
                "pressure": "1003",
                "vis": "16",
                "cloud": "40",
            },
        }

    @staticmethod
    def _demo_forecast(days: int) -> dict:
        import datetime

        today = datetime.date.today()
        texts = ["多云", "小雨", "阴", "晴", "雷阵雨", "多云", "晴"]
        daily = []
        for i in range(min(days, 7)):
            day = today + datetime.timedelta(days=i)
            daily.append(
                {
                    "fxDate": day.isoformat(),
                    "tempMax": str(29 + i % 3),
                    "tempMin": str(18 + i % 4),
                    "iconDay": "101",
                    "textDay": texts[i],
                    "iconNight": "150",
                    "textNight": texts[(i + 1) % 7],
                    "windDirDay": "东南风",
                    "windScaleDay": "2",
                    "windSpeedDay": "12",
                    "humidity": str(70 + i * 3),
                    "precip": f"{0.0 if i % 3 else 8.0}",
                    "pressure": "1003",
                    "vis": "16",
                    "cloud": "45",
                    "uvIndex": "5",
                }
            )
        return {"code": "200", "daily": daily}

    @staticmethod
    def _demo_air() -> dict:
        return {
            "code": "200",
            "now": {
                "pubTime": "2026-08-30T09:00+08:00",
                "aqi": "42",
                "level": "1",
                "category": "优",
                "primary": "NA",
                "pm10": "35",
                "pm2p5": "18",
                "no2": "12",
                "so2": "5",
                "co": "0.6",
                "o3": "80",
            },
        }

    @staticmethod
    def _demo_indices(types: str) -> dict:
        names = {"1": "运动指数", "2": "洗车指数", "3": "穿衣指数", "5": "紫外线指数", "6": "感冒指数"}
        levels = {"1": "较适宜", "2": "较适宜", "3": "热", "5": "中等", "6": "少发"}
        daily = []
        for t in types.split(","):
            if t in names:
                daily.append(
                    {
                        "type": t,
                        "name": names[t],
                        "level": levels.get(t, "1"),
                        "category": "较适宜",
                        "text": f"{names.get(t, '生活指数')}：当前天气条件对{names.get(t, '生活')}较为有利。",
                    }
                )
        return {"code": "200", "daily": daily}
