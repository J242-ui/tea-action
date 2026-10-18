"""每日自动归档天气 + 自动重训模型（定时任务入口）。

职责（与三引擎联动系统配套）：
1. 初始化数据库扩展表（weather_history / risk_feedback）
2. 遍历所有演示茶园，拉取今日 7 天预报并归档进 weather_history
3. 数据不足的茶园生成历史数据打底
4. 用全部茶园历史混合重训 LightGBM 分类器（ml_risk）并保存
5. 加载 bandit（LinUCB）状态，保持持久化文件就绪

单茶园失败不阻断整体；关键环节（数据库不可用导致无法归档/训练）失败时以非零码退出。
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime

from src import bandit as bandit_mod
from src import config, db, ml_risk, weather, ensemble

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("daily_job")


def _key(g: dict) -> str:
    return g.get("key") or g["id"]


def _loc(g: dict) -> str:
    if g.get("location_id"):
        return g["location_id"]
    if g.get("lon") is not None and g.get("lat") is not None:
        return f"{g['lon']},{g['lat']}"
    return ""


def main() -> int:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] === 每日任务开始 ===")

    # 1) 初始化数据库
    try:
        db.init_db()
        db.init_db_ext(strict=True)
    except Exception as e:  # noqa: BLE001
        log.error("数据库初始化失败: %s", e)
        print("ERROR: 数据库初始化失败，无法继续归档/重训。")
        return 1

    gardens = [{**g, "key": _key(g)} for g in config.load_gardens()]
    client = weather.WeatherClient()

    # 2) 归档今日预报
    archived = 0
    for g in gardens:
        try:
            loc = _loc(g)
            if not loc:
                log.warning("[%s] 缺少 location_id/经纬度，跳过归档", g["name"])
                continue
            data = client.get_forecast(loc, days=7)
            written = ensemble.archive_today(g, data)
            if written:
                archived += 1
                log.info("[%s] 今日预报已归档 %s 天", g["name"], written)
            else:
                log.info("[%s] 今日预报已存在或为空，跳过写入", g["name"])
        except Exception as e:  # noqa: BLE001
            log.error("[%s] 归档失败: %s", g["name"], e)

    # 3) 数据打底（无历史则生成）
    for g in gardens:
        try:
            ensemble.ensure_data(g)
        except Exception as e:  # noqa: BLE001
            log.error("[%s] 数据打底失败: %s", g["name"], e)

    # 4) 重训 ML 分类器（多茶园混合，茶园差异作为特征）
    pairs = []
    for g in gardens:
        hist = db.weather_history_series(g["key"], days=365)
        if len(hist) >= 30:
            pairs.append((g, hist))
    trained = False
    if pairs:
        try:
            trained = ml_risk.classifier.train_multiple(pairs)
        except Exception as e:  # noqa: BLE001
            log.error("分类器重训异常: %s", e)
            trained = False

    # 5) 加载 bandit 状态
    bandit_loaded = bandit_mod.bandit.load()

    print("=" * 52)
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] 每日任务完成")
    print(f"  归档茶园: {archived}/{len(gardens)}")
    print(f"  参与训练茶园(>=30天): {len(pairs)}")
    print(f"  分类器: 重训成功={trained}, 就绪={ml_risk.classifier.ready}")
    print(f"  bandit: 已加载={bandit_loaded}, 累计更新={bandit_mod.bandit.n_updates}")
    print("=" * 52)

    # 关键失败（数据库可用但归档与训练全失败）视为失败
    if archived == 0 and not trained:
        print("WARN: 本轮未归档任何茶园且未完成重训，请检查网络/API Key。")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
