"""ML 风险等级分类（LightGBM）。

定位：三引擎联动中的【分类引擎】。
- 输入：某天/某段气象特征 + 茶园环境因子（海拔/品种/月份）
- 输出：风险等级 {low, medium, high} 的概率
- 训练标签：冷启动阶段用规则引擎(risk.py)对历史气象做"弱标注"，后续用风险反馈修正
- 持久化：joblib 存到 data/models/ml_risk.joblib，数据积累后调用 train() 自动重训

冷启动：模型未训练或样本不足时，predict 回退到规则引擎判定（保证任何情况可用）。
"""
from __future__ import annotations

import hashlib
import random
from datetime import date, timedelta
from pathlib import Path

from . import config, db, risk

MODEL_DIR = Path(__file__).resolve().parent.parent / "data" / "models"
MODEL_PATH = MODEL_DIR / "ml_risk.joblib"

LEVELS = ["low", "medium", "high"]
_LEVEL_IDX = {lv: i for i, lv in enumerate(LEVELS)}

# 品种编码（稳定映射，未知品种归入 "other"）
_VARIETY_CODES = {
    "龙井43": 0, "龙井": 0, "碧螺春": 1, "水仙": 2, "肉桂": 2, "铁观音": 3, "信阳毛尖": 4,
}


def _variety_code(variety: str) -> float:
    v = str(variety or "")
    for key, code in _VARIETY_CODES.items():
        if key in v:
            return float(code)
    return 5.0  # other


def build_daily_features(d: dict, garden: dict) -> list[float]:
    """把某一天的气象 dict + 茶园环境因子 → 模型特征向量（顺序固定）。"""
    return [
        float(risk._num(d.get("tempMax"))),
        float(risk._num(d.get("tempMin"))),
        float(risk._num(d.get("humidity"))),
        float(risk._num(d.get("precip"))),
        float(risk._num(d.get("windScaleDay"))),
        float(garden.get("altitude_m") or 0),
        _variety_code(garden.get("variety")),
        float(_num_date(d.get("fxDate"))),
    ]


def _num_date(fx_date) -> float:
    """提取月份作为特征（0~12）。"""
    s = str(fx_date or "")
    try:
        return float(int(s[5:7])) if len(s) >= 7 else float(date.today().month)
    except (ValueError, IndexError):
        return float(date.today().month)


# ---------------------------------------------------------------------------
# 模拟历史数据生成（阶段 0 数据打底：无真实历史时让管道可演示）
# ---------------------------------------------------------------------------
def seed_history(garden_key: str, garden: dict, days: int = 180, seed: int = 7) -> int:
    """为该茶园生成 days 天模拟历史气象并写入 weather_history，返回写入条数。

    基于茶树常见气候基线随机游走，按概率注入高温/低温/暴雨/干旱事件，
    使风险标签分布有 low/medium/high 三类，便于训练分类器。
    仅当该茶园历史数据不足时生成。
    """
    if db.weather_history_count(garden_key) >= 30:
        return 0
    # Python 内置 hash 带随机化，跨进程不可复现；使用稳定摘要生成演示历史种子。
    digest = hashlib.sha256(f"{garden_key}:{seed}".encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    altitude = float(garden.get("altitude_m") or 200)
    base = 14.0 + altitude / 400.0  # 海拔越高基准温越低
    rows: list[dict] = []
    today = date.today()
    temp = base + 3.0
    for i in range(days, 0, -1):
        d = today - timedelta(days=i)
        temp += rng.uniform(-2.5, 2.5)
        temp = max(4.0, min(36.0, temp))
        # 偶发极端事件
        event = rng.random()
        precip = rng.uniform(0, 6)
        if event < 0.05:
            temp = rng.uniform(36, 39)          # 高温
            precip = 0.0
        elif event < 0.10:
            temp = rng.uniform(-2, 2)           # 低温/倒春寒
            precip = rng.uniform(0, 3)
        elif event < 0.16:
            precip = rng.uniform(50, 80)        # 暴雨
        elif event < 0.22:
            temp = rng.uniform(32, 36)          # 晴热（易干旱）
            precip = 0.0
        humidity = max(30.0, min(98.0, 70 + (18 - temp) * 1.6 + rng.uniform(-12, 12)))
        rows.append({
            "fxDate": d.isoformat(),
            "tempMax": f"{temp + rng.uniform(1, 3):.1f}",
            "tempMin": f"{temp - rng.uniform(3, 6):.1f}",
            "humidity": f"{humidity:.1f}",
            "precip": f"{precip:.1f}",
            "windScaleDay": f"{int(rng.uniform(1, 7))}",
        })
    db.archive_weather(garden_key, rows)
    return len(rows)


# ---------------------------------------------------------------------------
# 分类器
# ---------------------------------------------------------------------------
class RiskClassifier:
    """LightGBM 多分类器，输出风险等级概率。"""

    def __init__(self):
        self._model = None
        self._feature_names = [
            "temp_max", "temp_min", "humidity", "precip", "wind_scale",
            "altitude_m", "variety_code", "month",
        ]

    @property
    def ready(self) -> bool:
        return self._model is not None

    # ---------- 训练 ----------
    @staticmethod
    def _build_samples(pairs: list[tuple[dict, list[dict]]]) -> tuple[list, list]:
        """由 [(garden, history), ...] 构造 (X, y)。标签用规则引擎对每日气象弱标注。"""
        engine = risk.RiskEngine()
        X, y = [], []
        for garden, history in pairs:
            for d in history:
                now = {"temp": d["temp_max"], "humidity": d["humidity"],
                       "windScale": d["wind_scale"]}
                daily = [{
                    "fxDate": str(d["obs_date"]), "tempMax": d["temp_max"],
                    "tempMin": d["temp_min"], "humidity": d["humidity"],
                    "precip": d["precip"], "windScaleDay": d["wind_scale"],
                }]
                overall = engine.assess_all(now, daily)
                X.append(build_daily_features(daily[0], garden))
                y.append(_LEVEL_IDX[overall["overall_level"]])
        return X, y

    def _fit(self, X: list, y: list) -> bool:
        # 三分类模型必须看到三个等级；只有两类时训练出列数不稳定的模型，
        # 后续按固定索引取 high 概率会越界，因此宁可保持未就绪并继续积累数据。
        if len(X) < 30 or len(set(y)) < 3:
            return False
        from lightgbm import LGBMClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LGBMClassifier(
                n_estimators=200, learning_rate=0.05, num_leaves=15,
                objective="multiclass", num_class=3, random_state=42,
                verbose=-1, class_weight="balanced",
            )),
        ])
        pipe.fit(X, y)
        self._model = pipe
        self.save()
        return True

    def train_from_history(self, garden_key: str, garden: dict,
                           history: list[dict] | None = None,
                           force: bool = False) -> bool:
        """用单茶园历史 + 规则引擎弱标注训练。样本不足 30 时不训练。返回是否成功训练。"""
        if history is None:
            history = db.weather_history_series(garden_key, days=365)
        X, y = self._build_samples([(garden, history)])
        return self._fit(X, y)

    def train_multiple(self, pairs: list[tuple[dict, list[dict]]],
                       force: bool = False) -> bool:
        """用多个茶园的历史混合训练（茶园差异已作为特征输入，模型更泛化）。"""
        X, y = self._build_samples(pairs)
        return self._fit(X, y)

    # ---------- 预测 ----------
    def predict(self, features: list[float]) -> dict:
        """输入单日特征向量，返回 {low, medium, high} 概率。模型未就绪返回均匀分布。"""
        if not self.ready:
            return {lv: 1.0 / 3 for lv in LEVELS}
        proba = self._model.predict_proba([features])[0]
        classes = getattr(self._model, "classes_", None)
        if classes is None:
            return {lv: 1.0 / 3 for lv in LEVELS}
        class_positions = {int(cls): i for i, cls in enumerate(classes)}
        return {
            lv: float(proba[class_positions[_LEVEL_IDX[lv]]])
            if _LEVEL_IDX[lv] in class_positions else 0.0
            for lv in LEVELS
        }

    def predict_over_daily(self, daily: list[dict], garden: dict) -> dict:
        """对未来每天的预报逐日预测风险等级，返回：
        {per_day: [{fxDate, proba, level}], overall_level, proba}
        overall 取各天里等级最高的那个（以其概率为准）。
        """
        per_day = []
        for d in daily:
            feat = build_daily_features(d, garden)
            proba = self.predict(feat)
            level = max(proba, key=lambda k: proba[k])
            per_day.append({"fxDate": d.get("fxDate"), "proba": proba, "level": level})
        # 最高等级
        order = {"low": 0, "medium": 1, "high": 2}
        worst = max(per_day, key=lambda x: (order[x["level"]], x["proba"][x["level"]])) if per_day else None
        return {
            "per_day": per_day,
            "overall_level": worst["level"] if worst else "low",
            "proba": worst["proba"] if worst else {"low": 1.0, "medium": 0.0, "high": 0.0},
        }

    # ---------- 持久化 ----------
    def save(self) -> None:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        import joblib
        joblib.dump({"model": self._model, "feature_names": self._feature_names}, MODEL_PATH)

    def load(self) -> bool:
        if not MODEL_PATH.exists():
            return False
        try:
            import joblib
            obj = joblib.load(MODEL_PATH)
            self._model = obj.get("model")
            return self._model is not None
        except Exception:  # noqa: BLE001
            self._model = None
            return False


# 模块级单例（供页面复用）
classifier = RiskClassifier()
