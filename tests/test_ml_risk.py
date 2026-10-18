"""ML 风险分类模块单元测试。纯算法/特征层，不依赖数据库。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import ml_risk  # noqa: E402
from src.ml_risk import RiskClassifier, build_daily_features  # noqa: E402


def test_build_daily_features_shape_and_values():
    d = {"fxDate": "2026-04-15", "tempMax": "30", "tempMin": "18",
         "humidity": "70", "precip": "2.5", "windScaleDay": "3"}
    garden = {"altitude_m": 350, "variety": "武夷岩茶"}
    feat = build_daily_features(d, garden)
    assert len(feat) == 8
    assert feat[0] == 30.0      # temp_max
    assert feat[1] == 18.0      # temp_min
    assert feat[4] == 3.0       # wind
    assert feat[5] == 350.0     # altitude
    assert feat[7] == 4.0       # month=April


def test_variety_code_fallback():
    assert ml_risk._variety_code("龙井43") == 0.0
    assert ml_risk._variety_code("铁观音") == 3.0
    assert ml_risk._variety_code("未知品种") == 5.0


def test_predict_uniform_when_not_ready():
    c = RiskClassifier()
    assert not c.ready
    proba = c.predict([30.0, 18.0, 70.0, 2.0, 3.0, 200.0, 0.0, 4.0])
    assert abs(sum(proba.values()) - 1.0) < 1e-9
    for lv in ("low", "medium", "high"):
        assert proba[lv] == 1.0 / 3


def test_predict_over_daily_returns_structure():
    c = RiskClassifier()
    daily = [
        {"fxDate": "2026-09-24", "tempMax": "38", "tempMin": "25", "humidity": "40",
         "precip": "0", "windScaleDay": "2"},
        {"fxDate": "2026-09-25", "tempMax": "22", "tempMin": "16", "humidity": "75",
         "precip": "5", "windScaleDay": "2"},
    ]
    garden = {"altitude_m": 50, "variety": "龙井43"}
    res = c.predict_over_daily(daily, garden)
    assert res["overall_level"] in ("low", "medium", "high")
    assert len(res["per_day"]) == 2
    assert set(res["proba"].keys()) == {"low", "medium", "high"}


def test_predict_maps_sparse_model_classes_without_index_error():
    class FakeModel:
        classes_ = [0, 2]

        @staticmethod
        def predict_proba(_):
            return [[0.6, 0.4]]

    c = RiskClassifier()
    c._model = FakeModel()
    proba = c.predict([30.0, 18.0, 70.0, 2.0, 3.0, 200.0, 0.0, 4.0])

    assert proba == {"low": 0.6, "medium": 0.0, "high": 0.4}
