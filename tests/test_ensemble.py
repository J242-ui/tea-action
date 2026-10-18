"""三引擎编排层测试：天气归档必须兼容和风原始响应与页面包装响应。"""
from datetime import date

from src import ensemble


def _daily(count: int = 2) -> list[dict]:
    today = date.today()
    return [
        {
            "fxDate": (today.replace(day=min(28, today.day + i))).isoformat(),
            "tempMax": "30",
            "tempMin": "20",
            "humidity": "70",
            "precip": "0",
            "windScaleDay": "2",
        }
        for i in range(count)
    ]


def test_archive_today_accepts_direct_qweather_payload(monkeypatch):
    written = []
    monkeypatch.setattr(ensemble.db, "weather_history_has_date", lambda *_: False)
    monkeypatch.setattr(
        ensemble.db, "archive_weather",
        lambda key, rows: written.append((key, rows)),
    )

    count = ensemble.archive_today({"key": "g1"}, {"daily": _daily()})

    assert count == 2
    assert written == [("g1", _daily())]


def test_archive_today_accepts_nested_page_payload(monkeypatch):
    written = []
    monkeypatch.setattr(ensemble.db, "weather_history_has_date", lambda *_: False)
    monkeypatch.setattr(
        ensemble.db, "archive_weather",
        lambda key, rows: written.append((key, rows)),
    )

    count = ensemble.archive_today({"id": "g2"}, {"daily": {"daily": _daily(1)}})

    assert count == 1
    assert written[0][0] == "g2"


def test_archive_today_is_idempotent_when_today_already_exists(monkeypatch):
    monkeypatch.setattr(ensemble.db, "weather_history_has_date", lambda *_: True)
    monkeypatch.setattr(
        ensemble.db, "archive_weather",
        lambda *_: (_ for _ in ()).throw(AssertionError("不应重复写入")),
    )

    assert ensemble.archive_today({"key": "g1"}, {"daily": _daily()}) == 0


def test_archive_today_ignores_malformed_rows(monkeypatch):
    called = False

    def _archive(*_):
        nonlocal called
        called = True

    monkeypatch.setattr(ensemble.db, "weather_history_has_date", lambda *_: False)
    monkeypatch.setattr(ensemble.db, "archive_weather", _archive)

    assert ensemble.archive_today({"key": "g1"}, {"daily": [None, {"tempMax": 30}]}) == 0
    assert called is False
