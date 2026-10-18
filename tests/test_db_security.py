"""数据库层安全与解析回归测试（使用假连接，不依赖 MySQL）。"""
import json

from src import db


class _Cursor:
    def __init__(self):
        self.rowcount = 0
        self.queries: list[tuple[str, tuple]] = []
        self._one = None
        self._many = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def execute(self, sql, args=()):
        self.queries.append((sql, args))
        if sql.startswith("UPDATE"):
            self.rowcount = 0
            self._one = {"id": 1}
        elif sql.startswith("DELETE"):
            self.rowcount = 1

    def fetchone(self):
        return self._one

    def fetchall(self):
        return self._many


class _Conn:
    def __init__(self, cursor):
        self._cursor = cursor
        self.committed = False

    def cursor(self):
        return self._cursor

    def commit(self):
        self.committed = True

    def close(self):
        pass


def test_verify_login_returns_only_session_safe_fields(monkeypatch):
    salt = "a" * 32
    cur = _Cursor()
    cur._one = {
        "id": 7,
        "username": "alice",
        "nickname": "Alice",
        "created_at": "2026-01-01",
        "salt": salt,
        "password_hash": db._hash_password("secret", salt),
    }
    monkeypatch.setattr(db, "connect", lambda: _Conn(cur))
    monkeypatch.setattr(db, "_LOGIN_FAILURES", {})

    user = db.verify_login("alice", "secret")

    assert user == {
        "id": 7,
        "username": "alice",
        "nickname": "Alice",
        "created_at": "2026-01-01",
    }
    assert "password_hash" not in user and "salt" not in user


def test_update_and_delete_garden_include_owner_condition(monkeypatch):
    cur = _Cursor()
    conn = _Conn(cur)
    monkeypatch.setattr(db, "connect", lambda: conn)

    ok, _ = db.update_garden(12, 3, {"name": "茶园"})
    assert ok is True
    update_sql, update_args = next((s, a) for s, a in cur.queries if s.startswith("UPDATE"))
    assert "owner_id=%s" in update_sql
    assert update_args[-2:] == (12, 3)

    ok, _ = db.delete_garden(12, 3)
    assert ok is True
    delete_sql, delete_args = next((s, a) for s, a in cur.queries if s.startswith("DELETE"))
    assert "owner_id=%s" in delete_sql
    assert delete_args == (12, 3)


def test_agent_decision_history_exposes_parsed_inputs_and_actions(monkeypatch):
    cur = _Cursor()
    cur._many = [{
        "decision_type": "noop",
        "summary": "ok",
        "inputs_json": json.dumps({"soil": 50}),
        "actions_json": json.dumps([{"device": "drip_valve", "op": "close"}]),
        "overall_level": "low",
        "created_at": "2026-01-01",
    }]
    monkeypatch.setattr(db, "connect", lambda: _Conn(cur))

    rows = db.agent_decision_history("g1")

    assert rows[0]["inputs"] == {"soil": 50}
    assert rows[0]["actions"] == [{"device": "drip_valve", "op": "close"}]
