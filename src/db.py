"""MySQL 数据访问层：用户、用户茶园、推送记录。

连接参数从 .env 读取（DB_HOST/DB_PORT/DB_USER/DB_PASSWORD/DB_NAME），
未配置时默认本机 root（空密码）连 tea_garden 库；生产环境务必在 .env 中显式配置 DB_PASSWORD。
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import time
from datetime import datetime
from typing import Any, Iterable

import pymysql
from dotenv import load_dotenv
from pymysql.cursors import DictCursor

from .config import BASE_DIR

load_dotenv(BASE_DIR / ".env")

def _cfg(key: str, default: str) -> str:
    return os.getenv(key, default).strip()

DB_CONFIG = dict(
    host=_cfg("DB_HOST", "127.0.0.1"),
    port=int(_cfg("DB_PORT", "3306")),
    user=_cfg("DB_USER", "root"),
    password=_cfg("DB_PASSWORD", "1006"),
    database=_cfg("DB_NAME", "tea_garden"),
    charset="utf8mb4",
    cursorclass=DictCursor,
    # 快速失败：数据库不可达/慢时不阻塞 Agent 主循环（连接/读写均短超时）
    connect_timeout=2,
    read_timeout=3,
    write_timeout=3,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INT AUTO_INCREMENT PRIMARY KEY,
    username      VARCHAR(64)  NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    salt          VARCHAR(64)  NOT NULL,
    nickname      VARCHAR(64)  DEFAULT NULL,
    created_at    DATETIME     DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS gardens (
    id           INT AUTO_INCREMENT PRIMARY KEY,
    owner_id     INT          NOT NULL,
    name         VARCHAR(128) NOT NULL,
    location     VARCHAR(255) DEFAULT NULL,
    location_id  VARCHAR(32)  DEFAULT NULL,
    lon          DOUBLE       DEFAULT NULL,
    lat          DOUBLE       DEFAULT NULL,
    altitude_m   DOUBLE       DEFAULT NULL,
    variety      VARCHAR(64)  DEFAULT NULL,
    notes        VARCHAR(500) DEFAULT NULL,
    is_demo      TINYINT      DEFAULT 0,
    created_at   DATETIME     DEFAULT CURRENT_TIMESTAMP,
    KEY idx_owner (owner_id),
    CONSTRAINT fk_garden_owner FOREIGN KEY (owner_id) REFERENCES users(id)
      ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS push_log (
    id         INT AUTO_INCREMENT PRIMARY KEY,
    user_id    INT          NOT NULL,
    garden_id  INT          DEFAULT NULL,
    kind       VARCHAR(16)  NOT NULL,
    level      VARCHAR(16)  NOT NULL,
    title      VARCHAR(255) NOT NULL,
    body       TEXT,
    advice     TEXT,
    created_at DATETIME     DEFAULT CURRENT_TIMESTAMP,
    KEY idx_push_user (user_id, created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE TABLE IF NOT EXISTS analysis_log (
    id           INT AUTO_INCREMENT PRIMARY KEY,
    user_id      INT          NOT NULL,
    garden_id    INT          DEFAULT NULL,
    garden_name  VARCHAR(128) DEFAULT NULL,
    kind         VARCHAR(16)  NOT NULL,
    scoped_date  DATE         NOT NULL,
    overall_level VARCHAR(16) DEFAULT 'low',
    title        VARCHAR(255) DEFAULT NULL,
    summary      VARCHAR(600) DEFAULT NULL,
    items_json   MEDIUMTEXT,
    created_at   DATETIME     DEFAULT CURRENT_TIMESTAMP,
    updated_at   DATETIME     DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    KEY idx_al_user (user_id, garden_id, kind, scoped_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

# ----------------------------------------------------------------
# 连接与初始化
# ----------------------------------------------------------------
def connect() -> Any:
    return pymysql.connect(**DB_CONFIG)


def init_db() -> None:
    conn = connect()
    try:
        with conn.cursor() as cur:
            for stmt in _SCHEMA.split(';'):
                stmt = stmt.strip()
                if stmt:
                    cur.execute(stmt)
        conn.commit()
    finally:
        conn.close()


# ----------------------------------------------------------------
# 密码哈希（PBKDF2 + 随机盐）
# ----------------------------------------------------------------
def _hash_password(password: str, salt: str) -> str:
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 120000)
    return dk.hex()


def register(username: str, password: str, nickname: str | None = None) -> tuple[bool, str]:
    """注册。返回 (成功?, 提示)。"""
    username = (username or "").strip()
    if not username or len(username) < 2:
        return False, "用户名至少 2 个字符"
    if not password or len(password) < 4:
        return False, "密码至少 4 位"
    salt = secrets.token_hex(16)
    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM users WHERE username=%s", (username,))
            if cur.fetchone():
                return False, "用户名已存在"
            cur.execute(
                "INSERT INTO users (username, password_hash, salt, nickname) VALUES (%s,%s,%s,%s)",
                (username, _hash_password(password, salt), salt, nickname or username),
            )
        conn.commit()
        return True, "注册成功"
    except Exception as e:  # noqa: BLE001
        return False, f"注册失败：{e}"
    finally:
        conn.close()


# ----------------------------------------------------------------
# 登录限流（进程内）：连续失败达到上限后临时锁定，防暴力破解
# ----------------------------------------------------------------
_LOGIN_FAILURES: dict[str, list[float]] = {}  # username -> 失败时间戳
_LOGIN_MAX_FAIL = 5          # 窗口内最多允许失败次数
_LOGIN_WINDOW = 900          # 窗口时长（秒）= 15 分钟


def _record_login_failure(username: str) -> None:
    _LOGIN_FAILURES.setdefault(username, []).append(time.time())


def _is_login_locked(username: str) -> bool:
    now = time.time()
    ts = [t for t in _LOGIN_FAILURES.get(username, []) if now - t < _LOGIN_WINDOW]
    _LOGIN_FAILURES[username] = ts
    return len(ts) >= _LOGIN_MAX_FAIL


def login_remaining_lock(username: str) -> int | None:
    """返回该用户名距离锁定还剩几次失败机会；已锁定返回 None。"""
    now = time.time()
    ts = [t for t in _LOGIN_FAILURES.get(username, []) if now - t < _LOGIN_WINDOW]
    if len(ts) >= _LOGIN_MAX_FAIL:
        return None
    return _LOGIN_MAX_FAIL - len(ts)


def verify_login(username: str, password: str) -> dict | None:
    """校验登录。成功返回用户 dict，失败返回 None（失败次数过多会临时锁定）。"""
    uname = (username or "").strip()
    if _is_login_locked(uname):
        return None
    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM users WHERE username=%s", (uname,))
            row = cur.fetchone()
            if not row:
                _record_login_failure(uname)
                return None
            h = _hash_password(password, row["salt"])
            if h != row["password_hash"]:
                _record_login_failure(uname)
                return None
            # 登录成功：清零该用户名的失败记录
            _LOGIN_FAILURES.pop(uname, None)
            return dict(row)
    finally:
        conn.close()


# ----------------------------------------------------------------
# 用户的自有茶园
# ----------------------------------------------------------------
def user_gardens(user_id: int) -> list[dict]:
    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM gardens WHERE owner_id=%s ORDER BY id", (user_id,))
            return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def add_garden(user_id: int, g: dict) -> tuple[bool, str]:
    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO gardens (owner_id,name,location,location_id,lon,lat,altitude_m,variety,notes)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (user_id, g["name"], g.get("location"), g.get("location_id"),
                 g.get("lon"), g.get("lat"), g.get("altitude_m"),
                 g.get("variety"), g.get("notes")),
            )
        conn.commit()
        return True, "茶园已添加"
    except Exception as e:  # noqa: BLE001
        return False, f"添加失败：{e}"
    finally:
        conn.close()


def update_garden(garden_id: int, g: dict) -> tuple[bool, str]:
    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """UPDATE gardens SET name=%s, location=%s, location_id=%s, lon=%s, lat=%s,
                   altitude_m=%s, variety=%s, notes=%s WHERE id=%s""",
                (g["name"], g.get("location"), g.get("location_id"), g.get("lon"), g.get("lat"),
                 g.get("altitude_m"), g.get("variety"), g.get("notes"), garden_id),
            )
        conn.commit()
        return True, "已更新"
    except Exception as e:  # noqa: BLE001
        return False, f"更新失败：{e}"
    finally:
        conn.close()


def delete_garden(garden_id: int) -> None:
    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM gardens WHERE id=%s", (garden_id,))
        conn.commit()
    finally:
        conn.close()


# ----------------------------------------------------------------
# 推送记录
# ----------------------------------------------------------------
def log_push(user_id: int, garden_id: int | None, cards: Iterable[dict]) -> None:
    conn = connect()
    try:
        with conn.cursor() as cur:
            for c in cards:
                cur.execute(
                    "INSERT INTO push_log (user_id,garden_id,kind,level,title,body,advice) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s)",
                    (user_id, garden_id, c["kind"], c["level"], c["title"], c.get("body", ""), c.get("advice", "")),
                )
        conn.commit()
    finally:
        conn.close()


def push_history(user_id: int, garden_id: int | None = None, limit: int = 50) -> list[dict]:
    conn = connect()
    try:
        with conn.cursor() as cur:
            if garden_id:
                cur.execute(
                    "SELECT * FROM push_log WHERE user_id=%s AND garden_id=%s ORDER BY created_at DESC LIMIT %s",
                    (user_id, garden_id, limit),
                )
            else:
                cur.execute(
                    "SELECT * FROM push_log WHERE user_id=%s ORDER BY created_at DESC LIMIT %s",
                    (user_id, limit),
                )
            return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()
# ----------------------------------------------------------------
# 研判/建议历史快照 (analysis_log)
# 用户某天、某茶园、某 kind 只保留一条，重复评判则更新；每条存当日全部明细（items_json = json.dumps(list)）。
# kind: 'farm'=农事建议, 'risk'=风险预警。
# ----------------------------------------------------------------
def save_analysis(user_id: int, garden_id: int | None, garden_name: str, kind: str,
                  scoped_date, overall_level: str, title: str, summary: str, items: list) -> None:
    """"UPSERT：同一 (user, garden, kind, date) 报存一条快照，重复则覆盖。items 为 dict 列表，常带 cat/level/name/title/body/advice。"""
    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM analysis_log "
                "WHERE user_id=%s AND garden_id<=>%s AND kind=%s AND scoped_date=%s",
                (user_id, garden_id, kind, scoped_date))
            row = cur.fetchone()
            payload = json.dumps(items, ensure_ascii=False, default=str)
            if row:
                cur.execute(
                    "UPDATE analysis_log SET overall_level=%s,title=%s,summary=%s,items_json=%s "
                    "WHERE id=%s",
                    (overall_level, title, summary, payload, row["id"]))
            else:
                cur.execute(
                    "INSERT INTO analysis_log "
                    "(user_id,garden_id,garden_name,kind,scoped_date,overall_level,title,summary,items_json) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    (user_id, garden_id, garden_name, kind, scoped_date,
                     overall_level, title, summary, payload))
        conn.commit()
    finally:
        conn.close()


def analysis_records(user_id: int, kind: str, garden_id: int | None = None,
                     page: int = 1, per_page: int = 5) -> list[dict]:
    """"按日期从新到旧分页返回历史快照记录，items 已 json.loads。"""
    conn = connect()
    try:
        with conn.cursor() as cur:
            base = "FROM analysis_log WHERE user_id=%s AND kind=%s"
            args = [user_id, kind]
            if garden_id is not None:
                base += " AND garden_id=%s"
                args.append(garden_id)
            off = max((page - 1), 0) * per_page
            cur.execute("SELECT * " + base + " ORDER BY scoped_date DESC, updated_at DESC LIMIT %s OFFSET %s",
                        args + [per_page, off])
            rows = [dict(r) for r in cur.fetchall()]
        for r in rows:
            try:
                r["_items"] = json.loads(r.get("items_json") or "[]")
            except Exception:
                r["_items"] = []
        return rows
    finally:
        conn.close()


def analysis_count(user_id: int, kind: str, garden_id: int | None = None) -> int:
    conn = connect()
    try:
        with conn.cursor() as cur:
            sql = "SELECT COUNT(*) AS n FROM analysis_log WHERE user_id=%s AND kind=%s"
            args = [user_id, kind]
            if garden_id is not None:
                sql += " AND garden_id=%s"
                args.append(garden_id)
            cur.execute(sql, args)
            return int(cur.fetchone()["n"])
    finally:
        conn.close()


def delete_analysis_record(rid: int, user_id: int) -> None:
    """"删除某条历史记录（只能删自己的）。"""
    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM analysis_log WHERE id=%s AND user_id=%s", (rid, user_id))
        conn.commit()
    finally:
        conn.close()


# ===========================================================================
# 三引擎联动 · 数据打底层（阶段 0）
# ---------------------------------------------------------------------------
# weather_history : 茶园逐日气象归档（供时序特征与 ML 训练用）
# risk_feedback   : 风险推送后的反馈（是否采纳 / 风险是否成真），供闭环重训与 bandit 更新
# ===========================================================================

def _extend_schema() -> None:
    """幂等追加新表（历史气象 + 风险反馈）。"""
    extra = """
CREATE TABLE IF NOT EXISTS weather_history (
    id         INT AUTO_INCREMENT PRIMARY KEY,
    garden_key VARCHAR(64) NOT NULL,
    obs_date   DATE        NOT NULL,
    temp_max   DOUBLE DEFAULT NULL,
    temp_min   DOUBLE DEFAULT NULL,
    humidity   DOUBLE DEFAULT NULL,
    precip     DOUBLE DEFAULT NULL,
    wind_scale DOUBLE DEFAULT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uq_garden_date (garden_key, obs_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS risk_feedback (
    id             INT AUTO_INCREMENT PRIMARY KEY,
    user_id        INT          DEFAULT NULL,
    garden_key     VARCHAR(64)  NOT NULL,
    kind           VARCHAR(16)  NOT NULL,
    risk_name      VARCHAR(64)  DEFAULT NULL,
    predicted_level VARCHAR(16) NOT NULL,
    actual_level   VARCHAR(16)  DEFAULT NULL,
    adopted        TINYINT      DEFAULT 0,
    detail         VARCHAR(600) DEFAULT NULL,
    created_at     DATETIME     DEFAULT CURRENT_TIMESTAMP,
    KEY idx_fb_garden (garden_key)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS model_eval_log (
    id             INT AUTO_INCREMENT PRIMARY KEY,
    garden_key     VARCHAR(64)  DEFAULT NULL,
    eval_type      VARCHAR(32)  DEFAULT 'rolling',
    sample_count   INT          DEFAULT NULL,
    accuracy       DOUBLE       DEFAULT NULL,
    macro_f1       DOUBLE       DEFAULT NULL,
    high_recall    DOUBLE       DEFAULT NULL,
    confusion_json MEDIUMTEXT,
    report_json    MEDIUMTEXT,
    created_at     DATETIME     DEFAULT CURRENT_TIMESTAMP,
    KEY idx_eval_garden (garden_key),
    KEY idx_eval_created (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS sensor_log (
    id          BIGINT AUTO_INCREMENT PRIMARY KEY,
    garden_key  VARCHAR(64)  NOT NULL,
    sensor_type VARCHAR(32)  NOT NULL,
    value       DOUBLE       NOT NULL,
    unit        VARCHAR(16)  DEFAULT '%',
    meta_json   VARCHAR(500) DEFAULT NULL,
    created_at  DATETIME     DEFAULT CURRENT_TIMESTAMP,
    KEY idx_sensor (garden_key, sensor_type, created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS actuator_log (
    id           BIGINT AUTO_INCREMENT PRIMARY KEY,
    garden_key   VARCHAR(64)  NOT NULL,
    device       VARCHAR(32)  NOT NULL,
    operation    VARCHAR(16)  NOT NULL,
    duration_min DOUBLE       DEFAULT NULL,
    reason       VARCHAR(500) DEFAULT NULL,
    state_json   VARCHAR(500) DEFAULT NULL,
    created_at   DATETIME     DEFAULT CURRENT_TIMESTAMP,
    KEY idx_actuator (garden_key, device, created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS agent_decision_log (
    id            BIGINT AUTO_INCREMENT PRIMARY KEY,
    garden_key    VARCHAR(64)  NOT NULL,
    decision_type VARCHAR(32)  NOT NULL,
    summary       VARCHAR(600) DEFAULT NULL,
    inputs_json   MEDIUMTEXT,
    actions_json  MEDIUMTEXT,
    overall_level VARCHAR(16)  DEFAULT 'low',
    created_at    DATETIME     DEFAULT CURRENT_TIMESTAMP,
    KEY idx_decision (garden_key, created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""
    conn = connect()
    try:
        with conn.cursor() as cur:
            for stmt in extra.split(';'):
                stmt = stmt.strip()
                if stmt:
                    cur.execute(stmt)
        conn.commit()
    finally:
        conn.close()


def init_db_ext() -> None:
    """调用方在启动时调用，确保新表存在。"""
    try:
        _extend_schema()
    except Exception:  # noqa: BLE001 - 数据库不可用时静默，游客演示不受影响
        pass


# ----------------------------------------------------------------
# 气象归档
# ----------------------------------------------------------------
def archive_weather(garden_key: str, daily: list[dict]) -> None:
    """把某茶园当天的预报（或历史）写入 weather_history，按 (garden_key,date) 去重 UPSERT。"""
    if not daily:
        return
    conn = connect()
    try:
        with conn.cursor() as cur:
            for d in daily:
                cur.execute(
                    "INSERT INTO weather_history (garden_key,obs_date,temp_max,temp_min,humidity,precip,wind_scale) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s) "
                    "ON DUPLICATE KEY UPDATE temp_max=VALUES(temp_max),temp_min=VALUES(temp_min),"
                    "humidity=VALUES(humidity),precip=VALUES(precip),wind_scale=VALUES(wind_scale)",
                    (
                        garden_key,
                        d.get("fxDate"),
                        float(d.get("tempMax") or 0),
                        float(d.get("tempMin") or 0),
                        float(d.get("humidity") or 0),
                        float(d.get("precip") or 0),
                        float(d.get("windScaleDay") or 0),
                    ),
                )
        conn.commit()
    finally:
        conn.close()


def weather_history_series(garden_key: str, days: int = 90) -> list[dict]:
    """返回该茶园最近 days 天的历史气象记录（按日期升序）。"""
    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT obs_date,temp_max,temp_min,humidity,precip,wind_scale "
                "FROM weather_history WHERE garden_key=%s ORDER BY obs_date DESC LIMIT %s",
                (garden_key, days),
            )
            rows = [dict(r) for r in cur.fetchall()]
        rows.reverse()
        return rows
    finally:
        conn.close()


def weather_history_count(garden_key: str) -> int:
    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS n FROM weather_history WHERE garden_key=%s", (garden_key,))
            return int(cur.fetchone()["n"])
    finally:
        conn.close()


# ----------------------------------------------------------------
# 风险反馈
# ----------------------------------------------------------------
def log_risk_feedback(garden_key: str, kind: str, predicted_level: str,
                      risk_name: str | None = None, actual_level: str | None = None,
                      adopted: bool = False, detail: str | None = None,
                      user_id: int | None = None) -> None:
    """记录一次风险推送的反馈。adopted=农户是否采纳建议，actual_level=事后是否成真。"""
    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO risk_feedback (user_id,garden_key,kind,risk_name,predicted_level,"
                "actual_level,adopted,detail) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                (user_id, garden_key, kind, risk_name, predicted_level, actual_level,
                 int(bool(adopted)), detail),
            )
        conn.commit()
    finally:
        conn.close()


def risk_feedback_stats(garden_key: str, kind: str | None = None) -> dict:
    """按推送等级统计反馈（采纳数/成真数），用于 bandit 计算 reward 的样本。"""
    conn = connect()
    try:
        with conn.cursor() as cur:
            if kind:
                cur.execute(
                    "SELECT predicted_level, COUNT(*) AS n, SUM(adopted) AS adopted, "
                    "SUM(actual_level IS NOT NULL) AS labeled "
                    "FROM risk_feedback WHERE garden_key=%s AND kind=%s GROUP BY predicted_level",
                    (garden_key, kind),
                )
            else:
                cur.execute(
                    "SELECT predicted_level, COUNT(*) AS n, SUM(adopted) AS adopted, "
                    "SUM(actual_level IS NOT NULL) AS labeled "
                    "FROM risk_feedback WHERE garden_key=%s GROUP BY predicted_level",
                    (garden_key,),
                )
            return {
                r["predicted_level"]: {
                    "n": int(r["n"]),
                    "adopted": int(r["adopted"] or 0),
                    "labeled": int(r["labeled"] or 0),
                }
                for r in cur.fetchall()
            }
    finally:
        conn.close()


# ----------------------------------------------------------------
# 模型评估日志（开发者模式回测结果落库）
# ----------------------------------------------------------------
def save_eval_log(garden_key: str | None, eval_type: str, sample_count: int | None,
                  accuracy: float | None, macro_f1: float | None, high_recall: float | None,
                  confusion_json: str | None = None, report_json: str | None = None) -> None:
    """保存一次模型回测评估结果。"""
    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO model_eval_log "
                "(garden_key,eval_type,sample_count,accuracy,macro_f1,high_recall,confusion_json,report_json) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                (garden_key, eval_type, sample_count, accuracy, macro_f1, high_recall,
                 confusion_json, report_json),
            )
        conn.commit()
    finally:
        conn.close()


def eval_log_history(limit: int = 50) -> list[dict]:
    """按时间倒序返回历次模型评估记录。"""
    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id,garden_key,eval_type,sample_count,accuracy,macro_f1,high_recall,"
                "confusion_json,report_json,created_at FROM model_eval_log "
                "ORDER BY created_at DESC LIMIT %s",
                (limit,),
            )
            return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


# ===========================================================================
# 智能执行 Agent · 数据层（阶段 2）
# ---------------------------------------------------------------------------
# sensor_log        : 土壤湿度等传感器读数归档
# actuator_log      : 滴灌阀门/喷杆喷头等执行器动作日志
# agent_decision_log: Agent 每次融合决策的输入与动作快照
# 全部 try/except 降级：数据库不可用时仅打印告警，不影响 Agent 主流程。
# ===========================================================================

def log_sensor(garden_key: str, sensor_type: str, value: float,
               unit: str = "%", meta: dict | None = None) -> None:
    try:
        conn = connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO sensor_log (garden_key,sensor_type,value,unit,meta_json) "
                    "VALUES (%s,%s,%s,%s,%s)",
                    (garden_key, sensor_type, float(value), unit,
                     json.dumps(meta, ensure_ascii=False, default=str) if meta else None),
                )
            conn.commit()
        finally:
            conn.close()
    except Exception as e:  # noqa: BLE001
        print(f"[db] log_sensor 失败: {e}")


def recent_sensor(garden_key: str, sensor_type: str = "soil_moisture",
                  limit: int = 50) -> list[dict]:
    """返回某茶园某传感器最近的读数（时间升序）。"""
    try:
        conn = connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT value,unit,created_at FROM sensor_log "
                    "WHERE garden_key=%s AND sensor_type=%s "
                    "ORDER BY id DESC LIMIT %s",
                    (garden_key, sensor_type, limit),
                )
                rows = [dict(r) for r in cur.fetchall()]
            rows.reverse()
            return rows
        finally:
            conn.close()
    except Exception:  # noqa: BLE001
        return []


def log_actuator(garden_key: str, device: str, operation: str,
                 duration_min: float | None = None, reason: str | None = None,
                 state: dict | None = None) -> None:
    try:
        conn = connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO actuator_log (garden_key,device,operation,duration_min,reason,state_json) "
                    "VALUES (%s,%s,%s,%s,%s,%s)",
                    (garden_key, device, operation, duration_min, reason,
                     json.dumps(state, ensure_ascii=False, default=str) if state else None),
                )
            conn.commit()
        finally:
            conn.close()
    except Exception as e:  # noqa: BLE001
        print(f"[db] log_actuator 失败: {e}")


def actuator_history(garden_key: str, device: str | None = None, limit: int = 50) -> list[dict]:
    try:
        conn = connect()
        try:
            with conn.cursor() as cur:
                if device:
                    cur.execute(
                        "SELECT device,operation,duration_min,reason,created_at FROM actuator_log "
                        "WHERE garden_key=%s AND device=%s ORDER BY id DESC LIMIT %s",
                        (garden_key, device, limit),
                    )
                else:
                    cur.execute(
                        "SELECT device,operation,duration_min,reason,created_at FROM actuator_log "
                        "WHERE garden_key=%s ORDER BY id DESC LIMIT %s",
                        (garden_key, limit),
                    )
                return [dict(r) for r in cur.fetchall()]
        finally:
            conn.close()
    except Exception:  # noqa: BLE001
        return []


def log_agent_decision(garden_key: str, decision_type: str, summary: str | None,
                       inputs: dict | None = None, actions: list | None = None,
                       overall_level: str = "low") -> None:
    try:
        conn = connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO agent_decision_log "
                    "(garden_key,decision_type,summary,inputs_json,actions_json,overall_level) "
                    "VALUES (%s,%s,%s,%s,%s,%s)",
                    (garden_key, decision_type, summary,
                     json.dumps(inputs, ensure_ascii=False, default=str) if inputs else None,
                     json.dumps(actions, ensure_ascii=False, default=str) if actions else None,
                     overall_level),
                )
            conn.commit()
        finally:
            conn.close()
    except Exception as e:  # noqa: BLE001
        print(f"[db] log_agent_decision 失败: {e}")


def agent_decision_history(garden_key: str, limit: int = 30) -> list[dict]:
    """按时间倒序返回 Agent 决策记录，inputs/actions 已解析为 dict。"""
    try:
        conn = connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT decision_type,summary,inputs_json,actions_json,overall_level,created_at "
                    "FROM agent_decision_log WHERE garden_key=%s ORDER BY id DESC LIMIT %s",
                    (garden_key, limit),
                )
                rows = [dict(r) for r in cur.fetchall()]
            for r in rows:
                for k, jk in (("inputs_json", "inputs"), ("actions_json", "actions")):
                    try:
                        r[k] = json.loads(r.get(jk) or "null")
                    except Exception:  # noqa: BLE001
                        r[k] = None
            return rows
        finally:
            conn.close()
    except Exception:  # noqa: BLE001
        return []
