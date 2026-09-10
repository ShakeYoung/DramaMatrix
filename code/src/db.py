import sqlite3
import os
import json
from typing import Any

from pydantic import BaseModel

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dramamatrix.db")

def _connect() -> sqlite3.Connection:
    """统一连接入口：busy_timeout 避免运行中的只读方（dashboard/review）触发
    database is locked；WAL 为持久化设置，失败（个别文件系统不支持）不影响功能。
    动态读取模块级 DB_PATH，测试可通过替换该属性隔离数据库。
    """
    conn = sqlite3.connect(DB_PATH, timeout=5.0)
    conn.execute("PRAGMA busy_timeout = 5000")
    try:
        conn.execute("PRAGMA journal_mode = WAL")
    except sqlite3.OperationalError:
        pass
    return conn

def init_db():
    conn = _connect()
    cursor = conn.cursor()
    
    # Table for scraped novels
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS novels (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL UNIQUE,
        url TEXT,
        tags TEXT,
        content TEXT,
        status TEXT DEFAULT 'pending',
        scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS project_state_snapshots (
        project_id TEXT PRIMARY KEY,
        system_status TEXT NOT NULL,
        state_json TEXT NOT NULL,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Table for analytics / market feedback
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS analytics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ep_id TEXT,
        views INTEGER,
        cpa REAL,
        completion_rate REAL,
        tags TEXT,
        recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    # Structured run log (阶段4): one row per node transition per run
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS run_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id TEXT,
        node TEXT,
        system_status TEXT,
        cycle INTEGER,
        event TEXT,
        level TEXT DEFAULT 'INFO',
        recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    # Agnes 创建事实以 task_id 幂等落库。CSV 只作为可读报表，不能承担预算真相源；
    # 即使输出目录被移动/删除，项目创建预算仍可正确恢复。
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS agnes_usage (
        task_id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL,
        ep_key TEXT,
        shot_id TEXT,
        frames INTEGER,
        width INTEGER,
        height INTEGER,
        created_at_unix REAL,
        recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_agnes_usage_project ON agnes_usage(project_id)"
    )
    # V2：连续性质检结果沉淀为结构化实验数据（支持跨镜头/跨集对比分析）
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS shot_qc_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id TEXT NOT NULL,
        ep_key TEXT,
        shot_id TEXT,
        storyboard_version INTEGER,
        redraw_attempt INTEGER,
        passed INTEGER,
        brightness_diff REAL,
        threshold REAL,
        issues TEXT,
        metrics TEXT,
        prev_tail_sha256 TEXT,
        curr_head_sha256 TEXT,
        qc_version TEXT,
        recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_qc_project_ep ON shot_qc_results(project_id, ep_key)"
    )
    # V4：状态历史（追加式，支持版本回溯与审计）
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS state_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id TEXT NOT NULL,
        run_id TEXT,
        version INTEGER,
        system_status TEXT,
        state_json TEXT NOT NULL,
        run_context_json TEXT,
        recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_state_history_project ON state_history(project_id, version)"
    )
    # V5：人工评分/标注集
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS manual_scores (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id TEXT NOT NULL,
        ep_key TEXT,
        shot_id TEXT,
        scored_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        scorer TEXT,
        rubric_version TEXT DEFAULT 'v1',
        round INTEGER DEFAULT 1,
        video_sha256 TEXT,
        character_consistency REAL,
        wardrobe_consistency REAL,
        action_continuity REAL,
        subtitle_alignment REAL,
        voice_alignment REAL,
        overall REAL,
        notes TEXT,
        UNIQUE(project_id, ep_key, shot_id, scorer, round)
    )
    ''')
    # V5：参考资产引用链（哪一镜引用了哪张角色/场景/尾帧参考图）
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS reference_assets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id TEXT NOT NULL,
        asset_type TEXT,
        ref_id TEXT,
        local_path TEXT,
        sha256 TEXT,
        referenced_by_shot TEXT,
        recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    # V5：agnes_usage 扩列（幂等，兼容老库）
    for column, definition in [
        ("queue_wait_seconds", "REAL"),
        ("render_seconds", "REAL"),
        ("download_seconds", "REAL"),
        ("redraw_count", "INTEGER DEFAULT 0"),
        ("cost_estimate", "REAL"),
        ("currency", "TEXT"),
        ("quality_score", "REAL"),
        ("provider", "TEXT"),  # E5：记录使用哪个视频供应商
    ]:
        try:
            cursor.execute(f"ALTER TABLE agnes_usage ADD COLUMN {column} {definition}")
        except sqlite3.OperationalError:
            pass  # 列已存在
    # U6：先验知识库——Agent2 的 RAG 检索源（可运营增补，非代码内置）。
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS knowledge_entries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        category TEXT DEFAULT 'prior',
        title TEXT,
        content TEXT NOT NULL,
        tags TEXT DEFAULT '',
        created_at_unix REAL,
        recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    # W3：版权与素材权属声明（来源授权 / 生成物权利），投放包出工作室前留痕。
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS rights_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id TEXT NOT NULL,
        scope TEXT,
        ref_id TEXT,
        license TEXT,
        owner TEXT,
        note TEXT,
        created_at_unix REAL,
        recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    # R1：analytics 来源与去重列（幂等，兼容老库）——project_id/platform 标记归属，
    # source 区分 imported（真实导入）/ simulated（演示模拟）/ legacy_unverified（旧版存量），
    # dedup_key 唯一索引保证同一导入文件重复执行不重复插入。
    for column, definition in [
        ("project_id", "TEXT"),
        ("platform", "TEXT"),
        ("source", "TEXT"),
        ("dedup_key", "TEXT"),
    ]:
        try:
            cursor.execute(f"ALTER TABLE analytics ADD COLUMN {column} {definition}")
        except sqlite3.OperationalError:
            pass  # 列已存在
    # 旧版写入的行没有 source：标记为未验证，市场推荐查询会排除它们。
    cursor.execute(
        "UPDATE analytics SET source = 'legacy_unverified', dedup_key = 'legacy-' || id "
        "WHERE source IS NULL OR source = ''"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_analytics_project ON analytics(project_id)"
    )
    try:
        cursor.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_analytics_dedup ON analytics(dedup_key)"
        )
    except sqlite3.OperationalError:
        pass
    # R3：金额成本账本——按项目/集/镜头记录各阶段（video/tts/image/…）费用。
    # kind 生命周期：estimate（分镜后预估）→ reserved（付费请求已提交）→
    # confirmed（资产落地确认）；confirmed 由 UPDATE 升级，不重复计费。
    # attempt 记录同一镜头第几次付费创建（>1 即重绘），供重绘占比分析。
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS cost_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id TEXT NOT NULL,
        ep_key TEXT,
        shot_id TEXT,
        stage TEXT NOT NULL,
        kind TEXT NOT NULL,
        amount REAL NOT NULL,
        currency TEXT DEFAULT 'CNY',
        estimated INTEGER DEFAULT 1,
        attempt INTEGER DEFAULT 1,
        provider TEXT,
        ref_id TEXT,
        note TEXT,
        confirmed_at_unix REAL,
        created_at_unix REAL,
        recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_cost_events_project ON cost_events(project_id, ep_key)"
    )
    # 幂等：同一笔费用（项目+阶段+kind+ref）重复提交不重复入账。
    try:
        cursor.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_cost_events_ref "
            "ON cost_events(project_id, stage, kind, ref_id) WHERE ref_id IS NOT NULL"
        )
    except sqlite3.OperationalError:
        pass
    # P0-1：迁移旧版 manual_scores——补列、建唯一索引、清理重复数据。
    # CREATE TABLE IF NOT EXISTS 不会迁移已有表；旧库缺 rubric_version/round/video_sha256
    # 且无唯一约束，新版插入会报 "no column named rubric_version"。
    _migrate_manual_scores(cursor)
    conn.commit()
    conn.close()


def _migrate_manual_scores(cursor) -> None:
    """P0-1：把旧版 manual_scores 迁移到新版结构（幂等）。

    - 补列：rubric_version / round / video_sha256
    - 为已有行填充默认值（rubric_version='v1', round=1）
    - 清理 (project_id, ep_key, shot_id, scorer) 重复行（保留最新一条）
    - 建唯一索引（旧表无 UNIQUE 约束，用唯一索引实现同等去重）
    """
    existing = {row[1] for row in cursor.execute("PRAGMA table_info(manual_scores)").fetchall()}
    for column, definition, default in [
        ("rubric_version", "TEXT DEFAULT 'v1'", "'v1'"),
        ("round", "INTEGER DEFAULT 1", "1"),
        ("video_sha256", "TEXT", None),
    ]:
        if column not in existing:
            cursor.execute(f"ALTER TABLE manual_scores ADD COLUMN {column} {definition}")
            if default is not None:
                cursor.execute(f"UPDATE manual_scores SET {column} = {default} WHERE {column} IS NULL")
    # 清理重复评分（同一 project/ep/shot/scorer/round 保留最新 id），仅当存在重复时执行。
    cursor.execute(
        """
        DELETE FROM manual_scores WHERE id NOT IN (
            SELECT MAX(id) FROM manual_scores
            GROUP BY project_id, ep_key, shot_id, scorer, round
        )
        """
    )
    # 唯一索引（与 UPSERT 冲突目标一致：含 round，否则同 scorer 不同轮次的
    # 评分会被误去重，且与 ON CONFLICT(...,round) 语义冲突）。
    # 若存在旧版不含 round 的错误索引，先删除再重建。
    cursor.execute("DROP INDEX IF EXISTS uq_manual_scores_dedup")
    try:
        cursor.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_manual_scores_dedup "
            "ON manual_scores(project_id, ep_key, shot_id, scorer, round)"
        )
    except sqlite3.OperationalError:
        pass

def db_insert_novel(title, url, tags, content):
    conn = _connect()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO novels (title, url, tags, content) VALUES (?, ?, ?, ?)",
            (title, url, tags, content)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        # Title already exists
        return False
    finally:
        conn.close()

def db_get_unprocessed_novel(suggested_tags=None):
    conn = _connect()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    if suggested_tags:
        # Tag strings are persisted as JSON. Parameter binding avoids SQL injection.
        query_conditions = " OR ".join("tags LIKE ?" for _ in suggested_tags)
        parameters = [f"%{tag}%" for tag in suggested_tags]
        cursor.execute(
            f"SELECT * FROM novels WHERE status = 'pending' AND ({query_conditions}) LIMIT 1",
            parameters,
        )
        row = cursor.fetchone()
        if row:
            conn.close()
            return dict(row)
            
    cursor.execute("SELECT * FROM novels WHERE status = 'pending' ORDER BY scraped_at DESC LIMIT 1")
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None
    
def db_mark_novel_processed(title, status="processed"):
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute("UPDATE novels SET status = ? WHERE title = ?", (status, title))
    conn.commit()
    conn.close()


def db_insert_run_log(project_id, node, system_status, cycle=None, event="transition", level="INFO"):
    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO run_logs (project_id, node, system_status, cycle, event, level) VALUES (?, ?, ?, ?, ?, ?)",
            (project_id, node, system_status, cycle, event, level),
        )
        conn.commit()
    finally:
        conn.close()


def db_insert_qc_result(project_id, ep_key, shot_id, passed, brightness_diff=None,
                        threshold=None, issues=None, metrics=None, storyboard_version=None,
                        redraw_attempt=None, prev_tail_sha256=None, curr_head_sha256=None,
                        qc_version="default"):
    """V2：沉淀一镜的连续性质检结果为结构化实验数据。"""
    conn = _connect()
    try:
        conn.execute(
            """INSERT INTO shot_qc_results
            (project_id, ep_key, shot_id, storyboard_version, redraw_attempt,
             passed, brightness_diff, threshold, issues, metrics,
             prev_tail_sha256, curr_head_sha256, qc_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (project_id, ep_key, shot_id, storyboard_version, redraw_attempt,
             1 if passed else 0, brightness_diff, threshold,
             json.dumps(issues, ensure_ascii=False) if issues else None,
             json.dumps(metrics, ensure_ascii=False) if metrics else None,
             prev_tail_sha256, curr_head_sha256, qc_version),
        )
        conn.commit()
    finally:
        conn.close()


def db_query_qc_results(project_id, ep_key=None, limit=500):
    """V2：查询质检结果（可按集过滤），返回 dict 列表。"""
    conn = _connect()
    conn.row_factory = sqlite3.Row
    try:
        if ep_key:
            rows = conn.execute(
                "SELECT * FROM shot_qc_results WHERE project_id=? AND ep_key=? ORDER BY recorded_at DESC LIMIT ?",
                (project_id, ep_key, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM shot_qc_results WHERE project_id=? ORDER BY recorded_at DESC LIMIT ?",
                (project_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def db_get_state_history(project_id, limit=50):
    """V4：获取项目的状态历史（追加式），最新在前。"""
    conn = _connect()
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT * FROM state_history WHERE project_id=? ORDER BY version DESC LIMIT ?",
            (project_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def db_insert_manual_score(project_id, ep_key=None, shot_id=None, scorer=None,
                           character_consistency=None, wardrobe_consistency=None,
                           action_continuity=None, subtitle_alignment=None,
                           voice_alignment=None, overall=None, notes=None,
                           rubric_version="v1", round_number=1, video_sha256=None):
    """V5/F5：插入一条人工评分（UPSERT 去重：同 project/ep/shot/scorer/round 覆盖）。

    评分范围约束：0–5（可为小数），越界抛 ValueError 拒绝写入。
    """
    for label, value in [("character_consistency", character_consistency),
                         ("wardrobe_consistency", wardrobe_consistency),
                         ("action_continuity", action_continuity),
                         ("subtitle_alignment", subtitle_alignment),
                         ("voice_alignment", voice_alignment),
                         ("overall", overall)]:
        if value is not None and not (0 <= float(value) <= 5):
            raise ValueError(f"{label} 评分 {value} 超出 0–5 范围")
    conn = _connect()
    try:
        conn.execute(
            """INSERT INTO manual_scores
            (project_id, ep_key, shot_id, scorer, rubric_version, round, video_sha256,
             character_consistency, wardrobe_consistency, action_continuity,
             subtitle_alignment, voice_alignment, overall, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(project_id, ep_key, shot_id, scorer, round) DO UPDATE SET
                rubric_version = excluded.rubric_version,
                video_sha256 = excluded.video_sha256,
                character_consistency = excluded.character_consistency,
                wardrobe_consistency = excluded.wardrobe_consistency,
                action_continuity = excluded.action_continuity,
                subtitle_alignment = excluded.subtitle_alignment,
                voice_alignment = excluded.voice_alignment,
                overall = excluded.overall,
                notes = excluded.notes,
                scored_at = CURRENT_TIMESTAMP
            """,
            (project_id, ep_key, shot_id, scorer, rubric_version, round_number, video_sha256,
             character_consistency, wardrobe_consistency, action_continuity,
             subtitle_alignment, voice_alignment, overall, notes),
        )
        conn.commit()
    finally:
        conn.close()


def db_export_manual_scores(project_id, csv_path):
    """V5：导出人工评分为 CSV（便于离线标注与回填）。"""
    import csv as _csv
    conn = _connect()
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT * FROM manual_scores WHERE project_id=? ORDER BY scored_at", (project_id,)
        ).fetchall()
    finally:
        conn.close()
    fieldnames = ["id", "project_id", "ep_key", "shot_id", "scored_at", "scorer",
                  "rubric_version", "round", "video_sha256",
                  "character_consistency", "wardrobe_consistency", "action_continuity",
                  "subtitle_alignment", "voice_alignment", "overall", "notes"]
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = _csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))
    return csv_path


def db_import_manual_scores(csv_path):
    """V5/F5：从 CSV 导入人工评分（列须含 project_id；其余评分列可选）。

    UPSERT 去重（同 project/ep/shot/scorer/round 覆盖），评分越界抛 ValueError。
    """
    import csv as _csv
    inserted = 0
    with open(csv_path, "r", newline="", encoding="utf-8") as handle:
        reader = _csv.DictReader(handle)
        for row in reader:
            db_insert_manual_score(
                project_id=row["project_id"],
                ep_key=row.get("ep_key"),
                shot_id=row.get("shot_id"),
                scorer=row.get("scorer"),
                character_consistency=_to_float(row.get("character_consistency")),
                wardrobe_consistency=_to_float(row.get("wardrobe_consistency")),
                action_continuity=_to_float(row.get("action_continuity")),
                subtitle_alignment=_to_float(row.get("subtitle_alignment")),
                voice_alignment=_to_float(row.get("voice_alignment")),
                overall=_to_float(row.get("overall")),
                notes=row.get("notes"),
                rubric_version=row.get("rubric_version") or "v1",
                round_number=int(row["round"]) if (row.get("round") or "").isdigit() else 1,
                video_sha256=row.get("video_sha256"),
            )
            inserted += 1
    return inserted


def _to_float(value):
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def db_insert_reference_asset(project_id, asset_type, ref_id, local_path=None,
                              sha256=None, referenced_by_shot=None):
    """V5：记录参考资产引用关系（角色/场景/尾帧参考图）。"""
    conn = _connect()
    try:
        conn.execute(
            """INSERT INTO reference_assets
            (project_id, asset_type, ref_id, local_path, sha256, referenced_by_shot)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (project_id, asset_type, ref_id, local_path, sha256, referenced_by_shot),
        )
        conn.commit()
    finally:
        conn.close()


def db_record_agnes_usage(
    *,
    task_id: str,
    project_id: str,
    ep_key: str,
    shot_id: str,
    frames: int,
    width: int,
    height: int,
    created_at_unix: float | None = None,
    queue_wait_seconds: float | None = None,
    render_seconds: float | None = None,
    download_seconds: float | None = None,
    redraw_count: int = 0,
    provider: str | None = None,
) -> bool:
    """Persist one billed create idempotently. Returns True only for a new task.

    F3：扩展字段在创建/轮询/下载各阶段采集后由 db_update_agnes_usage_timing 更新。
    """
    conn = _connect()
    try:
        cursor = conn.execute(
            """
            INSERT OR IGNORE INTO agnes_usage
                (task_id, project_id, ep_key, shot_id, frames, width, height, created_at_unix,
                 queue_wait_seconds, render_seconds, download_seconds, redraw_count, provider)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (task_id, project_id, ep_key, shot_id, frames, width, height, created_at_unix,
             queue_wait_seconds, render_seconds, download_seconds, redraw_count, provider),
        )
        conn.commit()
        return cursor.rowcount == 1
    finally:
        conn.close()


def db_update_agnes_usage_timing(task_id: str, *, render_seconds=None, download_seconds=None, redraw_count=None):
    """F3：按 task_id 更新耗时/重绘字段（轮询与下载完成后调用）。"""
    conn = _connect()
    try:
        sets = []
        params = []
        if render_seconds is not None:
            sets.append("render_seconds = ?")
            params.append(render_seconds)
        if download_seconds is not None:
            sets.append("download_seconds = ?")
            params.append(download_seconds)
        if redraw_count is not None:
            sets.append("redraw_count = ?")
            params.append(redraw_count)
        if not sets:
            return
        params.append(task_id)
        conn.execute(f"UPDATE agnes_usage SET {', '.join(sets)} WHERE task_id = ?", params)
        conn.commit()
    finally:
        conn.close()


def db_count_agnes_usage(project_id: str) -> int:
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM agnes_usage WHERE project_id = ?",
            (project_id,),
        ).fetchone()
        return int(row[0] if row else 0)
    finally:
        conn.close()


def db_insert_cost_event(
    *,
    project_id: str,
    stage: str,
    kind: str,
    amount: float,
    ep_key: str | None = None,
    shot_id: str | None = None,
    currency: str = "CNY",
    estimated: bool = True,
    attempt: int = 1,
    provider: str | None = None,
    ref_id: str | None = None,
    note: str | None = None,
    created_at_unix: float | None = None,
) -> bool:
    """R3：写入一笔成本事件（幂等：同 project+stage+kind+ref 只记一次）。

    Returns True 表示新写入，False 表示已存在（重复提交被忽略）。
    """
    import time as _time
    conn = _connect()
    try:
        cursor = conn.execute(
            """INSERT OR IGNORE INTO cost_events
            (project_id, ep_key, shot_id, stage, kind, amount, currency,
             estimated, attempt, provider, ref_id, note, created_at_unix)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (project_id, ep_key, shot_id, stage, kind, float(amount), currency,
             1 if estimated else 0, attempt, provider, ref_id, note,
             created_at_unix if created_at_unix is not None else _time.time()),
        )
        conn.commit()
        return cursor.rowcount == 1
    finally:
        conn.close()


def db_confirm_cost_event(project_id: str, stage: str, ref_id: str) -> bool:
    """R3：把一笔 reserved 费用升级为 confirmed（资产落地）。

    Returns True 表示发生了升级；False 表示不存在该预留或已是 confirmed。
    """
    import time as _time
    conn = _connect()
    try:
        cursor = conn.execute(
            """UPDATE cost_events SET kind = 'confirmed',
               confirmed_at_unix = ?, estimated = 0
            WHERE project_id = ? AND stage = ? AND ref_id = ? AND kind = 'reserved'""",
            (_time.time(), project_id, stage, ref_id),
        )
        conn.commit()
        return cursor.rowcount == 1
    finally:
        conn.close()


def db_query_cost_events(project_id: str, ep_key: str | None = None) -> list[dict]:
    """R3：查询项目的成本事件（可按集过滤），返回 dict 列表。"""
    conn = _connect()
    conn.row_factory = sqlite3.Row
    try:
        if ep_key:
            rows = conn.execute(
                "SELECT * FROM cost_events WHERE project_id=? AND ep_key=? ORDER BY id",
                (project_id, ep_key),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM cost_events WHERE project_id=? ORDER BY id",
                (project_id,),
            ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def db_cost_spent(project_id: str, ep_key: str | None = None) -> float:
    """R3：已花费金额 = reserved（未确认）+ confirmed（已确认）之和。

    estimate 事件是分镜阶段的预估记录，不计入花费。
    """
    conn = _connect()
    try:
        if ep_key:
            row = conn.execute(
                "SELECT COALESCE(SUM(amount), 0) FROM cost_events "
                "WHERE project_id=? AND ep_key=? AND kind IN ('reserved','confirmed')",
                (project_id, ep_key),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT COALESCE(SUM(amount), 0) FROM cost_events "
                "WHERE project_id=? AND kind IN ('reserved','confirmed')",
                (project_id,),
            ).fetchone()
        return float(row[0] if row else 0.0)
    finally:
        conn.close()


def _json_default(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def db_save_project_state(state: dict[str, Any]) -> None:
    """Persist state and media metadata without persisting any API credential.

    R1 合并语义（按集运行保护）：传入 episodes 与快照中已有 episodes 按键合并，
    传入方优先（同 key 覆盖）；快照中独有、本次运行未携带的集（例如
    DRAMAMATRIX_EPISODE 按集限定运行）原样保留——保存永不删除快照中已有的集。
    项目运行锁保证同一时刻只有一个写者，合并在进程内安全。
    """
    project_id = state.get("project_id")
    if not project_id:
        raise ValueError("project_id is required to save a project state snapshot")
    state_to_save = _merge_episodes_with_snapshot(project_id, state)
    payload = json.dumps(state_to_save, ensure_ascii=False, default=_json_default)
    conn = _connect()
    try:
        conn.execute(
            """
            INSERT INTO project_state_snapshots (project_id, system_status, state_json, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(project_id) DO UPDATE SET
                system_status = excluded.system_status,
                state_json = excluded.state_json,
                updated_at = CURRENT_TIMESTAMP
            """,
            (project_id, state_to_save.get("system_status", "unknown"), payload),
        )
        # V4：同时向 state_history 追加一行（带 run_id + 自增 version），支持版本回溯。
        run_id = state_to_save.get("run_context", {}).get("run_id") if isinstance(state_to_save.get("run_context"), dict) else None
        run_context_json = json.dumps(state_to_save.get("run_context"), ensure_ascii=False, default=_json_default) if state_to_save.get("run_context") else None
        version_row = conn.execute(
            "SELECT COALESCE(MAX(version), 0) + 1 FROM state_history WHERE project_id = ?",
            (project_id,),
        ).fetchone()
        next_version = int(version_row[0]) if version_row else 1
        conn.execute(
            """INSERT INTO state_history
            (project_id, run_id, version, system_status, state_json, run_context_json)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (project_id, run_id, next_version, state_to_save.get("system_status", "unknown"), payload, run_context_json),
        )
        conn.commit()
    finally:
        conn.close()


def _merge_episodes_with_snapshot(project_id: str, state: dict[str, Any]) -> dict[str, Any]:
    """按集合并：返回补齐了快照中未被本次运行携带的集之后的状态（不修改入参）。

    快照缺失/损坏、或传入方携带全部集时原样返回（快路径，零拷贝）。
    """
    incoming = state.get("episodes")
    if not isinstance(incoming, dict) or not incoming:
        return state
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT state_json FROM project_state_snapshots WHERE project_id = ?",
            (project_id,),
        ).fetchone()
    finally:
        conn.close()
    if not row or not row[0]:
        return state
    try:
        persisted = json.loads(row[0])
    except (TypeError, ValueError):
        return state
    persisted_episodes = persisted.get("episodes") if isinstance(persisted, dict) else None
    if not isinstance(persisted_episodes, dict):
        return state
    missing = {k: v for k, v in persisted_episodes.items() if k not in incoming}
    if not missing:
        return state
    merged = dict(state)
    merged["episodes"] = {**missing, **incoming}
    return merged


def db_get_project_state_snapshot(project_id: str) -> dict[str, Any] | None:
    conn = _connect()
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT project_id, system_status, state_json, updated_at FROM project_state_snapshots WHERE project_id = ?",
            (project_id,),
        ).fetchone()
        if not row:
            return None
        result = dict(row)
        result["state"] = json.loads(result.pop("state_json"))
        return result
    finally:
        conn.close()

init_db()
