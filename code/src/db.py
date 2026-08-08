import sqlite3
import os
import json
from typing import Any

from pydantic import BaseModel

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dramamatrix.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
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
    ]:
        try:
            cursor.execute(f"ALTER TABLE agnes_usage ADD COLUMN {column} {definition}")
        except sqlite3.OperationalError:
            pass  # 列已存在
    conn.commit()
    conn.close()

def db_insert_novel(title, url, tags, content):
    conn = sqlite3.connect(DB_PATH)
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
    conn = sqlite3.connect(DB_PATH)
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
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("UPDATE novels SET status = ? WHERE title = ?", (status, title))
    conn.commit()
    conn.close()


def db_insert_run_log(project_id, node, system_status, cycle=None, event="transition", level="INFO"):
    conn = sqlite3.connect(DB_PATH)
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
    conn = sqlite3.connect(DB_PATH)
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
    conn = sqlite3.connect(DB_PATH)
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
    conn = sqlite3.connect(DB_PATH)
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
    conn = sqlite3.connect(DB_PATH)
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
    conn = sqlite3.connect(DB_PATH)
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
    conn = sqlite3.connect(DB_PATH)
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
) -> bool:
    """Persist one billed create idempotently. Returns True only for a new task.

    F3：扩展字段在创建/轮询/下载各阶段采集后由 db_update_agnes_usage_timing 更新。
    """
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.execute(
            """
            INSERT OR IGNORE INTO agnes_usage
                (task_id, project_id, ep_key, shot_id, frames, width, height, created_at_unix,
                 queue_wait_seconds, render_seconds, download_seconds, redraw_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (task_id, project_id, ep_key, shot_id, frames, width, height, created_at_unix,
             queue_wait_seconds, render_seconds, download_seconds, redraw_count),
        )
        conn.commit()
        return cursor.rowcount == 1
    finally:
        conn.close()


def db_update_agnes_usage_timing(task_id: str, *, render_seconds=None, download_seconds=None, redraw_count=None):
    """F3：按 task_id 更新耗时/重绘字段（轮询与下载完成后调用）。"""
    conn = sqlite3.connect(DB_PATH)
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
    conn = sqlite3.connect(DB_PATH)
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM agnes_usage WHERE project_id = ?",
            (project_id,),
        ).fetchone()
        return int(row[0] if row else 0)
    finally:
        conn.close()


def _json_default(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def db_save_project_state(state: dict[str, Any]) -> None:
    """Persist state and media metadata without persisting any API credential."""
    project_id = state.get("project_id")
    if not project_id:
        raise ValueError("project_id is required to save a project state snapshot")
    payload = json.dumps(state, ensure_ascii=False, default=_json_default)
    conn = sqlite3.connect(DB_PATH)
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
            (project_id, state.get("system_status", "unknown"), payload),
        )
        # V4：同时向 state_history 追加一行（带 run_id + 自增 version），支持版本回溯。
        run_id = state.get("run_context", {}).get("run_id") if isinstance(state.get("run_context"), dict) else None
        run_context_json = json.dumps(state.get("run_context"), ensure_ascii=False, default=_json_default) if state.get("run_context") else None
        version_row = conn.execute(
            "SELECT COALESCE(MAX(version), 0) + 1 FROM state_history WHERE project_id = ?",
            (project_id,),
        ).fetchone()
        next_version = int(version_row[0]) if version_row else 1
        conn.execute(
            """INSERT INTO state_history
            (project_id, run_id, version, system_status, state_json, run_context_json)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (project_id, run_id, next_version, state.get("system_status", "unknown"), payload, run_context_json),
        )
        conn.commit()
    finally:
        conn.close()


def db_get_project_state_snapshot(project_id: str) -> dict[str, Any] | None:
    conn = sqlite3.connect(DB_PATH)
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
