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
