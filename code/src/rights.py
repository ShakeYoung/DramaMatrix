"""Rights & ownership declarations (W3).

投放包走出工作室之前必须能回答"素材从哪来、有什么权利"。本模块：

- 配置：DRAMAMATRIX_SOURCE_LICENSE（owned/licensed/public-domain/unknown）、
  DRAMAMATRIX_SOURCE_OWNER（权利人/授权方）、DRAMAMATRIX_SOURCE_NOTE（授权
  范围/到期日等备注）。未配置时 license 记为 unknown 并在投放包中显式提示，
  而不是默默缺失。
- 落库：Agent1 选定源头素材后 record_source_rights 写入 rights_records
  （按 project+scope+ref_id 幂等），进入证据链。
- 出包：publish.py 的 publish_meta.json 携带 rights 块（来源授权 + 生成物
  权利声明 DRAMAMATRIX_ASSET_RIGHTS_NOTE）。
"""

from __future__ import annotations

import os
import sqlite3
import time

VALID_LICENSES = {"owned", "licensed", "public-domain", "unknown"}


def rights_config() -> dict:
    license_value = os.getenv("DRAMAMATRIX_SOURCE_LICENSE", "").strip().lower()
    if license_value not in VALID_LICENSES:
        license_value = "unknown"
    return {
        "license": license_value,
        "owner": os.getenv("DRAMAMATRIX_SOURCE_OWNER", "").strip(),
        "note": os.getenv("DRAMAMATRIX_SOURCE_NOTE", "").strip(),
    }


def record_source_rights(project_id: str, title: str) -> None:
    """记录源头素材的权属声明（幂等：同 project+source+title 只写一次）。"""
    config = rights_config()
    import src.db as db_module

    conn = sqlite3.connect(db_module.DB_PATH)
    try:
        existing = conn.execute(
            "SELECT 1 FROM rights_records WHERE project_id = ? AND scope = 'source' AND ref_id = ?",
            (project_id, title),
        ).fetchone()
        if existing:
            return
        conn.execute(
            "INSERT INTO rights_records (project_id, scope, ref_id, license, owner, note, created_at_unix)"
            " VALUES (?, 'source', ?, ?, ?, ?, ?)",
            (project_id, title, config["license"], config["owner"], config["note"], time.time()),
        )
        conn.commit()
        if config["license"] == "unknown":
            print(f"   ⚠️ 《{title}》未配置权属声明（DRAMAMATRIX_SOURCE_LICENSE），已记为 unknown。")
    finally:
        conn.close()


def rights_block(project_id: str | None = None) -> dict:
    """投放包携带的权属块：来源授权 + 生成物权利声明。

    project_id 给定时，从 rights_records 取该项目最近一条 source 声明补全
    书名/授权信息；查不到或未给 project_id 时仅用环境配置。
    """
    config = rights_config()
    source_title = ""
    if project_id:
        try:
            import src.db as db_module

            conn = sqlite3.connect(db_module.DB_PATH)
            try:
                row = conn.execute(
                    "SELECT ref_id, license, owner, note FROM rights_records"
                    " WHERE project_id = ? AND scope = 'source'"
                    " ORDER BY id DESC LIMIT 1",
                    (project_id,),
                ).fetchone()
            finally:
                conn.close()
            if row:
                source_title = row[0] or ""
                # 落库声明优先于环境缺省（例如库里有 licensed 而环境未配）。
                config = {
                    "license": row[1] or config["license"],
                    "owner": row[2] or config["owner"],
                    "note": row[3] or config["note"],
                }
        except sqlite3.Error:
            pass
    return {
        "source": {
            "title": source_title,
            "license": config["license"],
            "owner": config["owner"],
            "note": config["note"],
        },
        "generated_assets": {
            "license": os.getenv("DRAMAMATRIX_ASSET_LICENSE", "studio-owned").strip() or "studio-owned",
            "note": os.getenv("DRAMAMATRIX_ASSET_RIGHTS_NOTE", "").strip(),
            "disclosure": (
                "本包内视频/封面由 AI 生成流水线产出；请按所在平台要求声明 AI 生成内容。"
            ),
        },
    }
