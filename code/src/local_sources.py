"""Real operator-provided data sources (W2) — 合规替换 mock。

此前入口（Agent1 爬虫=3 本硬编码书目）与出口（Agent8 random.uniform 假投放
数据）均为模拟。为避免爬取盗版站点/伪造平台数据的合规风险，真实数据改为
"运营自备文件"：

- 本地小说库：DRAMAMATRIX_LOCAL_NOVEL_DIR 指向 *.txt 目录。文件名即书名；
  首行可写元数据 `# tags: 男频,玄幻,复仇`，其余为正文。
- 投放数据回流：DRAMAMATRIX_ANALYTICS_IMPORT 指向平台导出的 CSV/JSON。
  CSV 列：ep_id,views,cpa,completion_rate,tags（tags 用分号或竖线分隔；
  platform 为可选列，如 douyin/kuaishou）；JSON 为同字段对象数组。

production 模式下两者未配置都会阻塞对应阶段（缺书源/等数据）；
demo 模式（DRAMAMATRIX_RUN_MODE=demo）保留原有 mock 行为且输出带模拟标记。
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Optional


def load_local_novel(exclude_titles: Optional[set[str]] = None) -> Optional[dict]:
    """从 DRAMAMATRIX_LOCAL_NOVEL_DIR 选取一本未尝试过的本地小说。

    Returns {"title", "tags", "content", "url"}；目录未配置/无可用文件返回 None。
    """
    directory = os.getenv("DRAMAMATRIX_LOCAL_NOVEL_DIR", "").strip()
    if not directory:
        return None
    root = Path(directory)
    if not root.is_dir():
        print(f"   ⚠️ DRAMAMATRIX_LOCAL_NOVEL_DIR 不是有效目录：{directory}")
        return None
    excluded = set(exclude_titles or [])
    for path in sorted(root.glob("*.txt")):
        title = path.stem.strip()
        if not title or title in excluded:
            continue
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            continue
        lines = text.splitlines()
        tags: list[str] = []
        body_start = 0
        if lines and lines[0].strip().startswith("#"):
            # 元数据行：# tags: 男频,玄幻   （也接受 # tag: 或无空格）
            meta = lines[0].strip().lstrip("#").strip()
            if meta.lower().startswith("tag"):
                meta = meta.split(":", 1)[-1]
            tags = [t.strip() for t in meta.replace("，", ",").split(",") if t.strip()]
            body_start = 1
        content = "\n".join(lines[body_start:]).strip()
        if not content:
            continue
        return {
            "title": title,
            "tags": tags or ["未分类"],
            "content": content,
            "url": f"file://{path.resolve()}",
        }
    print("   ⚠️ 本地小说库中没有未尝试过的 txt 文件。")
    return None


def load_analytics_records() -> Optional[list[dict]]:
    """读取 DRAMAMATRIX_ANALYTICS_IMPORT 指向的投放数据（CSV/JSON）。

    Returns [{"ep_id", "views", "cpa", "completion_rate", "tags"}]；
    未配置返回 None（调用方回退 mock）；配置但解析失败抛 ValueError。
    """
    configured = os.getenv("DRAMAMATRIX_ANALYTICS_IMPORT", "").strip()
    if not configured:
        return None
    path = Path(configured)
    if not path.is_file():
        raise ValueError(f"DRAMAMATRIX_ANALYTICS_IMPORT 文件不存在：{configured}")
    records: list[dict] = []
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload if isinstance(payload, list) else payload.get("records", [])
        if not isinstance(rows, list):
            raise ValueError("JSON 投放数据应为对象数组或 {records: [...]}")
        for row in rows:
            records.append(_normalize_analytics_row(row))
    else:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                records.append(_normalize_analytics_row(row))
    if not records:
        raise ValueError("投放数据文件中没有可导入的记录。")
    return records


def _normalize_analytics_row(row: dict) -> dict:
    def _number(name: str) -> float:
        value = str(row.get(name, "") or "0").strip()
        try:
            return float(value)
        except ValueError:
            raise ValueError(f"投放数据字段 {name} 不是数字：{value!r}") from None

    tags_raw = str(row.get("tags", "") or "").strip()
    tags = [t.strip() for t in tags_raw.replace("；", ";").replace("|", ";").split(";") if t.strip()]
    ep_id = str(row.get("ep_id", "") or "").strip() or "ep_01"
    # R1：platform 为可选列（如 douyin/kuaishou/wechat），缺省 unknown——
    # 与 project/source 一起构成投放数据的归属维度。
    platform = str(row.get("platform", "") or "").strip() or "unknown"
    return {
        "ep_id": ep_id,
        "views": int(_number("views")),
        "cpa": _number("cpa"),
        "completion_rate": _number("completion_rate"),
        "tags": json.dumps(tags or ["未知"], ensure_ascii=False),
        "platform": platform,
    }
