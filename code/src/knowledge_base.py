"""Queryable prior-knowledge base for Agent 2 (U6).

把"假 RAG"（打印连接 ChromaDB、返回硬编码文本）替换为真实可查的检索：

- 存储：SQLite knowledge_entries 表（init_db 建表；运营可直接增补条目）。
- 检索：BM25-lite 词频评分（CJK 单字+二元组 / 拉丁词），标题×2、标签×3
  加权。表规模为运营级（百条内）时内存评分即可；嵌入后端留作扩展。
- 种子：首次使用时幂等写入内置先验（含原硬编码 3 条 + 节奏/角色/投放）。
- 降级：DB 不可用时回退到内置先验全文，行为不劣于升级前。

用法（Agent 2）：
    entries = retrieve(f"{title} {excerpt}", k=3)
    prior_knowledge = format_prior_knowledge(entries)
"""

from __future__ import annotations

import re
import sqlite3
import time
from typing import Optional

# (category, title, content, tags)
DEFAULT_PRIORS: list[tuple[str, str, str, str]] = [
    (
        "prior",
        "视觉猎奇度优先",
        "目前的生视频大模型（如 Runway/可灵）擅长表现超自然、诡异或极其华丽的场景。"
        "剧情必须包含普通实拍难以达成的“视觉奇观”（如漫天神佛、赛博克苏鲁、不可名状之物）。",
        "视觉,奇观,生视频,漫剧",
    ),
    (
        "prior",
        "情绪推进浓烈直白",
        "每隔 15 秒必须有一个情绪转折点（极度愤怒、极度绝望或极限装X），"
        "情绪推进必须极度浓烈且直白，不留平淡段落。",
        "节奏,情绪,爽点,转折",
    ),
    (
        "prior",
        "信息差与强烈反差",
        "主角的隐藏身份必须与表面形成巨大反差，如“扫地僧实为万古神帝”、“乞丐其实是隐藏龙王”，"
        "以便 AI 生成对比极度强烈的跨维度画风。",
        "身份,反差,打脸,设定",
    ),
    (
        "prior",
        "开篇三秒钩子",
        "竖屏短剧的完播率由前 3 秒决定：第一镜必须直接进入冲突现场（耳光/坠崖/退婚），"
        "禁止环境交代式开场；字幕用大字报体突出情绪关键词。",
        "开场,钩子,完播,竖屏",
    ),
    (
        "prior",
        "角色弧光与 CP 感",
        "男女主的每次同框都要推进关系错位（误解→试探→反转护航），反派智商在线但每次都差一步，"
        "观众爽感来自“预期又超预期”的打脸节奏。",
        "角色,关系,反派,CP",
    ),
    (
        "prior",
        "投放切片结构",
        "投流素材以“冲突钩子(0-5s)+身份反转(5-20s)+悬念断点(20-30s)”三段式切片，"
        "标题用第一人称冲突句式，标签命中题材+情绪双维度。",
        "投放,切片,投流,标题",
    ),
]

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fa5]")


def _tokenize(text: str, cjk_singles: bool = False) -> list[str]:
    """拉丁/数字整词小写；CJK 默认只取相邻二元组（cjk_singles=True 时附带单字）。

    查询侧禁用单字：中文单字（如"方/学"）过于泛化，会让无关查询误命中；
    二元组在无分词器的约束下是最小可靠语义单元。文档侧保留单字计数以
    兼容极短字段（如两字标题）。
    """
    tokens: list[str] = [m.group().lower() for m in _TOKEN_RE.finditer(text or "")]
    if not cjk_singles:
        tokens = [t for t in tokens if not _is_cjk(t)]
    chars = [c for c in (text or "") if "\u4e00" <= c <= "\u9fa5"]
    tokens.extend(a + b for a, b in zip(chars, chars[1:]))
    return tokens


def _is_cjk(token: str) -> bool:
    return len(token) == 1 and "\u4e00" <= token <= "\u9fa5"


def _connect() -> sqlite3.Connection:
    # 动态读取 src.db.DB_PATH：测试通过替换该属性隔离数据库。
    import src.db as db_module

    return sqlite3.connect(db_module.DB_PATH)


def seed_knowledge_base(force: bool = False) -> int:
    """把内置先验幂等写入 knowledge_entries。返回写入条数。"""
    conn = _connect()
    try:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS knowledge_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT DEFAULT 'prior',
                title TEXT,
                content TEXT NOT NULL,
                tags TEXT DEFAULT '',
                created_at_unix REAL,
                recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )"""
        )
        if force:
            conn.execute("DELETE FROM knowledge_entries WHERE category = 'prior'")
        count = conn.execute("SELECT COUNT(*) FROM knowledge_entries").fetchone()[0]
        if count and not force:
            return 0
        now = time.time()
        written = 0
        for category, title, content, tags in DEFAULT_PRIORS:
            duplicate = conn.execute(
                "SELECT 1 FROM knowledge_entries WHERE title = ? AND category = ?",
                (title, category),
            ).fetchone()
            if duplicate:
                continue
            conn.execute(
                "INSERT INTO knowledge_entries (category, title, content, tags, created_at_unix)"
                " VALUES (?, ?, ?, ?, ?)",
                (category, title, content, tags, now),
            )
            written += 1
        conn.commit()
        return written
    finally:
        conn.close()


def retrieve(query: str, k: int = 3) -> list[dict]:
    """按与 query 的词频相关性返回 top-k 先验条目。

    评分：sum(tf(query_token) * weight)，weight: title=2, tags=3, content=1。
    无命中时返回空列表（调用方回退到全量内置先验）。
    """
    try:
        conn = _connect()
    except sqlite3.Error:
        return []
    try:
        rows = conn.execute(
            "SELECT id, category, title, content, tags FROM knowledge_entries"
        ).fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []
    scored: list[tuple[float, dict]] = []
    for row_id, category, title, content, tags in rows:
        title_tokens = _tokenize(title or "", cjk_singles=True)
        tag_tokens = _tokenize(tags or "", cjk_singles=True)
        content_tokens = _tokenize(content or "", cjk_singles=True)
        score = 0.0
        for token in query_tokens:
            score += 2.0 * title_tokens.count(token)
            score += 3.0 * tag_tokens.count(token)
            score += 1.0 * content_tokens.count(token)
        if score > 0:
            scored.append(
                (
                    score,
                    {
                        "id": row_id,
                        "category": category,
                        "title": title,
                        "content": content,
                        "tags": tags,
                        "score": score,
                    },
                )
            )
    scored.sort(key=lambda item: item[0], reverse=True)
    return [entry for _, entry in scored[:k]]


def format_prior_knowledge(entries: list[dict]) -> Optional[str]:
    """把检索结果编排为注入 LLM 的先验文本；空输入返回 None。"""
    if not entries:
        return None
    lines = ["【AI爆款漫剧先验知识（按相关性检索自知识库）】"]
    for index, entry in enumerate(entries, 1):
        lines.append(f"{index}. {entry.get('title') or '先验'}：{entry.get('content') or ''}")
    return "\n".join(lines)


def load_prior_knowledge(query: str, k: int = 3) -> tuple[str, list[dict]]:
    """Agent2 入口：种子 + 检索 + 编排，任何异常回退内置先验全文。

    Returns (prior_text, entries)——entries 供调用方打印真实检索证据。
    """
    try:
        seed_knowledge_base()
        entries = retrieve(query, k=k)
        formatted = format_prior_knowledge(entries)
        if formatted:
            return formatted, entries
    except Exception as exc:  # noqa: BLE001 - 检索失败回退内置先验
        print(f"      [KnowledgeBase] 检索失败，回退内置先验：{exc}")
    fallback = "【AI爆款漫剧先验知识】\n" + "\n".join(
        f"{i}. {title}：{content}"
        for i, (_category, title, content, _tags) in enumerate(DEFAULT_PRIORS[:3], 1)
    )
    return fallback, []
