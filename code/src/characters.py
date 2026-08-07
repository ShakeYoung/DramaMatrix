"""Character consistency (阶段3) helpers.

Extracts a lightweight character sheet from the extracted/fallback scripts and
renders a text block to inject into storyboard and Agnes generation prompts so
that character appearance/wardrobe stay consistent across shots.
"""

from __future__ import annotations

import json
import re
from typing import Any, Sequence

from pydantic import BaseModel

from src.state import CharacterSheet


def extract_characters(text: str, fallback: Sequence[CharacterSheet] | None = None) -> list[CharacterSheet]:
    """Derive a character sheet from script text.

    Deterministic extraction looks for dialogue attributions like "X说"/"X道"
    to collect speaker names, then assigns a generic but stable description.
    When nothing matches, use the provided fallback (e.g. Agent 3's fallback
    episode outline) to keep the pipeline moving.
    """
    names: list[str] = []
    if text:
        # match dialogue attributions like “XX说/道/喊/问：...” — avoid capturing
        # leading adjectives (e.g. 反派冷冷道) by keeping only the last 2–4 chars.
        for match in re.finditer(r"([\u4e00-\u9fa5A-Za-z0-9]{2,6}?)\s*(?:说|道|喊|问|答|怒吼|冷笑)\s*[:：]", text):
            raw = match.group(1)
            # drop trailing adjective/suffix noise (冷冷/大声/淡淡…)
            cleaned = re.sub(r"(冷冷|淡淡|大声|低声|冷笑|轻声|缓缓|狠狠|厉声).*$", "", raw)
            name = cleaned.strip() or raw
            if name and name not in names:
                names.append(name)
    # cap to avoid spurious extraction
    names = names[:12]

    characters = [
        CharacterSheet(
            name=name,
            appearance="外表与服化以剧本一致为准",
            signature="",
            role="",
        )
        for name in names
    ]
    if not characters and fallback:
        characters = list(fallback)
    return characters


# 增强：叙述型外形/服化线索（P0-B），让"某某说"以外的人物也能被抽到。
_APPEARANCE_PATTERNS = [
    r"([\u4e00-\u9fa5A-Za-z0-9]{2,8}?)\s*(?:穿着|身披|披着|戴着|手持|提着|怀揣|身穿)\s*([^，。；,.;！!？?]{1,30})",
    r"([\u4e00-\u9fa5A-Za-z0-9]{2,8}?)\s*(?:是|乃|身份为|本是|原是)\s*([^，。；,.;！!？?]{1,30})",
]
# 名称引入线索："名为萧寒" / "叫做沈娇娇"
_NAME_INTRO_PATTERN = r"(?:名为|叫做|本名|名叫|叫做|名曰)\s*([\u4e00-\u9fa5A-Za-z0-9]{2,8})"
# 常见代词/指示词，不应作为角色名
_PRONOUNS = {"他", "她", "它", "他们", "她们", "这个", "那个", "此人", "对方", "其"}


def extract_characters_enhanced(text: str) -> list[CharacterSheet]:
    """Extract characters from narrative text using multiple cue patterns.

    Unlike extract_characters (dialogue attribution only), this also picks up
    wardrobe/appearance/identity cues ("X穿着…", "X是…/X乃…") and name
    introductions ("名为X"), so a purely narrative source no longer yields an
    empty character table. Deterministic extraction is best-effort; the LLM
    path in build_character_bible is authoritative.
    """
    found: dict[str, CharacterSheet] = {}
    if not text:
        return []
    # 先用对话归属补名
    for ch in extract_characters(text):
        if ch.name not in _PRONOUNS:
            found[ch.name] = ch
    # 再用外形/身份线索补名并丰富描述
    for pattern in _APPEARANCE_PATTERNS:
        for match in re.finditer(pattern, text):
            name = match.group(1).strip()
            cue = match.group(2).strip()
            if not name or len(name) > 10 or name in _PRONOUNS:
                continue
            existing = found.get(name)
            if existing:
                if cue and cue not in existing.appearance:
                    existing.appearance = (existing.appearance + "；" + cue).strip("；")
            else:
                found[name] = CharacterSheet(
                    name=name,
                    appearance=cue or "外表与服化以剧本一致为准",
                    signature="",
                    role="",
                )
    # 名称引入线索（名为/叫做…）
    for match in re.finditer(_NAME_INTRO_PATTERN, text):
        name = match.group(1).strip()
        if name and name not in _PRONOUNS and name not in found:
            found[name] = CharacterSheet(
                name=name,
                appearance="外表与服化以剧本一致为准",
                signature="",
                role="",
            )
    return list(found.values())[:20]


class CharacterBible(BaseModel):
    """LLM-generated structured character bible (P0-B)."""
    characters: list[CharacterSheet]


def build_character_bible(
    script_outline: str,
    episodes: Sequence[Any],
    raw_text: str,
) -> list[CharacterSheet]:
    """Build a character bible from multiple sources (P0-B).

    Tries an LLM (create_text_model) over a merged context of the master
    outline + per-episode outline/ending_hook + raw text. Falls back to the
    enhanced deterministic extractor when the LLM is unavailable or fails.

    `episodes` items may be EpisodeState or dicts; only outline/ending_hook are
    read.
    """
    context_parts = [f"【总纲】\n{(script_outline or '').strip()}"]
    for ep in episodes or []:
        outline = getattr(ep, "script_data", None)
        outline = getattr(outline, "outline", None) if outline is not None else None
        hook = getattr(ep, "script_data", None)
        hook = getattr(hook, "ending_hook", None) if hook is not None else None
        if outline or hook:
            context_parts.append(f"- 大纲：{outline or ''}；结尾钩子：{hook or ''}")
    context = "\n".join(context_parts) + f"\n【原文摘录】\n{(raw_text or '')[:2000]}"

    try:
        from src.text_model import create_text_model
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_core.output_parsers import PydanticOutputParser

        parser = PydanticOutputParser(pydantic_object=CharacterBible)
        system = (
            "你是短剧角色圣经编剧。请从给定的总纲、各集大纲与原文中，提取全部主要角色，"
            "为每个角色给出稳定的姓名、外形/服化描述、服装编号(wardrobe_id)、角色定位、"
            "出场集号(appears_in)与参考图生成提示词(reference_image_prompt)。"
            "外形描述要具体到发型/服饰/配饰/年龄段，保证跨镜一致。\n"
            + parser.get_format_instructions()
        )
        llm = create_text_model(temperature=0.3)
        response = llm.invoke([SystemMessage(content=system), HumanMessage(content=context)])
        bible: CharacterBible = parser.invoke(response)
        if bible.characters:
            return canonicalize_characters(bible.characters)
    except Exception as exc:  # noqa: BLE001 - fallback is intended
        print(f"      [CharacterBible] LLM 生成失败，回退到确定性抽取：{exc}")

    # 回退：增强的确定性抽取（叙述型线索）
    return canonicalize_characters(extract_characters_enhanced(raw_text + "\n" + script_outline))


def characters_for_episode(characters: Sequence[CharacterSheet], ep_key: str) -> list[CharacterSheet]:
    """Return the subset of the bible that appears in a given episode."""
    return [ch for ch in characters if not ch.appears_in or ep_key in ch.appears_in]


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein distance for short name strings."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[lb]


def canonicalize_characters(characters: Sequence[CharacterSheet]) -> list[CharacterSheet]:
    """Assign stable canonical IDs and merge near-duplicate names (P1-2 / R6).

    Fixes drift like "Zhang Ruochen" vs "Zhang Ruocheng" (1-char edit distance)
    by collapsing them onto one canonical entry with a stable character_id and
    canonical_name, recording variants as aliases. Uses dedicated fields rather
    than overloading signature.
    """
    merged: list[CharacterSheet] = []
    for ch in characters:
        canonical = ch
        for existing in merged:
            if (
                existing.name == ch.name
                or (len(ch.name) >= 4 and _edit_distance(existing.name, ch.name) <= 1)
            ):
                # Keep the first-seen name as canonical; enrich appearance/appearance.
                if ch.appears_in:
                    existing.appears_in = list(set(existing.appears_in) | set(ch.appears_in))
                if ch.appearance and ch.appearance not in existing.appearance:
                    existing.appearance = (existing.appearance + "；" + ch.appearance).strip("；")
                # Record the variant as an alias (R6).
                if ch.name and ch.name != existing.name and ch.name not in existing.aliases:
                    existing.aliases = list(existing.aliases) + [ch.name]
                canonical = None
                break
        if canonical is not None:
            merged.append(canonical)
    # Assign stable canonical IDs / canonical_name on dedicated fields (R6).
    for idx, ch in enumerate(merged, 1):
        ch.character_id = ch.character_id or f"char_{idx:02d}"
        ch.canonical_name = ch.canonical_name or ch.name
    return merged


def apply_alias_substitution(text: str, characters: Sequence[CharacterSheet]) -> str:
    """Replace alias names in a prompt with the canonical name (R6).

    So a storyboard prompt mentioning "Zhang Ruocheng" gets rewritten to the
    canonical "Zhang Ruochen", keeping a single identity in generation prompts.
    """
    if not text:
        return text
    result = text
    for ch in characters:
        canonical = ch.canonical_name or ch.name
        for alias in ch.aliases:
            if alias and alias != canonical and alias in result:
                result = result.replace(alias, canonical)
    return result


def render_character_block(characters: Sequence[CharacterSheet]) -> str:
    """Produce a stable English/Chinese block for prompt injection."""
    if not characters:
        return ""
    parts = ["【角色一致性参考 / Character consistency】"]
    for ch in characters:
        parts.append(f"- {ch.name}：{ch.appearance}{'（' + ch.signature + '）' if ch.signature else ''}")
    return "\n".join(parts) + "\n保持各角色外貌、服化、身份在每一镜之间完全一致。"


def characters_to_json(characters: Sequence[CharacterSheet]) -> list[dict]:
    return [ch.model_dump(mode="json") for ch in characters]