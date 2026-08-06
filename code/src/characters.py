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
            return bible.characters
    except Exception as exc:  # noqa: BLE001 - fallback is intended
        print(f"      [CharacterBible] LLM 生成失败，回退到确定性抽取：{exc}")

    # 回退：增强的确定性抽取（叙述型线索）
    return extract_characters_enhanced(raw_text + "\n" + script_outline)


def characters_for_episode(characters: Sequence[CharacterSheet], ep_key: str) -> list[CharacterSheet]:
    """Return the subset of the bible that appears in a given episode."""
    return [ch for ch in characters if not ch.appears_in or ep_key in ch.appears_in]


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