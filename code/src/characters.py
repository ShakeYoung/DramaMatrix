"""Character consistency (阶段3) helpers.

Extracts a lightweight character sheet from the extracted/fallback scripts and
renders a text block to inject into storyboard and Agnes generation prompts so
that character appearance/wardrobe stay consistent across shots.
"""

from __future__ import annotations

import json
import re
from typing import Sequence

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