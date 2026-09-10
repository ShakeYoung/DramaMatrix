"""Character reference-image generation & evidence chain (U2).

reference_image_prompt 此前"只生成不消费"。本模块把链路补全：

    角色圣经(reference_image_prompt)
        -> 图像供应商生成（DRAMAMATRIX_IMAGE_PROVIDER）
        -> outputs/<project>/refs/characters/<character_id>.png 落盘
        -> SHA-256 + reference_assets 落库（asset_type='character'）
        -> CharacterSheet.reference_image_path 回填（幂等：已有即跳过）
        -> Agent5 场景首镜作为条件生成输入（continuity.prepare_shot_reference）

失败语义：单个角色失败只告警不阻断——参考图是增强不是门禁；整段异常同样
降级为无参考图继续（流水线行为退回 U2 之前）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from src.agnes_video import output_root, safe_component, sha256_file
from src.image_providers import ImageProvider
from src.state import CharacterSheet


def _character_reference_dir(project_id: str) -> Path:
    directory = output_root() / safe_component(project_id) / "refs" / "characters"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def build_reference_prompt(character: CharacterSheet) -> str:
    """参考图提示词：优先用圣经字段，缺省时由外形描述兜底拼装。"""
    if character.reference_image_prompt.strip():
        return character.reference_image_prompt.strip()
    name = character.canonical_name or character.name
    return (
        f"{name}：{character.appearance}。全身立绘参考图，竖版构图，正面清晰面部，"
        "服化/发型/配饰与描述完全一致，纯色背景，高细节。"
    )


def ensure_character_reference_images(
    project_id: str,
    characters: Sequence[CharacterSheet],
    provider: ImageProvider,
) -> dict[str, str]:
    """为缺少参考图的角色生成图片，回填路径并记录证据链。

    Returns {canonical_name: local_path}（仅含本次可用的参考图）。
    幂等：reference_image_path 存在且文件在盘上时跳过，不重复扣费。
    """
    available: dict[str, str] = {}
    if not characters:
        return available
    directory = _character_reference_dir(project_id)
    for character in characters:
        canonical = character.canonical_name or character.name
        existing = character.reference_image_path
        if existing and Path(existing).is_file():
            available[canonical] = existing
            continue
        character_id = character.character_id or safe_component(character.name)
        destination = directory / f"{character_id}.png"
        if destination.is_file() and destination.stat().st_size > 0:
            # 历史产物复用：文件已生成过（如上次运行中断在回填前）。
            character.reference_image_path = str(destination)
            available[canonical] = str(destination)
            continue
        try:
            provider.generate(build_reference_prompt(character), destination)
        except Exception as exc:  # noqa: BLE001 - 单角色失败不阻断整集
            print(f"   ⚠️ 角色 {canonical} 参考图生成失败（继续无参考图渲染）：{exc}")
            continue
        if not destination.is_file():
            print(f"   ⚠️ 角色 {canonical} 参考图生成未产出文件，跳过。")
            continue
        # R3：参考图费用落账（价目表未配置则零费用、不落账；幂等于角色名）。
        try:
            from src import cost_ledger

            cost_ledger.record_image_cost(
                project_id, character_id, provider=getattr(provider, "name", "image")
            )
        except Exception:  # noqa: BLE001 - 记账失败不影响生产
            pass
        character.reference_image_path = str(destination)
        available[canonical] = str(destination)
        # 证据链：与场景参考/尾帧同表（reference_assets），asset_type='character'。
        try:
            from src.db import db_insert_reference_asset
            db_insert_reference_asset(
                project_id=project_id,
                asset_type="character",
                ref_id=character_id,
                local_path=str(destination),
                sha256=sha256_file(destination),
                referenced_by_shot=None,
            )
        except Exception as exc:  # noqa: BLE001 - 证据落库失败不影响生产
            print(f"   ⚠️ 角色 {canonical} 参考图证据落库失败（不阻断）：{exc}")
    return available


def character_reference_map(
    characters: Sequence[CharacterSheet],
    project_id: str | None = None,
) -> dict[str, str]:
    """收集当前已存在的角色参考图 {canonical_name: local_path}。

    供恢复运行使用：reference_image_path 已回填但未重新生成时也能拿到映射。
    """
    mapping: dict[str, str] = {}
    for character in characters:
        canonical = character.canonical_name or character.name
        path = character.reference_image_path
        if path and Path(path).is_file():
            mapping[canonical] = path
        elif project_id and character.character_id:
            # 兜底：从 refs 目录按 character_id 找历史产物。
            candidate = (
                output_root() / safe_component(project_id) / "refs" / "characters"
                / f"{character.character_id}.png"
            )
            if candidate.is_file():
                mapping[canonical] = str(candidate)
    return mapping
