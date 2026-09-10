"""R3 真实成本账本：金额口径的预算护栏与费用沉淀。

设计要点（对应"可控生产"评审结论）：
- 价目表可配置（DRAMAMATRIX_PRICE_*）；未配置（默认 0）时功能休眠，
  既有按创建次数的护栏（DRAMAMATRIX_MAX_AGNES_CREATES）保持不变。
  所有金额来自价目表估算（estimated=1），供应商不给账单 API 时明确标记
  为估算，允许后续对账修正。
- 三态生命周期：estimate（分镜后预估）→ reserved（付费请求已提交）→
  confirmed（资产落地）。提交结果不确定（下载失败/校验不过/重绘）时保留
  reserved 不释放——钱大概率已花，但不算"合格产出成本"。
- 每次付费请求前检查剩余额度（单集预算 DRAMAMATRIX_EPISODE_BUDGET、
  项目预算 DRAMAMATRIX_PROJECT_BUDGET，0=不限）。
- 幂等：同 ref 重复提交不重复入账（进程死亡后 --resume 不重复计费）。
"""

from __future__ import annotations

import os

import src.db as db_module


def _price_env(name: str) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return 0.0
    try:
        return max(float(raw), 0.0)
    except ValueError:
        print(f"⚠️ {name} 不是有效金额（{raw!r}），按未配置处理。")
        return 0.0


def currency() -> str:
    return os.getenv("DRAMAMATRIX_COST_CURRENCY", "CNY").strip() or "CNY"


# ---- 价目表（估算口径；0 = 不按金额跟踪该阶段）----

def video_create_price() -> float:
    """单次视频创建的估算费用（每镜一次；重绘也是一次新创建）。"""
    return _price_env("DRAMAMATRIX_PRICE_VIDEO_CREATE")


def tts_price_per_1k_chars() -> float:
    return _price_env("DRAMAMATRIX_PRICE_TTS_1K_CHARS")


def image_create_price() -> float:
    return _price_env("DRAMAMATRIX_PRICE_IMAGE_CREATE")


# ---- 预算 ----

def episode_budget() -> float:
    return _price_env("DRAMAMATRIX_EPISODE_BUDGET")


def project_budget() -> float:
    return _price_env("DRAMAMATRIX_PROJECT_BUDGET")


def episode_spent(project_id: str, ep_key: str) -> float:
    return db_module.db_cost_spent(project_id, ep_key)


def project_spent(project_id: str) -> float:
    return db_module.db_cost_spent(project_id)


def episode_budget_allows(project_id: str, ep_key: str, next_amount: float) -> bool:
    """单集额度检查：已花费 + 本次请求的估算费用不得超过单集预算。"""
    budget = episode_budget()
    if budget <= 0:
        return True
    return episode_spent(project_id, ep_key) + next_amount <= budget + 1e-9


def project_budget_allows(project_id: str, next_amount: float) -> bool:
    budget = project_budget()
    if budget <= 0:
        return True
    return project_spent(project_id) + next_amount <= budget + 1e-9


# ---- 记账 ----

def record_cost_event(*, project_id: str, stage: str, kind: str, amount: float,
                      ep_key: str | None = None, shot_id: str | None = None,
                      provider: str | None = None, ref_id: str | None = None,
                      note: str | None = None) -> bool:
    """写入一笔成本事件（幂等）。amount<=0 时跳过（价目未配置）。"""
    if amount <= 0:
        return False
    attempt = 1
    if shot_id and ep_key:
        try:
            prior = db_module.db_query_cost_events(project_id, ep_key)
            attempt = 1 + sum(
                1 for e in prior
                if e.get("shot_id") == shot_id and e.get("stage") == stage
                and e.get("kind") in ("reserved", "confirmed")
            )
        except Exception:
            attempt = 1
    return db_module.db_insert_cost_event(
        project_id=project_id, stage=stage, kind=kind, amount=amount,
        ep_key=ep_key, shot_id=shot_id, currency=currency(),
        estimated=True, attempt=attempt, provider=provider,
        ref_id=ref_id, note=note,
    )


def confirm_cost_event(project_id: str, stage: str, ref_id: str) -> bool:
    """reserved → confirmed（资产落地；estimated 置 0 表示已对账确认）。"""
    return db_module.db_confirm_cost_event(project_id, stage, ref_id)


def record_video_reservation(project_id: str, ep_key: str, shot_id: str,
                             task_id: str, provider: str | None = None) -> bool:
    """付费视频创建已提交：按价目表预留一笔费用（幂等于 task_id）。"""
    price = video_create_price()
    if price <= 0:
        return False
    return record_cost_event(
        project_id=project_id, ep_key=ep_key, shot_id=shot_id,
        stage="video", kind="reserved", amount=price,
        provider=provider, ref_id=f"video:{task_id}",
    )


def confirm_video_cost(project_id: str, task_id: str) -> bool:
    return confirm_cost_event(project_id, "video", f"video:{task_id}")


def record_image_cost(project_id: str, ref_name: str, provider: str | None = None) -> bool:
    """角色/场景参考图生成费用（幂等于参考名）。"""
    price = image_create_price()
    if price <= 0:
        return False
    return record_cost_event(
        project_id=project_id, stage="image", kind="confirmed", amount=price,
        provider=provider, ref_id=f"image:{ref_name}",
    )


def record_storyboard_estimate(project_id: str, ep_key: str, storyboard_version: int,
                               unrendered_shots: int, provider: str | None = None) -> float:
    """分镜确认后的成本预估（幂等于 ep+版本）；返回预估金额。

    预估只记一次、不参与花费计算（spend 口径只含 reserved/confirmed）。
    """
    price = video_create_price()
    if price <= 0:
        return 0.0
    amount = round(unrendered_shots * price, 4)
    if amount > 0:
        record_cost_event(
            project_id=project_id, ep_key=ep_key, stage="video", kind="estimate",
            amount=amount, provider=provider,
            ref_id=f"estimate:{ep_key}:v{storyboard_version}",
            note=f"{unrendered_shots} shots x {price}",
        )
    return amount
