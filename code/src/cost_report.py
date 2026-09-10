"""R3 成本报表：单集费用 / 每分钟成片费用 / 重绘费用占比。

用法：
    cd code && python -m src.cost_report <project_id> [--json]

数据源：cost_events 账本（金额为价目表估算口径，供应商不给账单 API 时
estimated=1，允许后续对账）+ 项目快照中的成片交付证据（时长）。
人工处理时间暂无自动采集，报表中说明（需运营自行记录）。
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import src.db as db_module

# 成片时长证据的优先级：字幕版 > 配音版 > master（越往后越接近最终交付物）。
_DURATION_KIND_PRIORITY = ("subtitled", "voiced", "master")


def _episode_duration_seconds(state: dict[str, Any], ep_key: str) -> float | None:
    """从成片交付证据取时长：字幕版 > 配音版 > master（越靠后越接近交付物）。"""
    ep = (state.get("episodes") or {}).get(ep_key)
    if not ep:
        return None
    by_kind = {
        asset.get("kind"): float(asset["actual_duration"])
        for asset in (ep.get("deliverables") or [])
        if asset.get("actual_duration") and asset["actual_duration"] > 0
    }
    for kind in _DURATION_KIND_PRIORITY:
        if kind in by_kind:
            return by_kind[kind]
    return None


def build_cost_report(project_id: str) -> dict[str, Any]:
    events = db_module.db_query_cost_events(project_id)
    eps: dict[str, dict[str, Any]] = {}
    for e in events:
        ep_key = e.get("ep_key") or "(项目级)"
        bucket = eps.setdefault(ep_key, {
            "confirmed": 0.0, "reserved_open": 0.0, "estimate": 0.0,
            "redraw_cost": 0.0, "currency": e.get("currency") or "CNY",
        })
        kind = e.get("kind")
        amount = float(e.get("amount") or 0.0)
        if kind == "confirmed":
            bucket["confirmed"] += amount
        elif kind == "reserved":
            bucket["reserved_open"] += amount
        elif kind == "estimate":
            bucket["estimate"] = max(bucket["estimate"], amount)
        # 重绘费用：同一镜头第 2 次及以后的付费创建。
        if kind in ("reserved", "confirmed") and int(e.get("attempt") or 1) > 1:
            bucket["redraw_cost"] += amount

    snapshot = db_module.db_get_project_state_snapshot(project_id)
    state = snapshot.get("state") if snapshot else None
    for ep_key, bucket in eps.items():
        duration = _episode_duration_seconds(state, ep_key) if state else None
        bucket["duration_seconds"] = duration
        spent = bucket["confirmed"] + bucket["reserved_open"]
        bucket["spent"] = round(spent, 4)
        bucket["per_minute_cost"] = (
            round(bucket["confirmed"] / (duration / 60.0), 4)
            if duration and duration > 0 and bucket["confirmed"] > 0 else None
        )
        bucket["redraw_share"] = (
            round(bucket["redraw_cost"] / spent, 4) if spent > 0 else 0.0
        )

    total_confirmed = sum(b["confirmed"] for b in eps.values())
    total_reserved_open = sum(b["reserved_open"] for b in eps.values())
    total_redraw = sum(b["redraw_cost"] for b in eps.values())
    total_spent = total_confirmed + total_reserved_open
    currency = next(iter(eps.values()), {}).get("currency", "CNY") if eps else "CNY"
    return {
        "project_id": project_id,
        "currency": currency,
        "episodes": eps,
        "totals": {
            "confirmed": round(total_confirmed, 4),
            "reserved_open": round(total_reserved_open, 4),
            "spent": round(total_spent, 4),
            "redraw_cost": round(total_redraw, 4),
            "redraw_share": round(total_redraw / total_spent, 4) if total_spent > 0 else 0.0,
        },
        "notes": [
            "金额为价目表估算口径（DRAMAMATRIX_PRICE_*）；estimated=1 的行尚未与供应商账单对账。",
            "reserved_open 为已提交但资产未落地确认的预留（含失败/重绘中），计入花费但不计入每分钟合格成片成本。",
            "人工处理时间暂无自动采集，请运营另行记录（审阅/修复/整集验收耗时）。",
        ],
    }


def render_report_text(report: dict[str, Any]) -> str:
    lines = []
    cur = report["currency"]
    lines.append(f"成本报表 · 项目 {report['project_id']}（货币：{cur}）")
    lines.append("=" * 72)
    header = f"{'集':<10}{'已确认':>10}{'未确认预留':>12}{'合计':>10}{'重绘占比':>10}{'每分钟成片':>12}"
    lines.append(header)
    lines.append("-" * 72)
    for ep_key, b in sorted(report["episodes"].items()):
        per_min = f"{b['per_minute_cost']:.2f}" if b["per_minute_cost"] is not None else "N/A"
        lines.append(
            f"{ep_key:<10}{b['confirmed']:>10.2f}{b['reserved_open']:>12.2f}"
            f"{b['spent']:>10.2f}{b['redraw_share']:>10.1%}{per_min:>12}"
        )
    t = report["totals"]
    lines.append("-" * 72)
    lines.append(
        f"{'合计':<10}{t['confirmed']:>10.2f}{t['reserved_open']:>12.2f}"
        f"{t['spent']:>10.2f}{t['redraw_share']:>10.1%}{'':>12}"
    )
    lines.append("")
    for note in report["notes"]:
        lines.append(f"· {note}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="查看项目的金额成本报表。")
    parser.add_argument("project_id")
    parser.add_argument("--json", action="store_true", help="输出 JSON（便于脚本处理）")
    args = parser.parse_args(argv)

    report = build_cost_report(args.project_id)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(render_report_text(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
