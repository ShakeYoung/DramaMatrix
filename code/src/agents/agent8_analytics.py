# /Users/yangkang/Library/CloudStorage/OneDrive-共享的库-onedrive/own_project/DramaMatrix/code/src/agents/agent8_analytics.py
import hashlib
import json
import sqlite3

import src.db as db_module
from src.state import DramaState, MarketFeedback
from src.runtime_options import is_demo_mode


def query_market_trends() -> dict:
    """从 analytics 表挖掘选题先验。

    R1：只有 source='imported'（运营导入的真实投放数据）参与推荐；
    simulated（演示模拟）与 legacy_unverified（旧版存量）一律排除，
    避免"没有市场证据"被放大成"系统已获得市场结论"。
    完播率按播放量加权（SUM(rate*views)/SUM(views)），避免单条小样本
    稀有标签组合凭裸均值登顶。
    """
    try:
        # 动态读取 src.db.DB_PATH：测试通过替换该属性隔离数据库（W2 修复
        # 顶层按值绑定导致的跨测试串库问题）。
        conn = db_module._connect()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT tags, SUM(completion_rate * views) / NULLIF(SUM(views), 0) AS weighted_comp
            FROM analytics
            WHERE source = 'imported'
            GROUP BY tags
            ORDER BY weighted_comp DESC
            LIMIT 1
            """
        )
        row = cursor.fetchone()
        conn.close()

        if row and row[0] and row[1] is not None:
            best_tags = json.loads(row[0]) if row[0].startswith('[') else [row[0]]
            return {
                "trend": (
                    f"数据库挖掘出炉！\n"
                    f"         - 最优标签系: {best_tags}\n"
                    f"         - 播放量加权平均完播率: {row[1]:.2f}\n"
                    f"         => 策略路由: 下一次寻找同类竞品。"
                ),
                "tags": best_tags,
            }
    except Exception as e:
        print(f"      [Analytics] DB query error: {e}")

    return {
        "trend": "暂无显著历史数据，建议尝试反套路的轻松向甜宠或脑洞系统文。",
        "tags": ["萌宝", "系统", "女频"],
    }


def _file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def process_agent8_analytics(state: DramaState) -> DramaState:
    """
    Agent 8: 数据大脑 (Analytics Feedback Agent) -> Insight Agent (Proprietary Database Mining)
    接收市场投放的数据回流存入数据库，并提炼指导下一次选品的标签偏好。

    R1 行为约束：
    - production 模式：未配置/导入失败/导入为空 → waiting_for_market_data，
      零写入（不生成随机数据），集保持 growth_ready，配置数据后 --resume 重进。
    - demo 模式：无数据时写一条 source='simulated' 模拟行（幂等：
      同项目同集只保留一条）。
    - 导入按 dedup_key（project + 文件内容哈希 + 行号）唯一索引去重，
      同一文件重复导入不会翻倍。
    - 处理完成后集进入 analytics_done 终态，task_cycle 仅在实际完成一次
      数据回流后 +1——重启不再重复进 Agent8。
    """
    print("--- [Agent 8: Insight Agent (Analytics / Database Mining)] ---")

    growth_eps = sorted(
        key for key, ep in state["episodes"].items() if ep.status == "growth_ready"
    )
    if not growth_eps:
        print("尚无投流测试完成的剧集，暂时无法收集市场反馈。")
        return state

    print("正在连接 SQLite 数据库拉取各大平台投放消耗与转化核心指标...")

    imported_rows = []
    import_error: Exception | None = None
    import_file = None
    records = None
    try:
        import os

        from src.local_sources import load_analytics_records

        import_file = os.getenv("DRAMAMATRIX_ANALYTICS_IMPORT", "").strip()
        records = load_analytics_records()
        if records is not None:
            print(f"-> 已读取投放数据文件 {len(records)} 条记录（analytics）。")
    except ValueError as exc:
        import_error = exc
    except Exception as exc:  # noqa: BLE001 - 导入链路故障在 production 下阻塞而非回退
        import_error = exc

    if import_error is not None:
        print(f"-> ⚠️ 投放数据导入失败：{import_error}")

    project_id = state.get("project_id")
    recorded_any = False

    if records:
        # 真实导入：dedup_key = project + 文件内容哈希 + 行号。
        # 同一文件重复执行 INSERT OR IGNORE 全部跳过；平台新导出的文件
        # 内容变化即产生新键，自然进入。
        try:
            file_digest = _file_sha256(import_file) if import_file else "nofile"
        except OSError:
            file_digest = "unreadable"
        for index, row in enumerate(records):
            dedup_key = hashlib.sha256(
                f"{project_id}|{file_digest}|{index}".encode("utf-8")
            ).hexdigest()
            imported_rows.append(
                (
                    row["ep_id"], row["views"], row["cpa"], row["completion_rate"],
                    row["tags"], project_id, row.get("platform") or "unknown",
                    "imported", dedup_key,
                )
            )

        try:
            conn = db_module._connect()
            cursor = conn.cursor()
            before = conn.total_changes
            cursor.executemany(
                """INSERT OR IGNORE INTO analytics
                   (ep_id, views, cpa, completion_rate, tags,
                    project_id, platform, source, dedup_key)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                imported_rows,
            )
            conn.commit()
            inserted = conn.total_changes - before
            conn.close()
            duplicated = len(imported_rows) - inserted
            print(f"-> 真实投放数据导入完成：新增 {inserted} 条"
                  + (f"，跳过重复 {duplicated} 条（dedup_key 命中）" if duplicated else "") + "。")
            recorded_any = True
        except Exception as e:
            print(f"-> 写入分析数据库失败：{e}")
            if not is_demo_mode():
                state["system_status"] = "waiting_for_market_data"
                print("   ❌ production 模式：入库失败保持等待（waiting_for_market_data），不回退模拟数据。")
                return state
    else:
        # 无真实数据：production 等待，demo 写模拟行（幂等）。
        if not is_demo_mode():
            state["system_status"] = "waiting_for_market_data"
            hint = import_error if import_error is not None else "未配置 DRAMAMATRIX_ANALYTICS_IMPORT"
            print(f"   ❌ production 模式：无真实投放数据（{hint}），进入等待，不写入模拟数据。")
            print("   → 配置 DRAMAMATRIX_ANALYTICS_IMPORT（平台导出 CSV/JSON）后 --resume 重进 Agent 8。")
            return state

        import random

        ep_key = growth_eps[0]
        current_tags = state.get("meta_info", {}).get("genre_tags", ["未知"])
        try:
            conn = db_module._connect()
            cursor = conn.cursor()
            cursor.execute(
                """INSERT OR IGNORE INTO analytics
                   (ep_id, views, cpa, completion_rate, tags,
                    project_id, platform, source, dedup_key)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    ep_key,
                    random.randint(1000, 50000),
                    random.uniform(5.0, 20.0),
                    random.uniform(0.1, 0.4),
                    json.dumps(current_tags, ensure_ascii=False),
                    project_id,
                    "simulated",
                    "simulated",
                    f"simulated:{project_id}:{ep_key}",
                ),
            )
            conn.commit()
            conn.close()
            print(f"-> ⚠️ demo 模式：写入模拟数据（source='simulated'，同项目同集幂等）。")
            recorded_any = True
        except Exception as e:
            print(f"-> 写入分析数据库失败：{e}")

    if not recorded_any:
        # 入库完全失败（demo 模式下才会走到这里）：本次不算完成，不动周期。
        state["system_status"] = "waiting_for_market_data"
        return state

    # Query for the next run
    print("-> 正在执行 SQL Query 进行数据洞察分析 (Chunked Analysis)...")
    trends = query_market_trends()
    print(f"-> 发现: {trends['trend']}")

    feedback = MarketFeedback(
        trend_analysis=trends['trend'],
        suggested_tags=trends['tags']
    )

    # 注入全局状态，供下次 Agent 1 抓取时使用
    state["market_feedback"] = feedback
    # R1：本轮已回流的集进入终态 analytics_done——重启后不再路由进 Agent8，
    # 不重复插数、不虚增周期。growth_ready 保留给尚未回流的集。
    for ep_key in growth_eps:
        state["episodes"][ep_key].status = "analytics_done"
        print(f"   -> {ep_key} 市场反馈已回流，进入终态 analytics_done。")
    state["task_cycle"] = state.get("task_cycle", 1) + 1
    # 新市场周期开始时重置换书尝试次数，避免上一周期消耗压缩本周期额度（F9）
    state["scout_attempts"] = 0
    import os

    if os.getenv("DRAMAMATRIX_AUTO_NEXT_CYCLE", "0").strip() in {"1", "true", "yes"}:
        state["system_status"] = "cycle_completed_ready_for_next"
    else:
        state["system_status"] = "cycle_completed"

    print(f"✅ 市场偏好提取完毕。建议下期标签: {feedback.suggested_tags}")

    return state
