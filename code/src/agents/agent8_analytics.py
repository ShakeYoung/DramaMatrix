# /Users/yangkang/Library/CloudStorage/OneDrive-共享的库-onedrive/own_project/DramaMatrix/code/src/agents/agent8_analytics.py
import sqlite3
import random
import json
from src.state import DramaState, MarketFeedback, EpisodeState
from src.db import DB_PATH

def query_market_trends() -> dict:
    """
    Simulate Insight Agent database mining from SQLite.
    In reality, you'd aggregate real CPA/conversion data.
    Here we fake some data and "query" it, or return a default insight.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Hypothetical query to find the best performing tags recently
        cursor.execute("SELECT tags, AVG(completion_rate) as avg_comp FROM analytics GROUP BY tags ORDER BY avg_comp DESC LIMIT 1")
        row = cursor.fetchone()
        conn.close()
        
        if row and row[0]:
            best_tags = json.loads(row[0]) if row[0].startswith('[') else [row[0]]
            return {
                "trend": f"数据库挖掘出炉！\n         - 最优标签系: {best_tags}\n         - 历史平均完播率: {row[1]:.2f}\n         => 策略路由: 下一次寻找同类竞品。",
                "tags": best_tags
            }
    except Exception as e:
        print(f"      [Analytics] DB query error: {e}")
        
    return {
        "trend": "暂无显著历史数据，建议尝试反套路的轻松向甜宠或脑洞系统文。",
        "tags": ["萌宝", "系统", "女频"]
    }


def process_agent8_analytics(state: DramaState) -> DramaState:
    """
    Agent 8: 数据大脑 (Analytics Feedback Agent) -> Insight Agent (Proprietary Database Mining)
    接收市场投放的数据回流存入数据库，并提炼指导下一次选品的标签偏好。
    """
    print("--- [Agent 8: Insight Agent (Analytics / Database Mining)] ---")
    
    has_growth_ready = any(ep.status == "growth_ready" for ep in state["episodes"].values())
    if not has_growth_ready:
        print("尚无投流测试完成的剧集，暂时无法收集市场反馈。")
        return state
        
    print("正在连接 SQLite 数据库拉取各大平台投放消耗与转化核心指标...")
    
    # Simulate writing results of current run back to Analytics DB
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        # Fake metrics for the current run
        mock_cpa = random.uniform(5.0, 20.0)
        mock_comp = random.uniform(0.1, 0.4)
        current_tags = state.get("meta_info", {}).get("genre_tags", ["未知"])
        
        cursor.execute(
            "INSERT INTO analytics (ep_id, views, cpa, completion_rate, tags) VALUES (?, ?, ?, ?, ?)",
            ("ep_01", random.randint(1000, 50000), mock_cpa, mock_comp, json.dumps(current_tags, ensure_ascii=False))
        )
        conn.commit()
        conn.close()
        print(f"-> 本期数据 (消耗 CPA: ¥{mock_cpa:.2f}, 有效完播率: {mock_comp*100:.1f}%) 已挂载并写入 Data Lake 结构。")
    except Exception as e:
        print(f"-> 写入分析数据库失败: {e}")
    
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
    state["task_cycle"] = state.get("task_cycle", 1) + 1
    # 新市场周期开始时重置换书尝试次数，避免上一周期消耗压缩本周期额度（F9）
    state["scout_attempts"] = 0
    state["system_status"] = "cycle_completed_ready_for_next"
    
    print(f"✅ 市场偏好提取完毕。建议下期标签: {feedback.suggested_tags}")
    
    return state
