import os
from dotenv import load_dotenv
from src.db import db_save_project_state
from src.graph import build_drama_matrix_graph

# 加载环境变量 (读取 .env 获取 OPENAI_API_KEY)
load_dotenv()

def main():
    if not os.getenv("AGNES_API_KEY"):
        print("⚠️ 未检测到 AGNES_API_KEY。请在 .env 中配置后再运行真实视频生产流程。")
        return
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ 未检测到 OPENAI_API_KEY；Agent 2–4 将使用其现有的回退逻辑。")
        
    app = build_drama_matrix_graph()
    
    # 初始化一个最初的系统全局 State，从 Agent 1 开始流转
    initial_state_dict = {
        "project_id": "Drama_20260307_001",
        "meta_info": {
            "source_title": "待定",
            "genre_tags": [],
        },
        "market_feedback": None,
        "source_material": {},
        "master_script_outline": "",
        "episodes": {},
        "system_status": "starting"
    }

    # 使用 LangGraph 的 stream 接口或者 invoke 接口运行
    print("==========🚀 开始 DramaMatrix [Agnes 视频生产链路] 🚀==========")
    
    # stream 方法可以让我们逐个节点看到状态变迁
    current_state = initial_state_dict
    for s in app.stream(initial_state_dict, {"recursion_limit": 50}):
        # 获取每次更新后输出的节点名称和对应状态
        for node_name, updated_state in s.items():
            current_state.update(updated_state)
            db_save_project_state(current_state)
            print(f"\n[状态流转] 当前刚离开节点: {node_name}")
            print(f" -> 当前系统阶段 (system_status): {updated_state.get('system_status')}")
            print("-" * 50)
            
    db_save_project_state(current_state)
    print(f"\n==========✅ 最终系统状态: {current_state.get('system_status')} ==========")
    
if __name__ == "__main__":
    main()
