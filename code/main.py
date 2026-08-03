import os
from dotenv import load_dotenv
from src.db import db_get_project_state_snapshot, db_save_project_state
from src.graph import build_drama_matrix_graph
from src.project_state import new_project_state, restore_project_state
from src.text_model import has_text_model_credentials

# 加载环境变量（文本模型和 Agnes 视频模型共用该配置文件）
load_dotenv()

def main():
    if not os.getenv("AGNES_API_KEY"):
        print("⚠️ 未检测到 AGNES_API_KEY。请在 .env 中配置后再运行真实视频生产流程。")
        return
    if not has_text_model_credentials():
        print("⚠️ 未检测到文本模型密钥；Agent 2–4 将使用其现有的回退逻辑。")
        
    app = build_drama_matrix_graph()
    
    project_id = os.getenv("DRAMAMATRIX_PROJECT_ID", "Drama_20260307_001")
    resume_enabled = os.getenv("DRAMAMATRIX_RESUME", "1").strip().lower() not in {"0", "false", "no"}
    snapshot = db_get_project_state_snapshot(project_id) if resume_enabled else None
    if snapshot:
        initial_state_dict = restore_project_state(snapshot)
        print(f"==========↩️ 恢复项目 {project_id}（上次阶段: {snapshot['system_status']}）==========")
    else:
        initial_state_dict = new_project_state(project_id)

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
