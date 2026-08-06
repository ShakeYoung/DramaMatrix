import os
from dotenv import load_dotenv
from src.db import db_get_project_state_snapshot, db_insert_run_log, db_save_project_state
from src.graph import build_drama_matrix_graph
from src.agnes_video import AgnesVideoClient, AgnesVideoError, AgnesVideoSettings
from src.network import configure_proxy_environment
from src.project_state import new_project_state, restore_project_state
from src.runtime_options import apply_runtime_options, parse_runtime_options
from src.text_model import has_text_model_credentials

# 加载环境变量（文本模型和 Agnes 视频模型共用该配置文件）
load_dotenv()


def main(argv=None):
    configure_proxy_environment()
    apply_runtime_options(parse_runtime_options(argv))
    if not os.getenv("AGNES_API_KEY"):
        # 不再硬阻塞：允许纯文本阶段（选品/立项/编剧/分镜）先行运行，
        # 视频生成阶段 Agent 5 会在缺少密钥时优雅地进入 render_failed。
        print("⚠️ 未检测到 AGNES_API_KEY。纯文本阶段（选品→分镜）仍可运行；")
        print("   视频生成阶段（Agent 5）将进入 render_failed，需配置密钥后重跑。")
    if os.getenv("DRAMAMATRIX_AGNES_PREFLIGHT_ONLY") == "1":
        try:
            AgnesVideoClient(AgnesVideoSettings.from_environment()).preflight()
        except AgnesVideoError as exc:
            print(f"❌ {exc}")
            return 2
        return 0
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
            # 结构化运行日志（阶段4）
            try:
                db_insert_run_log(
                    project_id=current_state.get("project_id"),
                    node=node_name,
                    system_status=current_state.get("system_status"),
                    cycle=current_state.get("task_cycle"),
                    event="node_transition",
                )
            except Exception:
                pass
            print(f"\n[状态流转] 当前刚离开节点: {node_name}")
            print(f" -> 当前系统阶段 (system_status): {updated_state.get('system_status')}")
            print("-" * 50)
            
    db_save_project_state(current_state)
    final_status = current_state.get("system_status", "")
    print(f"\n==========最终系统状态: {final_status} ==========")
    # P2-2：终态为 blocked_*/waiting_* 时返回非零退出码，便于调度器/守护进程识别未完成。
    if final_status.startswith(("blocked_", "waiting_", "failed")):
        print("⚠️ 流程未正常完成（阻塞/等待/失败），退出码 2。")
        return 2
    return 0
    
if __name__ == "__main__":
    raise SystemExit(main())
