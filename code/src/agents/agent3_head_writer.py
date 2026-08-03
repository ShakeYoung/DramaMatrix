# /Users/yangkang/Library/CloudStorage/OneDrive-共享的库-onedrive/own_project/DramaMatrix/code/src/agents/agent3_head_writer.py
import os
from pydantic import BaseModel, Field
from typing import List
from src.state import DramaState, EpisodeState, EpisodeScriptData

# Using standard ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser

class TimelineEpisode(BaseModel):
    ep_id: str = Field(description="集数编号, e.g., 'ep_01'")
    chronological_event: str = Field(description="时间线发生的具体事件")
    outline: str = Field(description="本集的核心剧情梗概")
    ending_hook: str = Field(description="本集结尾的悬念/钩子")

class MasterScriptReport(BaseModel):
    master_script_outline: str = Field(description="全剧故事主线大纲")
    episodes: List[TimelineEpisode] = Field(description="基于时间线拆分的单集列表")

def process_agent3_head_writer(state: DramaState) -> DramaState:
    """
    Agent 3: 主编剧 (Head Writer Agent) -> Report Agent (Timeline Analysis & Master Script)
    进行全剧的时间线梳理，并利用大模型依据时间线切分集数。
    """
    print("--- [Agent 3: Report Agent (主编剧)] ---")
    
    report = state.get("source_material", {}).get("report")
    if not report or not report.is_approved:
        print("源头素材未通过评估，无法编写剧本。")
        return state
        
    source_title = state.get("meta_info", {}).get("source_title", "未知短剧")
    raw_text = state.get("source_material", {}).get("raw_text", "")
    
    print(f"开始对《{source_title}》进行时间线分析 (Timeline Analysis) 与剧本拆解 (Chunked Reflection)...")
    
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o")
    
    
    parser = PydanticOutputParser(pydantic_object=MasterScriptReport)
    
    system_prompt = f"""你是一个专业的爆款短剧编剧 (Report Agent)。
你需要对传入的小说内容进行全局的【时间线分析 Timeline Analysis】。
请将主角的核心复仇/打脸路径按照时间顺序进行规划，整部剧的总集数请控制在 **30集左右**，使节奏紧凑、绝不拖沓。
基于这个整体规划，请为我详细拆解并输出 **前 5 到 10 集** 的具体内容。
每个小节产出：基于时间线的具体事件核心、剧本概览、末尾悬念（必须强剧情、高悬疑）。

{parser.get_format_instructions()}"""

    human_prompt = f"项目名称：《{source_title}》\n来源文本：\n{raw_text[:2000]}"

    try:
        llm = ChatOpenAI(model=model_name, temperature=0.7)
        
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ])
        result: MasterScriptReport = parser.invoke(response)
        
        state["master_script_outline"] = result.master_script_outline
        
        print("\n\n====== [Report Agent: Master Script Outline] ======")
        print(result.master_script_outline)
        print("===================================================\n")
        
        # Populate Episodes into DramaState
        for ep in result.episodes:
            # Type cast to EpisodeScriptData for correct Model validation
            script_data = EpisodeScriptData(
                ep_id=ep.ep_id,
                outline=ep.outline,
                ending_hook=ep.ending_hook
            )
            ep_state = EpisodeState(
                script_data=script_data,
                storyboard_data=[],
                feedback_log=[],
                status="script_done"
            )
            state["episodes"][ep.ep_id] = ep_state
            
            print(f"  -> 生成拆解剧集: {ep.ep_id}")
            print(f"     [本集大纲]: {ep.outline}")
            print(f"     [末尾悬念]: {ep.ending_hook}")
        
        print(f"✅ 完成 Final Report: 共切分 {len(result.episodes)} 集。")
        
    except Exception as e:
        print(f"      [Report Agent] 大模型调用失败或无API key: {e}")
        print("      使用内置回退方案生成主梗概和时间线...")
        
        state["master_script_outline"] = "全剧共80集。主线基于复仇和身份反转。"
        
        # fallback episodes
        ep_01_script = EpisodeScriptData(
            ep_id="ep_01",
            outline="女主雨中被男主抛弃，绝望中被神秘豪车接走",
            ending_hook="豪车窗户降下，竟然是男主的死对头..."
        )
        ep_02_script = EpisodeScriptData(
            ep_id="ep_02",
            outline="豪车内，死对头递给女主一份对赌协议。女主换装重返宴会打脸男主。",
            ending_hook="原配男主看到焕然一新的女主，震惊地摔碎了酒杯..."
        )
        state["episodes"]["ep_01"] = EpisodeState(status="script_done", script_data=ep_01_script)
        state["episodes"]["ep_02"] = EpisodeState(status="script_done", script_data=ep_02_script)
        print("✅ 使用内建兜底方案生成了全剧拆解与前两集大纲。")

    state["system_status"] = "ready_for_storyboard"
    return state
