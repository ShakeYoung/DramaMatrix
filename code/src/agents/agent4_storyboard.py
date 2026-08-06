from typing import List
from pydantic import BaseModel, Field
from src.text_model import TextModelSettings, create_text_model

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from src.state import DramaState, EpisodeState, ShotStoryboard
from src.characters import render_character_block
from src.agnes_video import purge_shot_versions_except
from src.continuity import normalize_continuity

class StoryboardOutput(BaseModel):
    """用于确保 LLM 严格输出列表列表的包装类"""
    shots: List[ShotStoryboard] = Field(description="分镜列表")


def _escape_template_text(value: str) -> str:
    """Escape literal braces before embedding arbitrary text in a LangChain f-string template."""
    return value.replace("{", "{{").replace("}", "}}")


# Prompt Template definition
STORYBOARD_SYSTEM_PROMPT = """你是一个资深的短剧分镜师和AI视频提示词专家。
你的任务是将编剧发来的【剧情梗概和结局悬念】转化为可被生视频大模型直接读取的【Structured Modal Cards (结构化多模态分镜表)】。

【短剧多模态转化规则】：
1. 避免使用抽象文学词汇，必须转化为物理可见画面（如：特写镜头，男人嘴角微微抽动，眼神直视镜头，顶光照明）。
2. 每段生成视频的长度控制在 3-5 秒，动作幅度不宜过大以防模型崩溃。
3. 运镜标签仅限使用：Pan Left/Right, Zoom in/out, Tracking, Static。
4. Audio提示：提供明确的音效暗示（BGM情绪与环境音，例如“沉重的低音提琴，雨声”）。
5. 每个分镜必须连续，总体构成一集的完整叙事，结尾停留在【悬念】处。
6. 单集时长通常在 1-2 分钟，因此请输出大约 15 到 25 个分镜来覆盖这一集的丰满剧情。
"""

STORYBOARD_RECOVERY_PROMPT = """【严重警告：恢复模式】
上一次你的分镜动作描述导致 AI 视频生成严重崩坏。
错误原因如下: {error_message}

请保持原有剧情情绪，但使用【面部特写】、【背影】、或者单纯的【静物局部特写 (如只拍手、掉落的酒杯等)】来重写视觉 Prompt。
切忌带有复杂的全身动作，降低重绘崩溃率。
"""

def process_agent4_storyboard(state: DramaState) -> DramaState:
    """
    Agent 4 Node for LangGraph -> Media Agent (Multimodal content analysis)
    """
    # 我们遍历所有准备好剧本的集数
    target_eps = [k for k, v in state["episodes"].items() if v.status == "script_done" or v.status == "director_rejected"]
    if not target_eps:
        print("没有找到需要分镜转化的剧集。")
        return state
        
    model_name = TextModelSettings.from_environment().model
    parser = PydanticOutputParser(pydantic_object=StoryboardOutput)
    
    for ep_key in target_eps:
        ep_state = state["episodes"][ep_key]
        
        # 检查是否处于被退回重写的状态
        is_recovery = ep_state.status == "director_rejected"
        sys_prompt = STORYBOARD_SYSTEM_PROMPT
        if is_recovery and ep_state.feedback_log:
            last_error = ep_state.feedback_log[-1].message
            sys_prompt += "\n\n" + STORYBOARD_RECOVERY_PROMPT.format(
                error_message=_escape_template_text(last_error)
            )
            print(f"!! 进入 Recovery 模式 (B计划), 处理反馈: {last_error}")
            # P0-A：分镜版本隔离——重写时递增版本号并清掉旧版本镜头目录，
            # 避免状态里 video_assets=[] 但磁盘残留旧 s01~s09 造成新旧混用。
            ep_state.storyboard_version = int(ep_state.storyboard_version or 1) + 1
            try:
                purge_shot_versions_except(
                    state["project_id"], ep_key, ep_state.storyboard_version
                )
                print(f"   已清理旧版本镜头目录，本次渲染写入 shots/v{ep_state.storyboard_version}/")
            except Exception as exc:
                print(f"   ⚠️ 清理旧版本镜头目录失败：{exc}")

        # 阶段3：注入角色一致性表，帮助分镜保持角色身份
        character_block = render_character_block(state.get("characters", []))
        if character_block:
            sys_prompt += "\n\n" + character_block
        
        sys_prompt += "\n\n" + parser.get_format_instructions().replace("{", "{{").replace("}", "}}")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", sys_prompt),
            ("human", "【剧集编号】\n{ep_id}\n\n【本集剧情梗概】\n{outline}\n\n【本集结尾悬念(钩子)】\n{ending_hook}")
        ])
        
        print(f"正在向 {model_name} 提交 {ep_key} 的分镜转化任务...")
        try:
            llm = create_text_model(temperature=0.7)
            chain = prompt | llm | parser
            result: StoryboardOutput = chain.invoke({
                "ep_id": ep_key,
                "outline": ep_state.script_data.outline,
                "ending_hook": ep_state.script_data.ending_hook
            })
            
            # 更新状态
            ep_state.storyboard_data = result.shots
            ep_state.video_assets = []
            ep_state.final_video_path = None
            ep_state.growth_assets = []
            ep_state.status = "storyboard_done"
            
            print(f"✅ 成功生成 {ep_key} 的 {len(result.shots)} 条分镜指令。详细 Prompt 如下：")
            print(f"\n====== [Media Agent: Storyboard Shots for {ep_key}] ======")
            for s in result.shots:
                print(f"  [镜头 {s.shot_id}] 时长: {s.duration}")
                print(f"    - 运镜: {s.camera}")
                print(f"    - 画面: {s.visual_prompt}")
                print(f"    - 对白: {s.dialogue}")
                print(f"    - 音效: {s.audio}")
                print("-" * 40)
            print("=================================================\n")
            
        except Exception as e:
            print(f"      [Media Agent] LLM 链条抛出异常或无API key: {e}")
            print("      使用内置兜底方案生成 Modal Cards...")
            mock_shot = ShotStoryboard(
                shot_id=f"{ep_key}_s01",
                camera="特写镜头, 静态",
                visual_prompt="男人嘴角微微抽动，眼神直视镜头，顶光照明，面部表情凝重。",
                dialogue="你以为你赢定了？",
                duration="4s",
                audio="沉重的低音提琴，雨声渐强"
            )
            ep_state.storyboard_data = [mock_shot]
            ep_state.video_assets = []
            ep_state.final_video_path = None
            ep_state.growth_assets = []
            ep_state.status = "storyboard_done"

        # P1-A：连续性归一化（软约束）——对齐相邻镜边界状态、回填 scene_id，
        # 收集 continuity_warnings 而非硬失败。
        warnings = normalize_continuity(ep_state.storyboard_data)
        if warnings:
            print(f"   ⚠️ 连续性告警 {len(warnings)} 条（已自动归一化）：")
            for w in warnings[:5]:
                print(f"      - {w.shot_id}.{w.field}: {w.message}")

        state["episodes"][ep_key] = ep_state
    
    return state
