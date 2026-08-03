from typing import Any, Dict, List, Literal, Optional, TypedDict
from pydantic import BaseModel, Field

# ----------------- Sub-models for the specific agent tasks -----------------

class ShotStoryboard(BaseModel):
    """单镜头分镜数据模型 (Data model for a single shot storyboard)"""
    shot_id: str = Field(description="镜头编号, e.g., 's1_01'")
    camera: str = Field(description="机位和运镜, e.g., '特写, 静态 (Close-up, Static)'")
    visual_prompt: str = Field(description="给视频生成模型的英文提示词 (Visual Prompt for Seedance/Kling)")
    dialogue: str = Field(description="角色台词 (Dialogue)")
    duration: str = Field(description="镜头持续时间，推荐 3s ~ 5s (Duration)")
    audio: str = Field(description="音效与配乐暗示 (Audio hints)")

class EpisodeScriptData(BaseModel):
    """单集剧本大纲数据模型"""
    ep_id: str = Field(description="集数编号, e.g., 'ep_01'")
    outline: str = Field(description="本集的核心剧情梗概")
    ending_hook: str = Field(description="本集结尾的悬念/钩子")

class EvaluationReport(BaseModel):
    """小说爆点评估与可行性报告 (Agent 2)"""
    score: int = Field(description="爽度与可行性综合打分 (1-100)")
    hook_analysis: str = Field(description="核心爽点、打脸、悬念分析")
    is_approved: bool = Field(description="是否通过立项")
    feedback: str = Field(description="给剧本嗅探者的退回原因或修改建议")

class FeedbackLog(BaseModel):
    """反馈日志记录模型"""
    from_agent: str = Field(description="反馈发送方, 如 'Agent_5_Director'")
    to_agent: str = Field(description="反馈接收方, 如 'Agent_4_Storyboard'")
    reason_code: str = Field(description="错误代码, 如 'RENDER_FAIL'")
    message: str = Field(description="详细反馈信息或建议")


class GeneratedVideoAsset(BaseModel):
    """Agnes 为单个分镜生成的视频及其本地副本。"""
    shot_id: str
    video_id: str
    task_id: Optional[str] = None
    status: str
    prompt: str
    remote_url: Optional[str] = None
    local_path: Optional[str] = None
    error: Optional[str] = None


class GrowthAsset(BaseModel):
    """可投放的短视频切片。"""
    name: str
    path: str
    start_seconds: float
    duration_seconds: float

class EpisodeState(BaseModel):
    """单集的完整状态字典模式 (在图节点中流转的状态封装)"""
    status: Literal[
        "pending_script",
        "script_done",
        "storyboard_done",
        "rendering",
        "render_pending",
        "director_rejected",
        "render_failed",
        "video_generated",
        "editing_failed",
        "edit_completed",
        "growth_failed",
        "growth_ready"
    ] = Field(default="pending_script")
    script_data: Optional[EpisodeScriptData] = Field(default=None)
    storyboard_data: List[ShotStoryboard] = Field(default_factory=list)
    feedback_log: List[FeedbackLog] = Field(default_factory=list)
    video_assets: List[GeneratedVideoAsset] = Field(default_factory=list)
    final_video_path: Optional[str] = None
    growth_assets: List[GrowthAsset] = Field(default_factory=list)


class MarketFeedback(BaseModel):
    """市场投流数据反馈 (Agent 8)"""
    trend_analysis: str = Field(description="近期市场爆款元素分析")
    suggested_tags: List[str] = Field(description="建议后续抓取的小说标签")


# ----------------- The Global State for LangGraph -----------------
# 我们使用 TypedDict，因为 LangGraph 依赖于它来进行键级别更新和状态编织

class DramaState(TypedDict):
    """
    Project-level state object passed around between Agents.
    """
    project_id: str
    meta_info: Dict[str, Any]
    
    # 市场反馈数据 (由 Agent 8 注入，供 Agent 1 选品指导)
    market_feedback: Optional[MarketFeedback]
    
    # 源头素材 (由 Agent 1 抓取，Agent 2 评估)
    source_material: Dict[str, Any] # e.g. {"raw_text": "...", "report": EvaluationReport}
    
    # 总剧本架构 (由 Agent 3 拆解)
    master_script_outline: str
    
    # 按照集数 (ep_01, ep_02...) 进行状态切分
    # 若在LangGraph中需要进行增量更新字典，可用 Annotated[Dict, sum]
    # 我们暂时直接整体覆盖某一集状态，以简化逻辑
    episodes: Dict[str, EpisodeState]
    
    # 全局系统卡点状态 (如 starting, drafting, blocked, done)
    system_status: str
