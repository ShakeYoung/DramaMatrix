from typing import Any, Dict, List, Literal, Optional, TypedDict
from pydantic import BaseModel, Field

# ----------------- Sub-models for the specific agent tasks -----------------

class ShotBoundaryState(BaseModel):
    """镜头边界姿态/位置/光线状态（P1-A），用于强制前后镜衔接。"""
    pose: str = Field(default="", description="人物姿态，如'站立背对镜头'")
    subject_position: str = Field(default="", description="主体在画面中的位置，如'画面中央偏左'")
    gaze_direction: str = Field(default="", description="视线方向，如'看向镜头右侧'")
    light_direction: str = Field(default="", description="主光方向，如'顶光/侧逆光'")
    color_temperature: str = Field(default="", description="色温，如'暖黄 3200K'")


class ShotStoryboard(BaseModel):
    """单镜头分镜数据模型 (Data model for a single shot storyboard)"""
    shot_id: str = Field(description="镜头编号, e.g., 's1_01'")
    camera: str = Field(description="机位和运镜, e.g., '特写, 静态 (Close-up, Static)'")
    visual_prompt: str = Field(description="给视频生成模型的英文提示词 (Visual Prompt for Seedance/Kling)")
    dialogue: str = Field(description="角色台词 (Dialogue)")
    duration: str = Field(description="镜头持续时间，推荐 3s ~ 5s (Duration)")
    audio: str = Field(description="音效与配乐暗示 (Audio hints)")
    # P1-A：连续性字段（全部 Optional，旧快照可平滑恢复）
    scene_id: Optional[str] = Field(default=None, description="场景编号，同场景共用参考图与 seed")
    location_id: Optional[str] = Field(default=None, description="地点编号")
    wardrobe_ids: List[str] = Field(default_factory=list, description="本镜出场角色的服装编号")
    time_of_day: Optional[str] = Field(default=None, description="时段：日/夜/黄昏")
    weather: Optional[str] = Field(default=None, description="天气")
    light_direction: Optional[str] = Field(default=None, description="主光方向")
    color_temperature: Optional[str] = Field(default=None, description="色温")
    start_pose: Optional[str] = Field(default=None, description="起始姿态")
    end_pose: Optional[str] = Field(default=None, description="结束姿态")
    subject_position: Optional[str] = Field(default=None, description="主体画面位置")
    gaze_direction: Optional[str] = Field(default=None, description="视线方向")
    motion_direction: Optional[str] = Field(default=None, description="运动方向")
    previous_shot_id: Optional[str] = Field(default=None, description="上一镜编号")
    transition_type: Optional[str] = Field(default=None, description="计划转场：hard_cut/crossfade/flash")
    start_state: Optional[ShotBoundaryState] = Field(default=None, description="起始边界状态")
    end_state: Optional[ShotBoundaryState] = Field(default=None, description="结束边界状态")

class EpisodeScriptData(BaseModel):
    """单集剧本大纲数据模型"""
    ep_id: str = Field(description="集数编号, e.g., 'ep_01'")
    outline: str = Field(description="本集的核心剧情梗概")
    ending_hook: str = Field(description="本集结尾的悬念/钩子")


class CharacterSheet(BaseModel):
    """单角色一致性描述表，供分镜与视频生成时的角色身份保持（阶段3）。"""
    name: str = Field(description="角色名")
    appearance: str = Field(description="外形与服化描述，例如'红衣黑发青年，佩戴玉坠'")
    signature: str = Field(description="标志性特征/口头禅，用于稳定身份")
    role: str = Field(default="", description="角色定位，如'男主/女主/反派'")
    # P0-B：角色圣经补充字段
    wardrobe_id: str = Field(default="", description="服装编号，便于跨镜服化锁定")
    reference_image_prompt: str = Field(default="", description="生成该角色参考图的提示词")
    appears_in: List[str] = Field(default_factory=list, description="出场集号列表，如 ['ep_01','ep_02']")
    # R6：独立 canonical 身份字段（不再兼用 signature）
    character_id: str = Field(default="", description="稳定角色 ID，如 char_01")
    canonical_name: str = Field(default="", description="归一化后的标准名")
    aliases: List[str] = Field(default_factory=list, description="别名/音译变体列表")

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
    # V1：完整性与版本证据（全 Optional，旧快照可平滑恢复）
    sha256: Optional[str] = Field(default=None, description="本地文件 SHA-256")
    file_size_bytes: Optional[int] = Field(default=None, description="文件大小（字节）")
    actual_duration: Optional[float] = Field(default=None, description="ffprobe 实测时长（秒）")
    width: Optional[int] = Field(default=None)
    height: Optional[int] = Field(default=None)
    frame_rate: Optional[float] = Field(default=None)
    bit_rate: Optional[int] = Field(default=None)
    has_audio: Optional[bool] = Field(default=None, description="是否含音轨")
    audio_duration: Optional[float] = Field(default=None)
    seed: Optional[int] = Field(default=None, description="实际使用的 seed")
    negative_prompt: Optional[str] = Field(default=None)
    reference_image_url: Optional[str] = Field(default=None, description="使用的参考图 URL/data URI")
    reference_image_sha256: Optional[str] = Field(default=None)
    model_version: Optional[str] = Field(default=None, description="Agnes 模型版本")
    downloaded_at: Optional[float] = Field(default=None, description="下载完成 unix 时间戳")
    agnes_response_summary: Optional[Dict[str, Any]] = Field(default=None, description="Agnes 原始响应摘要")


class GrowthAsset(BaseModel):
    """可投放的短视频切片。"""
    name: str
    path: str
    start_seconds: float
    duration_seconds: float
    # 投流元数据：用于上架货架/信息流的标题、简介与标签
    headline: Optional[str] = Field(default=None, description="切片用作广告时的标题文案")
    description: Optional[str] = Field(default=None, description="切片简介")
    tags: Optional[List[str]] = Field(default=None, description="推荐投放标签")


class GrowthMeta(BaseModel):
    """单集成片投流所需的发布元数据（标题/简介/标签/封面提示）。"""
    title: str = Field(description="成片发布标题")
    description: str = Field(description="成片发布简介/导语")
    tags: List[str] = Field(default_factory=list, description="发布推荐标签")
    cover_prompt: str = Field(description="封面图生成提示词")

class EpisodeState(BaseModel):
    """单集的完整状态字典模式 (在图节点中流转的状态封装)"""
    status: Literal[
        "pending_script",
        "script_done",
        "storyboard_done",
        "rendering",
        "render_pending",
        "render_partial",
        "submission_uncertain",
        "waiting_for_agnes_capacity",
        "waiting_for_connectivity",
        "storyboard_blocked",
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
    # 本集涉及的角色一致性表（阶段3）
    characters: List[CharacterSheet] = Field(default_factory=list)
    # 生产过程中附加的要素（阶段2）：配音/混音音轨与烧录后的字幕路径
    audio_track: Optional[str] = Field(default=None)
    subtitle_track: Optional[str] = Field(default=None)
    # 投流发布元数据（Agent 7）
    growth_meta: Optional[GrowthMeta] = Field(default=None)
    # 分镜版本（P0-A）：recovery 重写时递增，隔离新旧镜头文件
    storyboard_version: int = Field(default=1)
    # R10：队列/连接等待的恢复调度字段（next_retry_at 为 unix 时间戳，0 表示立即可重试）
    next_retry_at: float = Field(default=0.0)
    queue_retry_count: int = Field(default=0)
    last_queue_error: Optional[str] = Field(default=None)
    # 调试/受控渲染可能只生成分镜子集；记录实际完成进度，禁止误入 Agent6。
    rendered_shot_count: int = Field(default=0)
    planned_shot_count: int = Field(default=0)
    # V1：实际渲染的镜头资产（含哈希/真实媒体参数/seed/参考图等完整性证据）
    rendered_assets: List[GeneratedVideoAsset] = Field(default_factory=list)


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
    
    # 全剧循环控制：标记整部短剧正处于第几次"选品→成片→投流"周期，
    # 以及截至当前周期所尝试过的源头素材数（用于被否换书与市场回环的上限约束）。
    task_cycle: int
    scout_attempts: int
    
    # 全剧角色一致性表（阶段3）：由 Agent 3 抽取，Agent 4/5 注入生成提示
    characters: List[CharacterSheet]

    # V4：运行上下文（git sha/配置快照/依赖版本），随状态持久化用于复现
    run_context: Optional[Dict[str, Any]]

    # 全局系统卡点状态 (如 starting, drafting, blocked, done)
    system_status: str
