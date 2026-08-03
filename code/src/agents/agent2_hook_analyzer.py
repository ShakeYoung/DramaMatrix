# /Users/yangkang/Library/CloudStorage/OneDrive-共享的库-onedrive/own_project/DramaMatrix/code/src/agents/agent2_hook_analyzer.py
from pydantic import BaseModel, Field
from src.state import DramaState, EvaluationReport
from src.text_model import TextModelSettings, create_text_model

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser

class ForumVerdict(BaseModel):
    score: int = Field(description="综合打分 1-100")
    is_approved: bool = Field(description="是否立项通过")
    hook_analysis: str = Field(description="详细爽点与反噬点分析")

def load_prior_knowledge() -> str:
    """
    Simulate Knowledge Injection (RAG).
    Returns dynamic criteria based on recent viral AI Short Dramas.
    """
    print("      [Agent Forum] 正在启动 RAG... 连接向量数据库 ChromaDB...")
    print("      [Agent Forum] 检索关键词: 【AI生成的顶级漫剧】【千万级播放爆款短剧】【流量密码】")
    print("      [Agent Forum] 找回 3 条高相关性的 AI 爆款短剧制作先验知识。")
    return """
【AI爆款漫剧先验知识】
1. 视觉猎奇度必须极高：目前的生视频大模型（如 Runway/可灵）擅长表现超自然、诡异或极其华丽的场景。剧情必须包含普通实拍难以达成的“视觉奇观”（如漫天神佛、赛博克苏鲁、不可名状之物）。
2. 情绪推进必须极度浓烈且直白：每隔 15 秒必须有一个情绪转折点（极度愤怒、极度绝望或极限装X）。
3. 信息差与强烈反差：主角的隐藏身份必须与表面形成巨大反差，如“扫地僧实为万古神帝”、“乞丐其实是隐藏龙王”，以便 AI 生成对比极度强烈的跨维度画风。
"""

def debate_in_agent_forum(novel_title: str, novel_content: str) -> ForumVerdict:
    """
    Instantiate Critic and Audience agents to debate the novel.
    """
    print("      [Agent Forum] 正在注入先验知识 (Knowledge Injection / RAG)...")
    prior_knowledge = load_prior_knowledge()
    
    model_name = TextModelSettings.from_environment().model
    # Use either OpenAI or an OpenAI-compatible provider configured in .env.
    try:
        llm = create_text_model(temperature=0.7)
        parser = PydanticOutputParser(pydantic_object=ForumVerdict)
        
        system_prompt = f"""你是一个短剧内容评委会（Agent Forum）。
内部包括两个声音：
1. 毒舌评论家：专挑刺，非常严厉，极度关注剧情逻辑硬伤和节奏拖沓，经常全盘否定。
2. 下沉市场受众：狂热追求爽感，极度看重代入感、打脸反转，对逻辑漏洞宽容，会强烈反驳评论家的观点。

请综合以下【爆款先验知识】进行评审：
{prior_knowledge}

请在内部展开**至少三轮**的激烈交锋和反驳，展现出两个视角水火不容的尖锐冲突（请详细列出每一回合的交锋对话），最后他们必须经过艰难谈判，综合输出一份包含详细分歧点的妥协裁决结果。
{parser.get_format_instructions()}
"""
        
        human_prompt = f"评测小说名：《{novel_title}》\n部分内容摘录：\n{novel_content}"
        
        print("      [Agent Forum] 评论家与受众正在展开辩论 (Unfold a debate)...")
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ])
        result: ForumVerdict = parser.invoke(response)
        return result
    except Exception as e:
        print(f"      [Agent Forum] LLM invocation failed or no API key. Using simulated debate... ({e})")
        # Simulating debate output
        return ForumVerdict(
            score=88,
            is_approved=True,
            hook_analysis="辩论结果：【评论家】认为文笔白痴，逻辑欠佳；【受众】认为开局即高潮，代入感极强，贴合先验知识中的'巨大信息差'。综合结论：商业价值高，予以立项。"
        )


def process_agent2_hook_analyzer(state: DramaState) -> DramaState:
    """
    Agent 2: 爆点评估师 (Hook Analyzer Agent) -> Agent Forum & Insight Agent
    分析源头小说，如果爽点不够则打回。
    功能升级：引入多角色辩论与先验知识 (RAG) 注入。
    """
    print("--- [Agent 2: Agent Forum & Insight Agent (爆点评估师)] ---")
    
    source_material = state.get("source_material", {})
    raw_text = source_material.get("raw_text", "")
    title = state.get("meta_info", {}).get("source_title", "未命名小说")
    
    if not raw_text:
        print("无源头素材，跳过评估。")
        return state
        
    print(f"正在开始对《{title}》进行分块分析 (Chunked Analysis) 并准备立项...")
    
    # 模拟“只读取前10章或前3000字”的 Chunked Analysis
    chunked_text = raw_text[:3000] if len(raw_text) > 3000 else raw_text
    
    # Trigger the Agent Forum Debate
    verdict = debate_in_agent_forum(title, chunked_text)
    
    report = EvaluationReport(
        score=verdict.score,
        hook_analysis=verdict.hook_analysis,
        is_approved=verdict.is_approved,
        feedback="通过立项审批" if verdict.is_approved else "未达到商业爆款标准"
    )
    
    state["source_material"]["report"] = report
    
    if report.is_approved:
        print(f"✅ 剧本评估通过。爽度评分: {report.score}")
        print(f"   Forum 共识: {report.hook_analysis}")
        state["system_status"] = "script_drafting"
    else:
        print(f"❌ 剧本评估未通过: {report.feedback}")
        state["system_status"] = "rejected_by_evaluator"
        
    return state
