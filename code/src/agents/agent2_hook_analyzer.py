# /Users/yangkang/Library/CloudStorage/OneDrive-共享的库-onedrive/own_project/DramaMatrix/code/src/agents/agent2_hook_analyzer.py
from pydantic import BaseModel, Field
from typing import Optional
from src.state import DramaState, EvaluationReport
from src.text_model import TextModelSettings, create_text_model
from src.db import db_mark_novel_processed
from src.runtime_options import is_demo_mode

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser

from src.prompt_files import load_prompt

# U5：系统 prompt 外置到 prompts/agent2_forum.md；此处保留等价默认模板，
# 文件缺失时行为与外置前完全一致（占位符经 replace 填充，非 str.format）。
_AGENT2_FORUM_TEMPLATE = """你是一个短剧内容评委会（Agent Forum）。
内部包括两个声音：
1. 毒舌评论家：专挑刺，非常严厉，极度关注剧情逻辑硬伤和节奏拖沓，经常全盘否定。
2. 下沉市场受众：狂热追求爽感，极度看重代入感、打脸反转，对逻辑漏洞宽容，会强烈反驳评论家的观点。

请综合以下【爆款先验知识】进行评审：
{prior_knowledge}

请在内部展开**至少三轮**的激烈交锋和反驳，展现出两个视角水火不容的尖锐冲突（请详细列出每一回合的交锋对话），最后他们必须经过艰难谈判，综合输出一份包含详细分歧点的妥协裁决结果。
{format_instructions}"""

class ForumVerdict(BaseModel):
    score: int = Field(description="综合打分 1-100")
    is_approved: bool = Field(description="是否立项通过")
    hook_analysis: str = Field(description="详细爽点与反噬点分析")
    simulated: bool = Field(default=False, description="是否为演示模式的模拟评审（非模型产出）")

def load_prior_knowledge(query: str = "", k: int = 3) -> str:
    """U6：真实检索——从 knowledge_entries 表按相关性取 top-k 先验。

    检索源可运营增补（直接向表插入条目）；DB 不可用时回退内置先验，
    行为不劣于升级前的静态文本。
    """
    from src.knowledge_base import load_prior_knowledge as kb_load

    prior_knowledge, entries = kb_load(query, k=k)
    if entries:
        titles = "、".join(entry.get("title") or "?" for entry in entries)
        print(f"      [KnowledgeBase] 检索完成：命中 {len(entries)} 条先验（{titles}）。")
    else:
        print("      [KnowledgeBase] 未命中检索条目，使用内置先验。")
    return prior_knowledge

def debate_in_agent_forum(novel_title: str, novel_content: str) -> Optional[ForumVerdict]:
    """
    Instantiate Critic and Audience agents to debate the novel.

    返回 None 表示 production 模式下评审失败（调用方应阻塞）；demo 模式返回
    带 simulated 标记的模拟评审。
    """
    print("      [Agent Forum] 正在注入先验知识 (Knowledge Injection / 检索式 RAG)...")
    # U6：以书名+摘录构造查询，检索命中的先验才进入评审上下文。
    prior_knowledge = load_prior_knowledge(query=f"{novel_title}\n{novel_content[:500]}")

    model_name = TextModelSettings.from_environment().model
    # Use either OpenAI or an OpenAI-compatible provider configured in .env.
    try:
        llm = create_text_model(temperature=0.7)
        parser = PydanticOutputParser(pydantic_object=ForumVerdict)

        system_prompt = (
            load_prompt("agent2_forum", _AGENT2_FORUM_TEMPLATE)
            .replace("{prior_knowledge}", prior_knowledge)
            .replace("{format_instructions}", parser.get_format_instructions())
        )

        human_prompt = f"评测小说名：《{novel_title}》\n部分内容摘录：\n{novel_content}"

        print("      [Agent Forum] 评论家与受众正在展开辩论 (Unfold a debate)...")
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ])
        result: ForumVerdict = parser.invoke(response)
        return result
    except Exception as e:
        if not is_demo_mode():
            # R1：production 模式不模拟过审——评审失败必须保留失败状态，
            # 否则任何书都会以"88 分自动通过"进入编剧。
            print(f"      [Agent Forum] LLM 调用失败（{e}）。production 模式阻塞，配置密钥后 --resume 重试。")
            return None
        print(f"      [Agent Forum] LLM 调用失败，demo 模式使用模拟评审... ({e})")
        # Simulating debate output
        return ForumVerdict(
            score=88,
            is_approved=True,
            simulated=True,
            hook_analysis="【演示模式-模拟评审】辩论结果：【评论家】认为文笔白痴，逻辑欠佳；【受众】认为开局即高潮，代入感极强，贴合先验知识中的'巨大信息差'。综合结论：商业价值高，予以立项。"
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
    if verdict is None:
        # R1：production 模式评审失败——保留阻塞状态，不写 report（避免被
        # route_after_agent2 误判为"被否换书"而消耗换书额度）。
        state["system_status"] = "blocked_on_text_model"
        return state

    report = EvaluationReport(
        score=verdict.score,
        hook_analysis=verdict.hook_analysis,
        is_approved=verdict.is_approved,
        simulated=verdict.simulated,
        feedback="通过立项审批" if verdict.is_approved else "未达到商业爆款标准"
    )
    
    state["source_material"]["report"] = report
    
    if report.is_approved:
        print(f"✅ 剧本评估通过。爽度评分: {report.score}")
        print(f"   Forum 共识: {report.hook_analysis}")
        state["system_status"] = "script_drafting"
    else:
        print(f"❌ 剧本评估未通过: {report.feedback}")
        # 被否的书标记为 failed，Agent 1 换书重试时会跳过它
        source_title = state.get("meta_info", {}).get("source_title")
        if source_title:
            try:
                db_mark_novel_processed(source_title, "failed")
                print(f"   《{source_title}》已标记为 failed，后续换书将跳过。")
            except Exception as e:
                print(f"   ⚠️ 标记 {source_title} 为 failed 失败：{e}")
        state["system_status"] = "rejected_by_evaluator"
        
    return state
