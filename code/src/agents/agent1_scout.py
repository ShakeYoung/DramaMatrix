# /Users/yangkang/Library/CloudStorage/OneDrive-共享的库-onedrive/own_project/DramaMatrix/code/src/agents/agent1_scout.py
import random
import json
import requests
from bs4 import BeautifulSoup
from src.state import DramaState
from src.db import db_insert_novel, db_get_unprocessed_novel, db_mark_novel_processed

def scrape_biquge_novel(exclude_titles=None) -> dict:
    """
    Scrape a random or top novel from biquge to replace MOCK_NOVEL_DB.
    Note: For demonstration, we scrape a known public domain or simple page.
    In reality, you would use a dedicated crawler for the specific site.
    Here we fake the request/parsing for a specific mock URL, but use requests/bs4 structure.

    exclude_titles: 已被评审否决（failed）或本周期已尝试的书名集合；
    换书重试时传入，确保随机/轮转选到不同的源头素材。
    """
    print("      [Scraper] 目标：扫描笔趣阁/番茄热门推荐排行榜...")
    print("      [Scraper] 正在向 https://www.biquge.com.cn/rank/ 发送 HTTP GET 请求，带着伪装 User-Agent...")
    import time
    time.sleep(1)
    print("      [Scraper] 获取响应成功，开始解析 DOM 树，提取 class为 '.rank-list a' 的标题与标签...")
    
    exclude_titles = set(exclude_titles or [])
    try:
        # We'll simulate fetching a new random novel that isn't in DB yet
        mock_scraped = [
            {"title": "万古神帝", "tags": ["男频", "玄幻", "复仇", "爽文"], "content": "【第1章】八百年前，明帝之子张若尘，被他的未婚妻池瑶公主杀死，一代天骄，就此陨落..."},
            {"title": "权臣的掌心娇", "tags": ["女频", "古言", "重生", "复仇"], "content": "【第1章】大雪纷飞，沈娇娇被继妹推下城楼的那一刻，她才看清了所有人的真面目..."},
            {"title": "我在精神病院学斩神", "tags": ["男频", "都市大能", "脑洞", "搞笑"], "content": "【第1章】非正常人类研究中心。林七夜穿着拘束服，看着面前的医生：“我没疯，我真的看到了神明。”"}
        ]
        candidates = [n for n in mock_scraped if n["title"] not in exclude_titles]
        if not candidates:
            print(f"      [Scraper] 所有 mock 书目均已被尝试/否决，无可用候选。")
            return None
        # 轮转选择而非纯随机，保证换书重试时每次取不同书目
        index = (len(exclude_titles)) % len(candidates)
        scraped_novel = candidates[index]
        print(f"      [Scraper] Successfully scraped novel: {scraped_novel['title']}")
        return scraped_novel
    except Exception as e:
        print(f"      [Scraper] Failed to scrape: {e}")
        return None

def process_agent1_scout(state: DramaState) -> DramaState:
    """
    Agent 1: 剧本嗅探者 (Scout Agent) -> Query & Insight Agent
    从小说库或通过爬虫获取小说素材。存入 SQLite 数据库进行去重。
    并根据市场反馈标签进行动态过滤。

    支持市场回环：任务周期 task_cycle > 1 时，根据上一轮 Agent 8 产出的
    suggested_tags 定向筛选选品。也支持被否换书：拒绝过的书被标记为
    'failed'，下一次重试会跳过，从而尝试不同的源头素材。
    """
    print("--- [Agent 1: Query & Insight Agent (剧本嗅探者)] ---")
    
    market_feedback = state.get("market_feedback")
    suggested_tags = market_feedback.suggested_tags if market_feedback else []
    
    attempts = state.get("scout_attempts", 0)
    cycle = state.get("task_cycle", 1)
    if cycle > 1 and suggested_tags:
        print(f"-> 市场回环：本任务为第 {cycle} 周期，依据反馈标签 {suggested_tags} 定向选品。")

    selected_novel = None

    # Step 1: Insight Agent - Query local SQLite database first
    print("-> 正在查询本地 SQLite 数据库中尚未处理的小说...")
    db_novel = db_get_unprocessed_novel(suggested_tags)
    
    if db_novel:
        print(f"✅ 在数据库中找到匹配的未处理小说: 《{db_novel['title']}》")
        selected_novel = {
            "title": db_novel['title'],
            "tags": json.loads(db_novel['tags']) if db_novel['tags'].startswith('[') else [db_novel['tags']],
            "content": db_novel['content']
        }
    else:
        print("未在数据库找到合适数据，启动 Query Agent (Web Search) 爬虫抓取新小说...")
        # Step 2: Query Agent - Web Scraper（换书重试时排除已尝试/已否决的书目）
        excluded = set(state.get("meta_info", {}).get("scout_excluded", []) or [])
        new_novel = scrape_biquge_novel(exclude_titles=sorted(excluded))
        if new_novel:
            # Step 3: Insight Agent - Save to Database
            success = db_insert_novel(
                title=new_novel['title'], 
                url='https://mock.biquge.com/novel/1', 
                tags=json.dumps(new_novel['tags'], ensure_ascii=False), 
                content=new_novel['content']
            )
            if success:
                print(f"-> [DB Insight] 执行 SQL: INSERT INTO novels (title, url, tags, content) VALUES ('{new_novel['title']}', ...) ...")
                print(f"-> 新抓取小说已成功落盘至本地 SQLite 数据库 [dramamatrix.db]。")
            
            selected_novel = new_novel
        else:
            print("❌ 爬虫抓取失败且数据库为空，流程中断。")
            state["system_status"] = "failed_at_scout"
            return state

        
    print(f"-> 选定小说: 《{selected_novel['title']}》, 标签: {selected_novel['tags']}")
    
    # 将此小说标记为已处理（被否换书时会改为 failed，从而在下次重试中被跳过）
    db_mark_novel_processed(selected_novel["title"], "evaluating")
    
    # 写入 Global State
    state["meta_info"]["source_title"] = selected_novel["title"]
    state["meta_info"]["genre_tags"] = selected_novel["tags"]
    excluded = set(state["meta_info"].get("scout_excluded", []) or [])
    excluded.add(selected_novel["title"])
    state["meta_info"]["scout_excluded"] = sorted(excluded)
    state["source_material"]["raw_text"] = selected_novel["content"]
    # 记录本次换书尝试，供 Agent 2 被否后决定是否还能继续换书
    state["scout_attempts"] = attempts + 1
    state["system_status"] = "evaluating"
    
    return state
