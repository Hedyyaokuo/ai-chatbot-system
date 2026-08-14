from __future__ import annotations

import gzip
import json
import math
import os
import re
import threading
from collections import Counter, defaultdict
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, StateGraph
from openai import OpenAI


BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_PATH = BASE_DIR / "original_knowledge.json.gz"
MANIFEST_PATH = BASE_DIR / "knowledge_manifest.json"


class AgentState(TypedDict, total=False):
    session_id: str
    query: str
    effective_query: str
    query_section: str | None
    query_family: str
    retrieval_plan: str
    selected_tool: str
    sub_questions: list[str]
    retrieved_docs: list[dict]
    answer: str
    verification_result: str
    preferences: list[str]
    last_preference_focus: str
    trace: list[str]


def _tokenize(text: str) -> list[str]:
    lowered = text.lower()
    latin = re.findall(r"[a-z0-9][a-z0-9_+-]*", lowered)
    chinese = []
    for run in re.findall(r"[\u4e00-\u9fff]+", lowered):
        if len(run) == 1:
            chinese.append(run)
        else:
            chinese.extend(run[index:index + 2] for index in range(len(run) - 1))
    return latin + chinese


def _contains_any(text: str, words: list[str]) -> bool:
    return any(word in text for word in words)


QUERY_EXPANSIONS = {
    "活动": "event eventnow session",
    "组织者": "organiser organizer event management",
    "参与者": "participant application registration",
    "登录": "login page",
    "注册": "register registration",
    "日本": "japan japanese tokyo",
    "东京": "tokyo japan",
    "夜市": "night market street food",
    "灯笼": "lantern japanese culture",
    "新西兰": "new zealand travel",
    "地图": "map route itinerary",
    "图片": "image photo visual",
    "照片": "image photo visual",
    "焦虑": "anxiety stress panic",
    "呼吸": "breathing exercise relaxation",
    "正念": "mindfulness wellbeing",
    "数据库": "database distributed database",
    "事务": "transaction acid",
    "死锁": "deadlock locking",
    "学习笔记": "study notes handwritten notes",
}


def _expand_query(query: str) -> str:
    additions = [value for key, value in QUERY_EXPANSIONS.items() if key in query]
    return " ".join([query, *additions])


FAMILY_RULES = [
    ("eventnow_project", ["eventnow", "organiser", "organizer", "participant", "session", "application", "dashboard", "login", "register", "profile", "event card", "活动管理", "组织者", "参与者", "报名", "登录", "注册"]),
    ("ai_event_platform", ["ai powered event", "smart event", "event platform", "stripe", "qr code", "rbac", "serverless", "智能活动", "二维码", "无服务器"]),
    ("cloud_event_framework", ["cloud computing", "scalability", "resource management", "predictive analytics", "云计算", "可扩展性", "资源管理", "预测分析"]),
    ("lecture_agents", ["language agent", "planning", "reasoning", "memory", "rag triad", "faithfulness", "groundedness", "智能体", "规划", "推理", "记忆", "忠实度"]),
    ("esports_handbook", ["blast", "counter strike", "bounty", "rivals", "tournament regulation", "电竞手册", "赛事规则"]),
    ("esports_article", ["donk", "hltv", "spirit", "navi", "cs2 prodigy"]),
    ("gaming_visual", ["valorant", "video game", "game poster", "gaming", "roster", "游戏海报", "电竞图片"]),
    ("aespa_visual", ["aespa", "karina", "winter", "ningning", "giselle"]),
    ("bts_vogue_visual", ["bts", "bangtan", "vogue"]),
    ("bts_album_visual", ["love yourself", "bts album", "bts 专辑"]),
    ("japan_map", ["tokyo map", "ota map", "kamata map", "日本地图", "东京地图"]),
    ("japan_travel", ["japan", "japanese", "tokyo", "ota", "kamata", "kimono", "lantern", "street food", "akihabara", "日本", "东京", "和服", "灯笼", "街头美食"]),
    ("new_zealand_travel", ["new zealand", "queenstown", "auckland", "touring map", "kiwi", "新西兰", "皇后镇", "奥克兰"]),
    ("anxiety_workbook", ["anxiety", "stress", "fight or flight", "breathing", "panic", "mindfulness workbook", "焦虑", "压力", "呼吸", "恐慌", "正念"]),
    ("act_worksheet", ["act worksheet", "bullseye", "life compass", "willingness", "defusion", "接纳承诺疗法", "价值练习"]),
    ("study_notes", ["database notes", "study notes", "distributed database", "acid", "transaction", "deadlock", "2pl", "2pc", "数据库笔记", "事务", "死锁"]),
]

SECTION_RULES = [
    ("user_profile", ["profile", "account information", "个人资料", "账户信息"]),
    ("registration", ["registration", "application", "报名", "申请"]),
    ("event_management", ["event card", "event list", "dashboard", "活动列表", "仪表盘"]),
    ("japan_food_culture", ["street food", "night market", "lantern", "街头美食", "夜市", "灯笼"]),
    ("nz_map", ["new zealand map", "itinerary", "新西兰地图", "行程"]),
    ("database_notes", ["database", "acid", "transaction", "deadlock", "数据库", "事务", "死锁"]),
]


def detect_document_family(query: str) -> str | None:
    lowered = query.lower()
    for family, words in FAMILY_RULES:
        if _contains_any(lowered, words):
            return family
    return None


def detect_query_section(query: str) -> str | None:
    lowered = query.lower()
    for section, words in SECTION_RULES:
        if _contains_any(lowered, words):
            return section
    return None


class OriginalKnowledgeIndex:
    def __init__(self, path: Path):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            self.records: list[dict] = json.load(handle)
        self.term_frequencies: list[Counter] = []
        self.document_frequencies: Counter = Counter()
        self.lengths: list[int] = []

        for record in self.records:
            tokens = _tokenize(record.get("content", ""))
            frequencies = Counter(tokens)
            self.term_frequencies.append(frequencies)
            self.lengths.append(len(tokens))
            self.document_frequencies.update(frequencies.keys())
        self.average_length = sum(self.lengths) / max(1, len(self.lengths))

    def _allowed(self, record: dict, selected_tool: str) -> bool:
        modality = record.get("modality", "text")
        if selected_tool == "text_retrieval":
            return modality == "text"
        if selected_tool == "image_caption_retrieval":
            return modality == "image_caption"
        if selected_tool == "true_image_retrieval":
            return bool(record.get("has_true_image_embedding"))
        return modality in {"text", "image_caption"}

    def search(
        self,
        query: str,
        selected_tool: str,
        family_hint: str | None,
        section_hint: str | None,
        preferences: list[str],
        excluded_sources: list[str],
        limit: int = 4,
    ) -> list[dict]:
        query_tokens = _tokenize(_expand_query(query) + " " + " ".join(preferences))
        if not query_tokens:
            return []

        scores = []
        document_count = len(self.records)
        for index, frequencies in enumerate(self.term_frequencies):
            record = self.records[index]
            if not self._allowed(record, selected_tool):
                continue

            score = 0.0
            length = self.lengths[index]
            for token in query_tokens:
                frequency = frequencies.get(token, 0)
                if not frequency:
                    continue
                document_frequency = self.document_frequencies[token]
                inverse_frequency = math.log(
                    1 + (document_count - document_frequency + 0.5)
                    / (document_frequency + 0.5)
                )
                denominator = frequency + 1.2 * (
                    0.25 + 0.75 * length / max(1, self.average_length)
                )
                score += inverse_frequency * frequency * 2.2 / denominator

            metadata_text = " ".join([
                str(record.get("source_file", "")),
                str(record.get("document_family", "")),
                str(record.get("section", "")),
                str(record.get("tags", "")),
                str(record.get("visual_type", "")),
            ]).lower()
            metadata_hits = sum(token in metadata_text for token in query_tokens)
            score += metadata_hits * 1.8
            if family_hint and record.get("document_family") == family_hint:
                score += 8.0
            if section_hint and record.get("section") == section_hint:
                score += 5.0
            if record.get("source_file") in excluded_sources:
                score *= 0.72
            if score > 0:
                scores.append((score, index))

        scores.sort(reverse=True)
        results = []
        seen = set()
        for score, index in scores:
            record = self.records[index]
            key = (record.get("source_file"), record.get("chunk_id"))
            if key in seen:
                continue
            seen.add(key)
            result = dict(record)
            result["retrieval_score"] = round(score, 4)
            result["retrieval_tool"] = selected_tool
            results.append(result)
            if len(results) >= limit:
                break
        return results


KNOWLEDGE_INDEX = OriginalKnowledgeIndex(KNOWLEDGE_PATH)
KNOWLEDGE_MANIFEST = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
SESSION_MEMORY: dict[str, dict] = defaultdict(
    lambda: {
        "history": [],
        "preferences": [],
        "last_preference_focus": "",
        "recommended_sources": [],
    }
)
MEMORY_LOCK = threading.Lock()


def _extract_preferences(query: str, existing: list[str]) -> tuple[list[str], str]:
    lowered = query.lower()
    preferences = list(existing)
    focus = ""
    preference_trigger = _contains_any(
        lowered,
        ["i like", "i prefer", "i love", "interested in", "我喜欢", "我偏好", "我对", "适合我"],
    )
    if preference_trigger:
        groups = [
            ("food_street_culture", ["street food", "night market", "lantern", "街头美食", "夜市", "灯笼"]),
            ("japan_travel", ["japan", "tokyo", "kimono", "日本", "东京", "和服"]),
            ("new_zealand_travel", ["new zealand", "queenstown", "新西兰", "皇后镇"]),
            ("gaming", ["valorant", "gaming", "esports", "游戏", "电竞"]),
            ("visual_content", ["image", "photo", "visual", "fashion", "图片", "照片", "视觉", "时尚"]),
            ("wellbeing", ["anxiety", "stress", "mindfulness", "焦虑", "压力", "正念"]),
            ("study", ["database", "study notes", "数据库", "学习笔记"]),
        ]
        for preference, words in groups:
            if _contains_any(lowered, words):
                preferences.append(preference)
                focus = preference
    return list(dict.fromkeys(preferences)), focus


def memory_node(state: AgentState) -> AgentState:
    with MEMORY_LOCK:
        memory = SESSION_MEMORY[state["session_id"]]
        preferences, new_focus = _extract_preferences(
            state["query"], memory["preferences"]
        )
        history = list(memory["history"])
    return {
        **state,
        "preferences": preferences,
        "last_preference_focus": new_focus or memory["last_preference_focus"],
        "trace": ["读取会话记忆", f"识别用户偏好：{preferences}"],
        "_history": history,
        "_recommended_sources": list(memory["recommended_sources"]),
    }


def rewrite_query_node(state: AgentState) -> AgentState:
    query = state["query"].strip()
    history = state.get("_history", [])
    effective_query = query
    if history and len(_tokenize(query)) < 8:
        previous_query = next(
            (item["content"] for item in reversed(history) if item["role"] == "user"),
            "",
        )
        effective_query = f"{previous_query} {query} {state.get('last_preference_focus', '')}".strip()
    return {
        **state,
        "effective_query": effective_query,
        "trace": state["trace"] + [f"查询改写：{effective_query}"],
    }


def router_node(state: AgentState) -> AgentState:
    query = state["effective_query"].lower()
    visual_words = ["image", "picture", "photo", "visual", "screenshot", "map", "poster", "login page", "dashboard", "profile page", "图片", "照片", "截图", "地图", "海报", "视觉", "页面", "登录页", "仪表盘"]
    analytical_words = ["compare", "difference", "combine", "relationship", "why", "outperform", "比较", "区别", "结合", "关系", "为什么", "如何共同"]
    personalised_words = ["for me", "suit me", "recommend", "my taste", "another", "适合我", "为我", "推荐", "我的偏好", "另一个", "再推荐"]

    if _contains_any(query, personalised_words) or (
        state["preferences"] and _contains_any(query, ["recommend", "another", "推荐", "另一个", "更多"])
    ):
        family = "personalised_context"
    elif _contains_any(query, analytical_words):
        family = "analytical_multi_hop"
    elif _contains_any(query, visual_words):
        family = "cross_modal"
    else:
        family = "factual_retrieval"
    section = detect_query_section(query)
    return {
        **state,
        "query_family": family,
        "query_section": section,
        "trace": state["trace"] + [f"路由分类：{family}", f"章节提示：{section or '无'}"],
    }


def planner_node(state: AgentState) -> AgentState:
    query = state["effective_query"].lower()
    family = state["query_family"]
    known_caption_words = ["aespa", "bts", "donk", "valorant", "eventnow", "login", "dashboard", "japan", "new zealand", "study notes", "food", "map", "图片", "截图", "日本", "新西兰", "数据库笔记"]
    if family == "personalised_context":
        tool = "personalised_hybrid_retrieval"
        plan = "结合会话偏好检索文本块和图像描述"
    elif family == "analytical_multi_hop":
        tool = "hybrid_retrieval"
        plan = "组合文本证据和图像描述完成多跳回答"
    elif family == "cross_modal" and _contains_any(query, known_caption_words):
        tool = "image_caption_retrieval"
        plan = "从原始图像语义描述知识中检索"
    elif family == "cross_modal":
        tool = "true_image_retrieval"
        plan = "查询原 OpenCLIP 图像索引关联的图片"
    else:
        tool = "text_retrieval"
        plan = "从原始 PDF 切分文本中检索"
    return {
        **state,
        "selected_tool": tool,
        "retrieval_plan": plan,
        "trace": state["trace"] + [f"检索规划：{plan}", f"选择工具：{tool}"],
    }


def decomposition_node(state: AgentState) -> AgentState:
    if state["query_family"] == "analytical_multi_hop":
        sub_questions = [
            f"相关文本证据：{state['effective_query']}",
            f"相关视觉证据：{state['effective_query']}",
            "如何组合这些证据？",
        ]
    elif state["query_family"] == "personalised_context":
        sub_questions = [
            state["effective_query"],
            f"哪些证据符合偏好：{state['preferences']}",
        ]
    else:
        sub_questions = [state["effective_query"]]
    return {
        **state,
        "sub_questions": sub_questions,
        "trace": state["trace"] + [f"问题分解：{len(sub_questions)} 个子问题"],
    }


def retrieval_node(state: AgentState) -> AgentState:
    family_hint = detect_document_family(state["effective_query"])
    documents = KNOWLEDGE_INDEX.search(
        query=state["effective_query"],
        selected_tool=state["selected_tool"],
        family_hint=family_hint,
        section_hint=state.get("query_section"),
        preferences=state.get("preferences", []),
        excluded_sources=state.get("_recommended_sources", []),
        limit=4,
    )
    trace = state["trace"] + [
        f"文档族提示：{family_hint or '无'}",
        f"从原始 1813 个知识块中检索到 {len(documents)} 条证据",
    ]
    trace.extend(
        f"证据 {index + 1}：{doc['source_file']} / {doc['modality']} / chunk {doc.get('chunk_id')}"
        for index, doc in enumerate(documents)
    )
    return {**state, "retrieved_docs": documents, "trace": trace}


def _fallback_answer(documents: list[dict]) -> str:
    if not documents:
        return "原始知识库中没有检索到足够相关的内容。"
    excerpts = []
    for document in documents[:3]:
        excerpts.append(
            f"来源：{document['source_file']}\n{document['content'][:420].strip()}"
        )
    return "根据原始知识库检索到以下内容：\n\n" + "\n\n".join(excerpts)


def generation_node(state: AgentState) -> AgentState:
    documents = state.get("retrieved_docs", [])
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        answer = _fallback_answer(documents)
        provider = "原文摘录模式"
    else:
        context = "\n\n".join(
            f"[证据 {index + 1}]\n"
            f"来源文件：{doc['source_file']}\n"
            f"页码：{doc.get('page_label', '')}\n"
            f"文档族：{doc.get('document_family', '')}\n"
            f"章节：{doc.get('section', '')}\n"
            f"模态：{doc.get('modality', '')}\n"
            f"内容：{doc['content']}"
            for index, doc in enumerate(documents)
        )
        client = OpenAI(
            api_key=api_key,
            base_url=os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
            timeout=45,
        )
        response = client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "openai/gpt-oss-20b"),
            temperature=0.1,
            max_tokens=800,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是 Yixin Zhang 原始个性化多模态 RAG 项目的智能体。"
                        "你收到的证据直接来自项目原 Chroma 知识库中按 650 字符、100 字符重叠切分的知识块。"
                        "只依据证据回答，优先使用中文；涉及图片时指出 source_file，涉及文档时指出文件名和页码。"
                        "不要把系统报告或评估摘要误当成用户知识库内容，证据不足时明确说明。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"原始问题：{state['query']}\n"
                        f"有效检索问题：{state['effective_query']}\n"
                        f"检索工具：{state['selected_tool']}\n"
                        f"用户偏好：{state.get('preferences', [])}\n\n"
                        f"检索证据：\n{context}"
                    ),
                },
            ],
        )
        answer = response.choices[0].message.content or _fallback_answer(documents)
        provider = "Groq 云端模型"
    return {
        **state,
        "answer": answer.strip(),
        "trace": state["trace"] + [f"使用 {provider} 基于真实检索证据生成回答"],
    }


def update_memory_node(state: AgentState) -> AgentState:
    with MEMORY_LOCK:
        memory = SESSION_MEMORY[state["session_id"]]
        memory["history"].extend([
            {"role": "user", "content": state["query"]},
            {"role": "assistant", "content": state["answer"]},
        ])
        memory["history"] = memory["history"][-12:]
        memory["preferences"] = state.get("preferences", [])
        memory["last_preference_focus"] = state.get("last_preference_focus", "")
        for document in state.get("retrieved_docs", [])[:1]:
            source = document.get("source_file")
            if source and source not in memory["recommended_sources"]:
                memory["recommended_sources"].append(source)
        memory["recommended_sources"] = memory["recommended_sources"][-12:]
    return {**state, "trace": state["trace"] + ["更新会话记忆和已推荐来源"]}


def verification_node(state: AgentState) -> AgentState:
    modalities = {doc.get("modality") for doc in state.get("retrieved_docs", [])}
    selected_tool = state.get("selected_tool")
    if not state.get("answer", "").strip():
        result = "failed: empty answer"
    elif not state.get("retrieved_docs"):
        result = "limited_evidence"
    elif selected_tool == "text_retrieval" and "text" not in modalities:
        result = "warning: no text evidence"
    elif selected_tool in {"image_caption_retrieval", "true_image_retrieval"} and "image_caption" not in modalities:
        result = "warning: no image evidence"
    else:
        result = "passed: original knowledge evidence retrieved"
    return {
        **state,
        "verification_result": result,
        "trace": state["trace"] + [f"证据验证：{result}"],
    }


workflow = StateGraph(AgentState)
workflow.add_node("memory", memory_node)
workflow.add_node("rewrite", rewrite_query_node)
workflow.add_node("router", router_node)
workflow.add_node("planner", planner_node)
workflow.add_node("decomposer", decomposition_node)
workflow.add_node("retriever", retrieval_node)
workflow.add_node("generator", generation_node)
workflow.add_node("memory_update", update_memory_node)
workflow.add_node("verifier", verification_node)
workflow.set_entry_point("memory")
workflow.add_edge("memory", "rewrite")
workflow.add_edge("rewrite", "router")
workflow.add_edge("router", "planner")
workflow.add_edge("planner", "decomposer")
workflow.add_edge("decomposer", "retriever")
workflow.add_edge("retriever", "generator")
workflow.add_edge("generator", "memory_update")
workflow.add_edge("memory_update", "verifier")
workflow.add_edge("verifier", END)
AGENT = workflow.compile()


def run_cloud_agent(query: str, session_id: str) -> AgentState:
    return AGENT.invoke({"query": query, "session_id": session_id})
