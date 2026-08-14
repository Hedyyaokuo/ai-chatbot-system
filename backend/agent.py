from __future__ import annotations

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
KNOWLEDGE_PATH = BASE_DIR / "knowledge.json"


class AgentState(TypedDict, total=False):
    session_id: str
    query: str
    rewritten_query: str
    query_family: str
    retrieved_docs: list[dict]
    answer: str
    verification_result: str
    trace: list[str]


def _tokenize(text: str) -> list[str]:
    lowered = text.lower()
    latin = re.findall(r"[a-z0-9][a-z0-9_+-]*", lowered)
    chinese_runs = re.findall(r"[\u4e00-\u9fff]+", lowered)
    chinese = []
    for run in chinese_runs:
        chinese.extend(run[index:index + 2] for index in range(max(1, len(run) - 1)))
    return latin + chinese


class KnowledgeIndex:
    def __init__(self, path: Path):
        records = json.loads(path.read_text(encoding="utf-8"))
        self.records = records
        self.term_frequencies: list[Counter] = []
        self.document_frequencies: Counter = Counter()
        self.lengths: list[int] = []

        for record in records:
            tokens = _tokenize(
                f"{record.get('title', '')} {record.get('family', '')} "
                f"{record.get('section', '')} {record.get('content', '')}"
            )
            frequencies = Counter(tokens)
            self.term_frequencies.append(frequencies)
            self.lengths.append(len(tokens))
            self.document_frequencies.update(frequencies.keys())

        self.average_length = sum(self.lengths) / max(1, len(self.lengths))

    def search(self, query: str, family: str, limit: int = 5) -> list[dict]:
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scores = []
        document_count = len(self.records)
        for index, frequencies in enumerate(self.term_frequencies):
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
                    1 - 0.75 + 0.75 * length / max(1, self.average_length)
                )
                score += inverse_frequency * frequency * 2.2 / denominator

            if family != "general" and self.records[index].get("family") == family:
                score *= 1.25
            if score > 0:
                scores.append((score, index))

        scores.sort(reverse=True)
        return [self.records[index] for _, index in scores[:limit]]


KNOWLEDGE_INDEX = KnowledgeIndex(KNOWLEDGE_PATH)
SESSION_MEMORY: dict[str, list[dict[str, str]]] = defaultdict(list)
MEMORY_LOCK = threading.Lock()


def _route_family(query: str) -> str:
    lowered = query.lower()
    routes = {
        "evaluation": ["evaluation", "评估", "评分", "命中率", "hit rate", "judge", "实验"],
        "retrieval": ["retrieval", "检索", "rag", "hybrid", "text-only", "多模态", "向量"],
        "memory": ["memory", "记忆", "个性化", "追问", "上下文"],
        "eventnow": ["eventnow", "活动", "组织者", "参与者", "session", "报名"],
        "architecture": ["langgraph", "架构", "流程", "智能体", "agent", "工具", "路由"],
    }
    for family, keywords in routes.items():
        if any(keyword in lowered for keyword in keywords):
            return family
    return "general"


def _prepare_query(state: AgentState) -> AgentState:
    query = state["query"].strip()
    with MEMORY_LOCK:
        history = SESSION_MEMORY[state["session_id"]][-4:]
    previous_user_messages = [item["content"] for item in history if item["role"] == "user"]
    rewritten = query
    if previous_user_messages and len(_tokenize(query)) < 8:
        rewritten = f"前文主题：{previous_user_messages[-1]}\n当前追问：{query}"
    return {
        **state,
        "rewritten_query": rewritten,
        "trace": ["读取会话记忆", "改写上下文查询"],
    }


def _route_query(state: AgentState) -> AgentState:
    family = _route_family(state["rewritten_query"])
    return {
        **state,
        "query_family": family,
        "trace": state["trace"] + [f"路由到 {family} 知识域"],
    }


def _retrieve(state: AgentState) -> AgentState:
    documents = KNOWLEDGE_INDEX.search(
        state["rewritten_query"], state["query_family"], limit=5
    )
    return {
        **state,
        "retrieved_docs": documents,
        "trace": state["trace"] + [f"检索到 {len(documents)} 条证据"],
    }


def _fallback_answer(state: AgentState) -> str:
    documents = state.get("retrieved_docs", [])
    if not documents:
        return "知识库中暂时没有找到与这个问题直接相关的内容。请尝试询问项目架构、评估结果、Hybrid RAG、对话记忆或 EventNow。"
    excerpts = "\n\n".join(document["content"][:420] for document in documents[:3])
    return f"根据项目知识库，相关信息如下：\n\n{excerpts}"


def _generate(state: AgentState) -> AgentState:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        answer = _fallback_answer(state)
        provider = "extractive-fallback"
    else:
        context = "\n\n".join(
            f"[来源 {index + 1}] {document['title']} / {document.get('section', '')}\n"
            f"{document['content']}"
            for index, document in enumerate(state.get("retrieved_docs", []))
        )
        client = OpenAI(
            api_key=api_key,
            base_url=os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
            timeout=45,
        )
        response = client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "openai/gpt-oss-20b"),
            temperature=0.2,
            max_tokens=700,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是 Yixin Zhang 的个性化多模态智能体项目助手。"
                        "只根据给定证据回答，优先使用中文，表达清楚而简洁。"
                        "证据不足时明确说明，不要编造评估数字、功能或来源。"
                    ),
                },
                {
                    "role": "user",
                    "content": f"问题：{state['query']}\n\n项目证据：\n{context}",
                },
            ],
        )
        answer = response.choices[0].message.content or _fallback_answer(state)
        provider = "groq"

    return {
        **state,
        "answer": answer.strip(),
        "trace": state["trace"] + [f"使用 {provider} 生成回答"],
    }


def _verify(state: AgentState) -> AgentState:
    has_answer = bool(state.get("answer", "").strip())
    has_sources = bool(state.get("retrieved_docs"))
    result = "passed" if has_answer and has_sources else "limited_evidence"
    return {
        **state,
        "verification_result": result,
        "trace": state["trace"] + [f"证据验证：{result}"],
    }


workflow = StateGraph(AgentState)
workflow.add_node("prepare", _prepare_query)
workflow.add_node("route", _route_query)
workflow.add_node("retrieve", _retrieve)
workflow.add_node("generate", _generate)
workflow.add_node("verify", _verify)
workflow.set_entry_point("prepare")
workflow.add_edge("prepare", "route")
workflow.add_edge("route", "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", "verify")
workflow.add_edge("verify", END)
AGENT = workflow.compile()


def run_cloud_agent(query: str, session_id: str) -> AgentState:
    result = AGENT.invoke({"query": query, "session_id": session_id})
    with MEMORY_LOCK:
        memory = SESSION_MEMORY[session_id]
        memory.extend([
            {"role": "user", "content": query},
            {"role": "assistant", "content": result["answer"]},
        ])
        SESSION_MEMORY[session_id] = memory[-12:]
    return result

