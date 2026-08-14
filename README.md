# 个性化多模态智能体聊天系统

[在线体验](https://hedyyaokuo.github.io/ai-chatbot-system/) · [公网 API 健康检查](https://yixin-ai-chatbot-api.onrender.com/api/health)

这是原始课程项目的公网部署版。GitHub Pages 托管评价展示与聊天界面，Render 托管 Flask/LangGraph API，Groq 提供云端大模型推理。后端检索内容直接导出自原项目的 Chroma 数据库，不再使用报告摘要组成的展示知识库。

## 在线架构

```text
浏览器
  -> GitHub Pages 前端
  -> Render Flask API
  -> LangGraph：记忆 -> 查询改写 -> 路由 -> 规划 -> 分解 -> 检索 -> 生成 -> 记忆更新 -> 验证
  -> 原始知识库：1813 个已切分知识块
  -> Groq 云端模型
```

## 主要功能

- 展示多模态检索 Top-3 Hit Rate 与 LLM-as-Judge 评估结果。
- 使用原项目 `chunk_size=650`、`chunk_overlap=100` 生成的知识块。
- 包含 1786 个 PDF 文本块、27 个图像语义描述、39 个源文件。
- 保留文档族识别、章节提示、文本检索、图像描述检索、混合检索与个性化混合检索。
- 支持每位访客独立的短期会话记忆与偏好驱动追问。
- 回答会返回真实源文件、页码和原始 `chunk_id`。
- 对请求长度和访问频率进行基础限制，避免公开 API 被轻易滥用。
- GitHub `main` 分支更新后自动部署前端和后端。

## 仓库结构

```text
.
├── index.html
├── assets/
├── data/
├── backend/
│   ├── app.py
│   ├── agent.py
│   ├── original_knowledge.json.gz
│   ├── knowledge_manifest.json
│   ├── export_original_knowledge.py
│   └── requirements.txt
├── render.yaml
└── .github/workflows/deploy-pages.yml
```

## 本地运行云端版后端

```bash
cd backend
pip install -r requirements.txt
set GROQ_API_KEY=你的密钥
python app.py
```

API 端点：

```text
GET  /api/health
POST /api/chat
```

请求示例：

```json
{
  "message": "Hybrid Retrieval 为什么比 text-only 更好？",
  "session_id": "example_session_001"
}
```

## 与原始项目的关系

`backend/original_knowledge.json.gz` 由原项目的以下数据库直接导出：

```text
vector_db/chroma.sqlite3
  collection: personalised_multimodal_knowledge_base

image_vector_db/chroma.sqlite3
  collection: eventnow_true_image_index
```

云端免费实例无法稳定运行本机 Ollama 和 OpenCLIP，因此生成模型替换为 Groq，查询排序使用 BM25 与原始元数据规则。知识内容、切分边界、来源元数据、图像描述及 LangGraph 的智能体步骤均来自原项目。

## 自动部署

- GitHub Pages 使用 `.github/workflows/deploy-pages.yml`，只发布前端文件。
- Render 使用 `render.yaml` 创建免费 Web Service，并跟随 GitHub 提交自动部署。
- `GROQ_API_KEY` 只保存在 Render Secret 环境变量中，不得提交到仓库。

Render 免费实例闲置 15 分钟后会休眠，首次请求唤醒通常需要约一分钟。该限制适合课程展示与作品集，不建议直接用于生产业务。
