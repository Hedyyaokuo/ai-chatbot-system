# 个性化多模态智能体聊天系统

[在线体验](https://hedyyaokuo.github.io/ai-chatbot-system/) · [公网 API 健康检查](https://yixin-ai-chatbot-api.onrender.com/api/health)

这是一个由完整前端包装的个性化知识库聊天机器人。GitHub Pages 托管评价展示与聊天界面，Render 托管 Flask/LangGraph API，Groq 提供云端大模型推理。

## 在线架构

```text
浏览器
  -> GitHub Pages 前端
  -> Render Flask API
  -> LangGraph：记忆 -> 路由 -> 检索 -> 生成 -> 验证
  -> Groq 云端模型
```

## 主要功能

- 展示多模态检索 Top-3 Hit Rate 与 LLM-as-Judge 评估结果。
- 支持每位访客独立的短期会话记忆。
- 根据问题自动路由到 evaluation、retrieval、memory、EventNow 或 architecture 知识域。
- 从项目报告、README 与评估结果中检索证据，并返回来源。
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
│   ├── knowledge.json
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

## 自动部署

- GitHub Pages 使用 `.github/workflows/deploy-pages.yml`，只发布前端文件。
- Render 使用 `render.yaml` 创建免费 Web Service，并跟随 GitHub 提交自动部署。
- `GROQ_API_KEY` 只保存在 Render Secret 环境变量中，不得提交到仓库。

Render 免费实例闲置 15 分钟后会休眠，首次请求唤醒通常需要约一分钟。该限制适合课程展示与作品集，不建议直接用于生产业务。
