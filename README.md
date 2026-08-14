# 个性化多模态智能体聊天系统

[在线体验](https://hedyyaokuo.github.io/ai-chatbot-system/) · [公网 API 健康检查](https://yixin-ai-chatbot-api.onrender.com/api/health)

这是原始个性化多模态智能体项目的公网部署版本。项目目标不是将普通大模型包装成聊天页面，而是让原项目中的知识库、文档切分方式、图像描述、对话记忆和 LangGraph 智能体流程能够在公网环境中继续运行。

系统直接读取从原 Chroma 数据库导出的知识块。当前知识库包含 1813 个记录，其中包括 1786 个 PDF 文本块和 27 个图像语义描述。在线 chatbot 可以根据问题选择文本检索、图像描述检索、混合检索或个性化混合检索，并在回答中返回真实源文件、页码、`chunk_id` 和相关图片。

## 核心亮点

- 使用原项目 `chunk_size=650`、`chunk_overlap=100` 产生的知识块，不使用报告摘要代替真实知识库。
- 保留文档族识别、章节提示、查询改写、路由分类、检索规划、问题分解、答案生成和验证流程。
- 支持 EventNow 项目材料、UI 截图、旅行文化资料、K-pop 与电竞图片、数据库学习笔记和焦虑管理资料。
- 支持每位访问者独立的短期会话记忆，并使用用户偏好改写后续问题。
- 使用 Groq 综合多条检索证据，以自然语言重新组织回答，而不是直接拼接知识块。
- 图像检索结果会直接显示对应图片，而不只是返回图片文件名。
- 使用 Marked 解析 Markdown，并通过 DOMPurify 清理后再写入聊天界面。
- 提供多模态检索 Top-3 Hit Rate 与 LLM-as-Judge 评估结果展示。
- GitHub `main` 分支更新后，GitHub Pages 和 Render 会自动部署最新版本。

## 技术栈

- Python 3.12
- Flask
- LangGraph
- Groq OpenAI-compatible API
- BM25 与元数据感知检索
- GitHub Pages
- Render
- HTML、CSS、JavaScript

原始本地实验版本使用 Ollama、ChromaDB、LangChain 和 OpenCLIP。由于 Render 免费实例不适合加载本地 Ollama 与 OpenCLIP 模型，公网版本将回答生成替换为 Groq，并使用 BM25 与原始元数据规则完成云端检索。知识内容、切分边界、图片描述和智能体步骤仍然来自原项目。

## 在线架构

```text
浏览器
  -> GitHub Pages 前端
  -> Render Flask API
  -> LangGraph
     -> 记忆读取
     -> 查询改写
     -> 路由分类
     -> 检索规划
     -> 问题分解
     -> 原始知识块检索
     -> Groq 答案生成
     -> 记忆更新
     -> 证据验证
```

## 项目结构

```text
ai-chatbot-system/
├── index.html
├── assets/
│   ├── app.js
│   ├── styles.css
│   ├── image-results.css
│   ├── knowledge-images/
│   └── evaluation images
├── data/
│   └── evaluation csv files
├── backend/
│   ├── app.py
│   ├── agent.py
│   ├── original_knowledge.json.gz
│   ├── knowledge_manifest.json
│   ├── export_original_knowledge.py
│   ├── requirements.txt
│   └── test_app.py
├── render.yaml
└── .github/workflows/deploy-pages.yml
```

## 主要模块

- `backend/agent.py`：云端智能体主入口，负责会话记忆、查询改写、路由、规划、知识检索、答案生成和验证。
- `backend/app.py`：提供 Flask API、CORS 配置、访问频率限制、来源序列化和图片公开地址。
- `backend/original_knowledge.json.gz`：从原始 Chroma 数据库导出的 1813 个知识块。
- `backend/export_original_knowledge.py`：重新导出原始文本块、图像描述和元数据的脚本。
- `assets/app.js`：连接公网或本地 API，并展示聊天回答、图片和检索来源。
- `assets/vendor/`：固定版本的 Marked 与 DOMPurify，用于安全显示模型生成的 Markdown。
- `assets/knowledge-images/`：原个性化知识库中的 27 张图像资源。
- `backend/test_app.py`：验证 API、CORS、文本检索、图片检索、中文查询和会话记忆。

## 本地安装

### 1. 下载项目

```bash
git clone https://github.com/Hedyyaokuo/ai-chatbot-system.git
cd ai-chatbot-system
```

### 2. 创建虚拟环境

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Windows Command Prompt：

```bat
python -m venv .venv
.venv\Scripts\activate.bat
```

macOS 或 Linux：

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. 安装后端依赖

```bash
python -m pip install --upgrade pip
pip install -r backend/requirements.txt
```

### 4. 配置 Groq API Key

先在 [Groq Console](https://console.groq.com/keys) 创建 API Key。密钥只应保存在本地环境变量或 Render Secret 中，不要写入代码、README 或 Git 提交。

Windows PowerShell：

```powershell
$env:GROQ_API_KEY="你的 Groq API Key"
$env:FRONTEND_BASE_URL="http://127.0.0.1:8000"
```

Windows Command Prompt：

```bat
set GROQ_API_KEY=你的 Groq API Key
set FRONTEND_BASE_URL=http://127.0.0.1:8000
```

macOS 或 Linux：

```bash
export GROQ_API_KEY="你的 Groq API Key"
export FRONTEND_BASE_URL="http://127.0.0.1:8000"
```

`FRONTEND_BASE_URL` 用于生成图片公开地址。本地前端运行在 `8000` 端口时，应设置为 `http://127.0.0.1:8000`。如果不设置，后端默认返回 GitHub Pages 上的图片地址。

### 5. 启动后端 API

```bash
cd backend
python app.py
```

后端默认运行在：

[http://127.0.0.1:5000](http://127.0.0.1:5000)

健康检查：

[http://127.0.0.1:5000/api/health](http://127.0.0.1:5000/api/health)

健康接口应返回知识库统计，例如：

```json
{
  "ok": true,
  "service": "yixin-original-personalised-multimodal-agent",
  "knowledge_base": {
    "records": 1813,
    "text_chunks": 1786,
    "image_captions": 27,
    "source_files": 39,
    "chunk_size": 650,
    "chunk_overlap": 100
  }
}
```

### 6. 启动前端页面

保持后端终端继续运行，再打开一个新终端并回到项目根目录：

```bash
python -m http.server 8000
```

然后打开：

[http://127.0.0.1:8000](http://127.0.0.1:8000)

当前端运行在 `127.0.0.1` 或 `localhost` 时，`assets/app.js` 会自动连接：

```text
http://127.0.0.1:5000/api/chat
```

## API 使用方式

API 端点：

```text
GET  /api/health
POST /api/chat
```

请求示例：

```json
{
  "message": "请找到并返回最符合日本夜市和灯笼主题的图片",
  "session_id": "example_session_001"
}
```

图像检索成功时，`sources` 中会包含：

```json
{
  "source_file": "Japan culture.jpg",
  "modality": "image_caption",
  "image_url": "https://hedyyaokuo.github.io/ai-chatbot-system/assets/knowledge-images/Japan%20culture.jpg"
}
```

## 重新导出原始知识库

只有在本地仍然保留原项目的 `vector_db` 和 `image_vector_db` 时，才需要执行这一步。普通使用者可以直接使用仓库中已经导出的 `original_knowledge.json.gz`。

Windows 示例：

```powershell
python backend\export_original_knowledge.py `
  "C:\path\to\S5004312_YixinZhang" `
  backend\original_knowledge.json.gz `
  --manifest backend\knowledge_manifest.json
```

macOS 或 Linux 示例：

```bash
python backend/export_original_knowledge.py \
  /path/to/S5004312_YixinZhang \
  backend/original_knowledge.json.gz \
  --manifest backend/knowledge_manifest.json
```

导出脚本会读取：

```text
vector_db/chroma.sqlite3
  collection: personalised_multimodal_knowledge_base

image_vector_db/chroma.sqlite3
  collection: eventnow_true_image_index
```

## 运行测试

在项目根目录运行：

```bash
python -m unittest discover -s backend -p "test_*.py" -v
python -m py_compile backend/app.py backend/agent.py backend/export_original_knowledge.py
```

测试覆盖以下内容：

- API 健康状态与知识库统计。
- GitHub Pages 的 CORS 请求。
- EventNow 文本知识检索。
- 日本夜市图片检索与图片 URL。
- 焦虑管理资料的中文查询扩展。
- 独立会话记忆与个性化追问。
- 无效请求和会话编号检查。

## 自动部署

- GitHub Pages 使用 `.github/workflows/deploy-pages.yml` 发布 `index.html`、`assets/` 和 `data/`。
- Render 使用 `render.yaml` 部署 `backend/`，并跟随 `main` 分支自动更新。
- `GROQ_API_KEY` 保存在 Render Secret 环境变量中。
- `FRONTEND_BASE_URL` 默认指向当前 GitHub Pages 地址。

Render 免费实例在闲置后会休眠，因此第一次聊天可能需要等待约一分钟。服务唤醒后，后续请求通常会明显更快。

## 局限性

- 公网版本使用 BM25 与元数据加权，不等同于原本由 `nomic-embed-text` 生成的向量相似度。
- 在线图片检索依赖原项目编写的图片描述和 OpenCLIP 索引关联结果，不会重新分析用户临时上传的图片。
- 会话记忆保存在 Render 进程内存中，服务重启或休眠后会被清空。
- 个性化知识库中的部分主题存在语义重叠，仍可能产生检索漂移。
- 当前系统适合作品集与课程项目展示，不等同于生产级智能体服务。

## AI 使用说明

开发过程中使用了生成式 AI 工具辅助理解 RAG、LangGraph、多模态检索、云端部署和前端展示，也用于帮助检查代码、调试接口和整理说明文档。知识库内容、切分结果和评估结果来自原项目文件，未使用 AI 编造实验结果。所有实现、整合、测试和最终解释责任均由项目作者承担。
