# 个性化多模态智能体公开展示页

这是 AI 聊天系统的 GitHub Pages 静态展示版本。页面会展示 evaluation 图表、检索摘要、LLM-as-Judge 指标、智能体工作流和聊天界面。

聊天界面默认调用：

```text
http://127.0.0.1:5000/api/chat
```

如果本地 API 没有启动，页面会自动切换为前端演示回复。真实 LangGraph/Ollama 智能体需要运行项目根目录下的：

```bash
python app.py
```

## GitHub Pages

仓库启用 Pages 后，GitHub Actions 会使用 `.github/workflows/deploy-pages.yml` 部署本静态站点。
