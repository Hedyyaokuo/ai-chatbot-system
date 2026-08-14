const retrievalTable = document.getElementById("retrievalTable");
const bestRetrieval = document.getElementById("bestRetrieval");
const bestMethod = document.getElementById("bestMethod");
const judgeAverage = document.getElementById("judgeAverage");
const messages = document.getElementById("messages");
const chatForm = document.getElementById("chatForm");
const chatInput = document.getElementById("chatInput");
const apiStatus = document.getElementById("apiStatus");
const apiEndpoint = document.getElementById("apiEndpoint");

const LOCAL_HOSTS = new Set(["127.0.0.1", "localhost"]);
const DEFAULT_CHAT_API_URL = LOCAL_HOSTS.has(globalThis.location.hostname)
  ? "http://127.0.0.1:5000/api/chat"
  : "https://yixin-ai-chatbot-api.onrender.com/api/chat";
const CHAT_API_URL = localStorage.getItem("CHAT_API_URL") || DEFAULT_CHAT_API_URL;
const HEALTH_API_URL = CHAT_API_URL.replace(/\/api\/chat\/?$/, "/api/health");
const SESSION_ID = getSessionId();

function getSessionId() {
  const saved = localStorage.getItem("CHAT_SESSION_ID");
  if (saved) return saved;
  const generated = globalThis.crypto?.randomUUID?.().replaceAll("-", "")
    || `session_${Date.now()}_${Math.random().toString(36).slice(2)}`;
  localStorage.setItem("CHAT_SESSION_ID", generated);
  return generated;
}

function parseCsv(text) {
  const rows = [];
  let cell = "";
  let row = [];
  let quoted = false;
  for (let index = 0; index < text.length; index += 1) {
    const current = text[index];
    const next = text[index + 1];
    if (current === '"' && quoted && next === '"') {
      cell += '"';
      index += 1;
    } else if (current === '"') {
      quoted = !quoted;
    } else if (current === "," && !quoted) {
      row.push(cell);
      cell = "";
    } else if ((current === "\n" || current === "\r") && !quoted) {
      if (current === "\r" && next === "\n") index += 1;
      row.push(cell);
      if (row.some((value) => value.trim() !== "")) rows.push(row);
      row = [];
      cell = "";
    } else {
      cell += current;
    }
  }
  if (cell || row.length) {
    row.push(cell);
    rows.push(row);
  }
  const [headers, ...dataRows] = rows;
  return dataRows.map((dataRow) => {
    const item = {};
    headers.forEach((header, index) => {
      item[header] = dataRow[index] || "";
    });
    return item;
  });
}

async function loadRetrievalSummary() {
  const response = await fetch("data/multimodal_retrieval_summary.csv");
  const rows = parseCsv(await response.text());
  let best = rows[0];
  retrievalTable.innerHTML = rows.map((row) => {
    const hitRate = Number(row.top3_hit_rate);
    const latency = Number(row.average_latency);
    if (hitRate > Number(best.top3_hit_rate)) best = row;
    return `<tr><td>${row.method}</td><td>${(hitRate * 100).toFixed(1)}%</td><td>${latency.toFixed(2)}s</td></tr>`;
  }).join("");
  bestRetrieval.textContent = `${(Number(best.top3_hit_rate) * 100).toFixed(0)}%`;
  bestMethod.textContent = best.method.replace(" Retrieval", "");
}

async function loadJudgeScores() {
  const response = await fetch("data/llm_judge_scores.csv");
  const rows = parseCsv(await response.text());
  const scores = rows
    .map((row) => Number(row.average_quality_score))
    .filter(Number.isFinite);
  judgeAverage.textContent = (
    scores.reduce((sum, value) => sum + value, 0) / scores.length
  ).toFixed(2);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatSources(sources = []) {
  if (!sources.length) return "";
  const items = sources.slice(0, 5).map((source) => {
    const location = source.page_label ? `第 ${source.page_label} 页` : source.modality || "text";
    return `<li>${escapeHtml(source.source_file || "原始知识库")}`
      + ` <span>${escapeHtml(location)} · chunk ${escapeHtml(source.chunk_id ?? "-")}</span></li>`;
  }).join("");
  return `<details class="sources"><summary>查看检索来源</summary><ul>${items}</ul></details>`;
}

function formatImages(sources = []) {
  const imageSources = sources.filter((source) => source.image_url).slice(0, 4);
  if (!imageSources.length) return "";
  const figures = imageSources.map((source) => (
    `<figure><a href="${escapeHtml(source.image_url)}" target="_blank" rel="noopener noreferrer">`
    + `<img src="${escapeHtml(source.image_url)}" alt="${escapeHtml(source.source_file || "检索图片")}" loading="lazy" />`
    + `</a><figcaption>${escapeHtml(source.source_file || "检索图片")}</figcaption></figure>`
  )).join("");
  return `<div class="message-images">${figures}</div>`;
}

function addMessage(role, text, options = {}) {
  const element = document.createElement("div");
  element.className = `message ${role}`;
  const label = role === "user" ? "你" : options.pending ? "智能体正在思考" : "云端智能体";
  element.innerHTML = `<small>${label}</small><div>${escapeHtml(text).replaceAll("\n", "<br>")}</div>${formatImages(options.sources)}${formatSources(options.sources)}`;
  messages.appendChild(element);
  messages.scrollTop = messages.scrollHeight;
  return element;
}

async function fetchWithTimeout(url, options, timeout = 90000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

async function askAgent(prompt) {
  const response = await fetchWithTimeout(CHAT_API_URL, {
    method: "POST",
    // text/plain is CORS-safelisted, so browsers can send the request
    // without a separate OPTIONS preflight on sleeping free instances.
    headers: { "Content-Type": "text/plain;charset=UTF-8" },
    body: JSON.stringify({ message: prompt, session_id: SESSION_ID }),
  });
  const data = await response.json();
  if (!response.ok || !data.ok) {
    throw new Error(data.error || "公网聊天 API 调用失败");
  }
  return data;
}

async function checkApiStatus() {
  apiEndpoint.textContent = new URL(CHAT_API_URL).host;
  apiStatus.textContent = "正在连接";
  try {
    const response = await fetchWithTimeout(HEALTH_API_URL, {}, 70000);
    const data = await response.json();
    if (!response.ok || !data.ok) throw new Error("服务未就绪");
    apiStatus.textContent = data.mode === "groq" ? "在线 · 云端模型" : "在线 · 检索模式";
  } catch (error) {
    apiStatus.textContent = "正在唤醒服务";
    console.warn(error);
  }
}

async function handlePrompt(prompt) {
  addMessage("user", prompt);
  chatInput.disabled = true;
  const pending = addMessage(
    "bot",
    "正在执行记忆读取、查询路由、知识检索和答案验证，请稍等……",
    { pending: true },
  );
  try {
    const data = await askAgent(prompt);
    pending.remove();
    addMessage("bot", data.answer || "智能体没有返回答案。", {
      sources: data.sources || [],
    });
    apiStatus.textContent = "在线 · 云端模型";
    console.info("Agent trace", data.trace);
  } catch (error) {
    pending.remove();
    addMessage(
      "bot",
      `暂时无法连接云端智能体：${error.message}\n\n免费服务在闲置后首次唤醒可能需要约一分钟，请稍后重新发送。`,
    );
    apiStatus.textContent = "连接异常";
  } finally {
    chatInput.disabled = false;
    chatInput.focus();
  }
}

chatForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const value = chatInput.value.trim();
  if (!value) return;
  chatInput.value = "";
  handlePrompt(value);
});

document.querySelectorAll("[data-prompt]").forEach((button) => {
  button.addEventListener("click", () => handlePrompt(button.dataset.prompt));
});

addMessage("bot", "你好，我已连接到项目原始知识库。你可以询问 EventNow、旅行地图与文化、K-pop 和电竞图片、数据库笔记、焦虑管理资料，也可以告诉我你的偏好后继续追问。", {});
checkApiStatus();
loadRetrievalSummary().catch(() => {
  retrievalTable.innerHTML = '<tr><td colspan="3">评估摘要加载失败。</td></tr>';
});
loadJudgeScores().catch(() => {
  judgeAverage.textContent = "--";
});
