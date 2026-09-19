import os
import re
import json
import time
import shutil
import subprocess
from typing import List, Dict, Any, TypedDict, Optional
from urllib.parse import urlparse

import requests
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from langgraph.graph import StateGraph, END

# Import DDGS safely from `ddgs` or `duckduckgo_search` without warnings
try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

# Set explicit User-Agent to avoid library warnings
os.environ.setdefault(
    "USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 MultiAgentResearcher/2.0"
)

try:
    from langchain_community.document_loaders import WebBaseLoader
except ImportError:
    WebBaseLoader = None

app = FastAPI(title="Multi-Agent Research Intelligence")
templates = Jinja2Templates(directory="templates")

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3")


def find_ollama_executable() -> Optional[str]:
    """Finds the ollama binary on Windows / Linux / macOS."""
    path_which = shutil.which("ollama")
    if path_which:
        return path_which

    # Standard Windows install locations
    local_app_data = os.getenv("LOCALAPPDATA", "")
    candidates = [
        os.path.join(local_app_data, "Programs", "Ollama", "ollama.exe"),
        os.path.expanduser(r"~\AppData\Local\Programs\Ollama\ollama.exe"),
        r"C:\Program Files\Ollama\ollama.exe",
        r"C:\Program Files (x86)\Ollama\ollama.exe"
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None


def try_auto_start_ollama() -> bool:
    """Attempts to start the Ollama background daemon if the binary exists."""
    exe = find_ollama_executable()
    if not exe:
        return False
    try:
        creation_flags = 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW
        subprocess.Popen([exe, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=creation_flags)
        for _ in range(6):
            time.sleep(0.5)
            try:
                res = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=1.0)
                if res.status_code == 200:
                    return True
            except Exception:
                pass
    except Exception as e:
        print(f"[Ollama Auto-Start Error] {e}")
    return False


def get_available_ollama_model(host: str = OLLAMA_HOST) -> Optional[str]:
    """Detects active installed models in Ollama."""
    try:
        resp = requests.get(f"{host}/api/tags", timeout=1.5)
        if resp.status_code == 200:
            models = [m.get("name") for m in resp.json().get("models", [])]
            if not models:
                return None
            for pref in ["llama3", "llama3:latest", "llama3.2", "llama3.1", "mistral", "qwen"]:
                for m in models:
                    if pref in m.lower():
                        return m
            return models[0]
    except Exception:
        pass
    return None


def check_ollama_status(host: str = OLLAMA_HOST) -> Dict[str, Any]:
    """Checks Ollama server connectivity and model availability."""
    installed = find_ollama_executable() is not None
    try:
        resp = requests.get(f"{host}/api/tags", timeout=1.5)
        if resp.status_code == 200:
            models = [m.get("name") for m in resp.json().get("models", [])]
            active_model = get_available_ollama_model(host) or DEFAULT_MODEL
            return {
                "available": True,
                "host": host,
                "installed": installed,
                "models": models,
                "active_model": active_model,
                "message": f"Connected to Ollama ({active_model})"
            }
    except Exception:
        pass

    return {
        "available": False,
        "host": host,
        "installed": installed,
        "models": [],
        "active_model": None,
        "message": "Ollama not running on localhost:11434"
    }


def call_cloud_fallback_llm(prompt: str) -> Optional[str]:
    """Calls Groq or OpenAI API if API key is present in environment."""
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        try:
            res = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"},
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.4
                },
                timeout=45
            )
            if res.status_code == 200:
                return res.json()["choices"][0]["message"]["content"]
        except Exception:
            pass

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        try:
            res = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {openai_key}", "Content-Type": "application/json"},
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.4
                },
                timeout=45
            )
            if res.status_code == 200:
                return res.json()["choices"][0]["message"]["content"]
        except Exception:
            pass

    return None


def call_llm(prompt: str, model: Optional[str] = None, host: str = OLLAMA_HOST, timeout: int = 180) -> tuple[Optional[str], str]:
    """
    Invokes LLM with auto-start attempt, local Ollama execution, and optional Cloud API fallback.
    Returns (response_text, engine_name).
    """
    # 1. Check local Ollama or try auto-start if binary exists
    status = check_ollama_status(host)
    if not status["available"] and status["installed"]:
        print("[Ollama Auto-Start] Detected Ollama binary, attempting background serve...")
        if try_auto_start_ollama():
            status = check_ollama_status(host)

    if status["available"]:
        target_model = model or get_available_ollama_model(host) or DEFAULT_MODEL
        url = f"{host}/api/generate"
        try:
            payload = {
                "model": target_model,
                "prompt": prompt,
                "stream": False
            }
            res = requests.post(url, json=payload, timeout=timeout)
            if res.status_code == 200:
                ans = res.json().get("response", "").strip()
                if ans:
                    return ans, f"Ollama ({target_model})"
        except Exception as e:
            print(f"[Ollama Call Warning] {e}")

    # 2. Check Cloud LLM if environment variable is set
    cloud_ans = call_cloud_fallback_llm(prompt)
    if cloud_ans:
        cloud_name = "Groq (llama-3.3-70b)" if os.getenv("GROQ_API_KEY") else "OpenAI (gpt-4o-mini)"
        return cloud_ans, cloud_name

    return None, "Offline (No LLM)"


# --- STATE DEFINITIONS ---
class SourceItem(TypedDict):
    title: str
    url: str
    domain: str
    snippet: str


class MetricItem(TypedDict):
    label: str
    value: str
    context: str


class AgentState(TypedDict):
    topic: str
    focus_mode: str
    search_queries: List[str]
    sources: List[SourceItem]
    research_data: str
    metrics: List[MetricItem]
    final_report: str
    engine_used: str
    agent_logs: List[Dict[str, str]]


# --- AGENT 1: PLANNER ---
def planner_node(state: AgentState) -> Dict[str, Any]:
    topic = state.get("topic", "").strip()
    focus = state.get("focus_mode", "comprehensive")
    logs = list(state.get("agent_logs", []))

    logs.append({
        "agent": "Planner",
        "action": f"Formulating search queries for '{topic}' (focus: {focus})",
        "timestamp": time.strftime("%H:%M:%S")
    })

    prompt = f"""You are an expert research planner.
Topic: {topic}
Focus: {focus}

Generate 2 to 3 concise, highly effective search queries to research this topic thoroughly.
Output ONLY a comma-separated list of the search queries, with no extra commentary or numbering."""

    llm_resp, engine = call_llm(prompt, timeout=40)
    queries = []

    if llm_resp:
        for q in llm_resp.replace("\n", ",").split(","):
            cleaned = re.sub(r'^\d+[\.\)\-]\s*', '', q).strip().strip('"').strip("'")
            if cleaned and len(cleaned) > 4 and cleaned not in queries:
                queries.append(cleaned)

    if not queries:
        queries = [
            f"{topic}",
            f"{topic} overview developments",
            f"{topic} trends analysis"
        ]

    queries = queries[:3]
    logs.append({
        "agent": "Planner",
        "action": f"Queries generated: {', '.join(queries)} (via {engine if llm_resp else 'Direct Vectoring'})",
        "timestamp": time.strftime("%H:%M:%S")
    })

    return {"search_queries": queries, "agent_logs": logs}


# --- AGENT 2: RESEARCHER ---
def researcher_node(state: AgentState) -> Dict[str, Any]:
    queries = state.get("search_queries", [])
    logs = list(state.get("agent_logs", []))
    sources: List[SourceItem] = []
    text_blocks = []

    logs.append({
        "agent": "Researcher",
        "action": f"Executing web search for {len(queries)} queries",
        "timestamp": time.strftime("%H:%M:%S")
    })

    for q in queries:
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(q, max_results=3))
                for r in results:
                    href = r.get("href", "")
                    title = r.get("title", "Untitled Source")
                    body = r.get("body", "")
                    domain = urlparse(href).netloc.replace("www.", "") if href else "web"

                    if href and not any(s["url"] == href for s in sources):
                        sources.append({
                            "title": title,
                            "url": href,
                            "domain": domain,
                            "snippet": body[:350]
                        })
                        text_blocks.append(f"Source [{title} - {domain}]:\n{body}")
        except Exception as e:
            print(f"[Search Notice] {q}: {e}")

    # Optional deep scrape for top URLs
    if WebBaseLoader and sources:
        top_urls = [s["url"] for s in sources[:2] if s["url"].startswith("http")]
        if top_urls:
            try:
                loader = WebBaseLoader(
                    web_paths=top_urls,
                    header_template={"User-Agent": os.environ["USER_AGENT"]}
                )
                docs = loader.load()
                for doc in docs:
                    clean = re.sub(r'\s+', ' ', doc.page_content).strip()
                    if len(clean) > 200:
                        text_blocks.append(f"Deep Page Extract ({doc.metadata.get('source', 'web')}):\n{clean[:2500]}")
            except Exception as e:
                print(f"[Deep Scrape skipped] {e}")

    research_data = "\n\n".join(text_blocks)
    logs.append({
        "agent": "Researcher",
        "action": f"Gathered {len(sources)} sources from the web",
        "timestamp": time.strftime("%H:%M:%S")
    })

    return {
        "sources": sources,
        "research_data": research_data[:25000],
        "agent_logs": logs
    }


# --- AGENT 3: ANALYST ---
def analyst_node(state: AgentState) -> Dict[str, Any]:
    corpus = state.get("research_data", "")
    sources = state.get("sources", [])
    logs = list(state.get("agent_logs", []))

    logs.append({
        "agent": "Analyst",
        "action": "Extracting empirical data and key indicators from gathered research",
        "timestamp": time.strftime("%H:%M:%S")
    })

    # Only extract genuine numbers/metrics present in the scraped text
    metrics: List[MetricItem] = []

    # Real percentage data in the corpus
    percent_matches = re.finditer(r'(\b\d+(?:\.\d+)?%|\b\d+\s*percent\b)(?:[^\.\n]{5,65})', corpus, flags=re.IGNORECASE)
    for match in percent_matches:
        val = match.group(1).strip()
        context = match.group(0).strip()
        if not any(m["value"] == val for m in metrics):
            metrics.append({
                "label": "Metric",
                "value": val,
                "context": context[:85]
            })
        if len(metrics) >= 3:
            break

    # Real monetary or scale data in the corpus
    if len(metrics) < 4:
        currency_matches = re.finditer(r'(\$\s*\d+(?:\.\d+)?\s*(?:billion|trillion|million|B|M|T)?\b)(?:[^\.\n]{5,65})', corpus, flags=re.IGNORECASE)
        for match in currency_matches:
            val = match.group(1).strip()
            context = match.group(0).strip()
            if not any(m["value"] == val for m in metrics):
                metrics.append({
                    "label": "Financial / Volume",
                    "value": val,
                    "context": context[:85]
                })
            if len(metrics) >= 4:
                break

    logs.append({
        "agent": "Analyst",
        "action": f"Found {len(metrics)} quantitative data points in scraped sources",
        "timestamp": time.strftime("%H:%M:%S")
    })

    return {
        "metrics": metrics,
        "agent_logs": logs
    }


# --- AGENT 4: WRITER ---
def writer_node(state: AgentState) -> Dict[str, Any]:
    topic = state.get("topic", "")
    focus = state.get("focus_mode", "comprehensive")
    corpus = state.get("research_data", "")
    sources = state.get("sources", [])
    logs = list(state.get("agent_logs", []))

    logs.append({
        "agent": "Writer",
        "action": "Sending research data to LLM to synthesize final report",
        "timestamp": time.strftime("%H:%M:%S")
    })

    # Let the LLM determine its own structure, headings, and formatting
    prompt = f"""You are an expert researcher and technical writer.
Review the research data gathered from the live web below and write a comprehensive, well-structured professional report exploring the topic.

Topic: {topic}
Focus: {focus}

Research Data Gathered from the Web:
{corpus[:15000]}

Please write the complete report in Markdown. You have full freedom to choose the structure, headings, depth, and presentation format that best represents the findings."""

    llm_report, engine_used = call_llm(prompt, timeout=180)

    if llm_report:
        logs.append({
            "agent": "Writer",
            "action": f"Report successfully generated by {engine_used} with custom structure",
            "timestamp": time.strftime("%H:%M:%S")
        })
    else:
        # If Ollama is offline, present the real web research cleanly without any fake hardcoded essay
        engine_used = "Ollama Offline (Live Research Mode)"
        llm_report = build_offline_notice(topic, sources, corpus)
        logs.append({
            "agent": "Writer",
            "action": "Ollama connection offline. Presenting verified sources and extracted data.",
            "timestamp": time.strftime("%H:%M:%S")
        })

    return {
        "final_report": llm_report,
        "engine_used": engine_used,
        "agent_logs": logs
    }


def build_offline_notice(topic: str, sources: List[SourceItem], corpus: str) -> str:
    """
    Transparently displays the real web research collected when Ollama is offline,
    with clear guidance on how to enable LLM synthesis.
    """
    sources_text = "\n".join([
        f"- **[{s['domain']}]** [{s['title']}]({s['url']})\n  > *{s['snippet']}*"
        for s in sources
    ]) if sources else "No external web sources were retrieved."

    raw_snippets = corpus[:3500] if corpus else "No content scraped."

    return f"""# Research Overview: {topic}

> ### ⚠️ Local LLM Offline (Ollama Not Detected at `localhost:11434`)
>
> The multi-agent pipeline gathered live information from the web below.
> To have the LLM write its own synthesized report, choose one of these quick steps:
>
> **Option 1 (Recommended): Run Local Ollama**
> Open any command prompt or PowerShell and run:
> ```bash
> ollama run llama3
> ```
> *(If not installed, download free from [ollama.com](https://ollama.com))*
>
> **Option 2: Cloud API (Fast & Free Tier)**
> Set an environment variable in your terminal before launching:
> ```powershell
> $env:GROQ_API_KEY="your-groq-key"   # Ultra-fast free Llama 3.3
> ```
> Once either option is active, re-submit your topic to get a custom LLM-generated report.

---

## Live Sources Retrieved by Researcher Agent

{sources_text}

---

## Live Web Data Gathered

{raw_snippets}
"""


# --- BUILD WORKFLOW ---
workflow = StateGraph(AgentState)

workflow.add_node("planner", planner_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("analyst", analyst_node)
workflow.add_node("writer", writer_node)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "researcher")
workflow.add_edge("researcher", "analyst")
workflow.add_edge("analyst", "writer")
workflow.add_edge("writer", END)

app_graph = workflow.compile()


# --- HTTP ROUTES ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request, "index.html", {})


@app.get("/api/status")
async def get_system_status():
    status = check_ollama_status()
    has_cloud = bool(os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY"))
    return JSONResponse(content={
        "ollama": status,
        "cloud_available": has_cloud,
        "pipeline": {
            "agents": [
                {"id": "planner", "name": "Planner", "role": "Query Formulation"},
                {"id": "researcher", "name": "Researcher", "role": "Live Web Reconnaissance"},
                {"id": "analyst", "name": "Analyst", "role": "Empirical Signal Extraction"},
                {"id": "writer", "name": "Writer", "role": "Dynamic LLM Synthesis"}
            ],
            "framework": "LangGraph StateGraph"
        }
    })


@app.post("/api/start-ollama")
async def start_ollama_endpoint():
    """Attempts to auto-launch Ollama background daemon on demand."""
    success = try_auto_start_ollama()
    status = check_ollama_status()
    return JSONResponse(content={"started": success, "status": status})


@app.post("/research")
async def research_endpoint(request: Request):
    try:
        data = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Malformed JSON payload"})

    topic = data.get("topic", "").strip()
    focus_mode = data.get("focus_mode", "comprehensive").strip().lower()

    if not topic:
        return JSONResponse(status_code=400, content={"error": "Topic is required to initiate research."})

    initial_state: AgentState = {
        "topic": topic,
        "focus_mode": focus_mode,
        "search_queries": [],
        "sources": [],
        "research_data": "",
        "metrics": [],
        "final_report": "",
        "engine_used": "",
        "agent_logs": []
    }

    try:
        final_state = app_graph.invoke(initial_state)

        return JSONResponse(content={
            "report": final_state.get("final_report", ""),
            "queries": final_state.get("search_queries", []),
            "sources": final_state.get("sources", []),
            "metrics": final_state.get("metrics", []),
            "engine_used": final_state.get("engine_used", "Local Pipeline"),
            "logs": final_state.get("agent_logs", [])
        })
    except Exception as e:
        print(f"[Research Execution Error] {e}")
        return JSONResponse(content={
            "report": f"### Pipeline Notice\n\nAn unexpected runtime exception occurred: `{str(e)}`.\n\nPlease check server logs and ensure Ollama is running (`ollama run llama3`).",
            "queries": [],
            "sources": [],
            "metrics": [],
            "engine_used": "Error Recovery",
            "logs": [{"agent": "System", "action": f"Error: {str(e)}", "timestamp": time.strftime("%H:%M:%S")}]
        })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8003, reload=True)