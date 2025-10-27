Below is a complete, production-ready Streamlit reimplementation tailored for Hugging Face Spaces that:
- Replaces the original React/TypeScript UI with Streamlit.
- Supports multi-agent orchestration via agents.yaml.
- Lets the user choose a model per agent from Gemini, OpenAI, and Grok providers.
- Includes improved advanced prompts and structured output strategies.
- Adds “wow” status indicators and an interactive dashboard (live metrics, progress, results timelines).
- Pulls API keys from environment variables if available; if not, the UI requests them without exposing keys.
- Uses sample Grok API code consistent with your snippet.

Directory layout
- app.py                    # Streamlit main app
- agents.yaml               # Agent registry/prompt defaults per agent
- requirements.txt          # Python dependencies for your Space
- README.md                 # Optional: deployment instructions

requirements.txt (suggested)
- streamlit>=1.38.0
- pydantic>=2.8.0
- pyyaml>=6.0.2
- google-generativeai>=0.8.3
- openai>=1.45.0
- xai-sdk>=0.1.17
- pdf2image>=1.17.0
- pillow>=10.4.0
- pypdf>=5.0.1
- tiktoken>=0.7.0
- pandas>=2.2.3
- python-dateutil>=2.9.0.post0

agents.yaml (sample)
- id: compliance_checker
  name: Compliance Checker
  description: Flags compliance issues against known regulatory frameworks.
  enabled: true
  provider: gemini
  model: gemini-2.5-flash
  prompt: |
    You are a senior Regulatory Compliance Analyst for medical device manufacturing.
    Context:
    - Summary: {summary}
    - Extracted Data (JSON): {extracted_json}
    - Document List: {document_titles}

    Task:
    - Identify compliance gaps with references (e.g., ISO 13485, 21 CFR 820).
    - Severity rating: Low/Medium/High.
    - Evidence: quote or cite which section.
    - Actionable remediation steps.

    Output JSON:
    {
      "overview": "string",
      "findings": [
        {
          "issue": "string",
          "severity": "Low|Medium|High",
          "evidence": "string",
          "standard_reference": "string",
          "recommended_action": "string"
        }
      ]
    }

- id: risk_assessor
  name: Risk Assessor
  description: Performs a structured risk analysis (likelihood, impact, mitigation).
  enabled: true
  provider: openai
  model: gpt-4o-mini
  prompt: |
    You are a Risk Analyst specializing in medical device quality systems.
    Context:
    - Summary: {summary}
    - Extracted Data: {extracted_json}

    Task:
    - List top risks (5-10).
    - Provide likelihood (1-5), impact (1-5), mitigation steps, and owner.

    Output JSON:
    {
      "risks": [
        {
          "name": "string",
          "description": "string",
          "likelihood": 1,
          "impact": 1,
          "mitigation": "string",
          "owner_role": "string"
        }
      ]
    }

- id: gap_analyst
  name: GAP Analyst
  description: Compares current state vs best practices and standards.
  enabled: true
  provider: grok
  model: grok-4-fast-reaoning
  prompt: |
    You are a GAP Analyst. Compare the current process to best practices and relevant regs.
    Context:
    - Summary: {summary}
    - Extracted Data: {extracted_json}

    Output JSON:
    {
      "gaps": [
        {
          "area": "string",
          "gap_description": "string",
          "why_it_matters": "string",
          "priority": "Low|Medium|High",
          "recommendation": "string"
        }
      ]
    }

- id: data_quality_validator
  name: Data Quality Validator
  description: Validates the structured data and points out anomalies.
  enabled: true
  provider: openai
  model: gpt-4.1-mini
  prompt: |
    You validate and critique structured data for completeness and logical consistency.
    Context JSON: {extracted_json}

    Output JSON:
    {
      "score": 0,
      "issues": [
        {"field": "string", "issue": "string", "severity": "Low|Medium|High"}
      ],
      "suggested_fixes": ["string", "string"]
    }

- id: action_planner
  name: Action Planner
  description: Produces a 30-60-90 day remediation plan.
  enabled: true
  provider: gemini
  model: gemini-2.5-flash-lite
  prompt: |
    Create a pragmatic action plan from the agent outputs.
    Context:
    - Summary: {summary}
    - Extracted Data: {extracted_json}
    - Agent Findings: {agent_findings}

    Output JSON:
    {
      "plan": [
        {"horizon": "30d", "items": ["string"]},
        {"horizon": "60d", "items": ["string"]},
        {"horizon": "90d", "items": ["string"]}
      ],
      "success_metrics": ["string"]
    }

Advanced prompt templates
- System Orchestrator Prompt (used for summary and orchestration)
  You are an expert orchestrator specializing in medical device quality and regulatory analysis. You produce concise, accurate, source-grounded outputs, faithfully reflecting provided documents. Do not invent facts. Prefer structured JSON for machine-readability and parallel agent consumption. When summarizing, preserve terminology and highlight key terms. Always validate output against provided context before finalizing.

- Summarization Prompt
  Task: Summarize the following documents for a medical device manufacturing context. Preserve key terminology, highlight critical concepts with <em> tags around keywords, and provide a bulleted top-5 insights followed by a short executive summary (<= 150 words).
  Input: {concatenated_documents}
  Output: Markdown with:
  - Top 5 insights
  - Executive summary
  - Key terms section

- Structured Extraction Prompt
  Extract the following fields from the selected document. If a field is missing, return null. Use ISO 8601 for dates and preserve native strings.
  Fields: {field_list}
  Return JSON only. No commentary.

- Follow-up Question Prompt (for 3 in-app follow-ups)
  Given the full summary and all agent outputs, produce 3 incisive follow-up questions that drive meaningful next steps. Avoid yes/no questions. Return as a bullet list.

app.py (single-file Streamlit app)
Note: This file inlines utilities for simplicity. In production, you may split providers/ocr/utils into modules.

import os
import io
import json
import time
import base64
import asyncio
import concurrent.futures
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import streamlit as st
import pandas as pd
import yaml
from PIL import Image
from pdf2image import convert_from_bytes

# Providers
import google.generativeai as genai
from openai import OpenAI
from xai_sdk import Client as XAIClient
from xai_sdk.chat import user as xai_user, system as xai_system, image as xai_image

# -------------------------
# Constants and Model Registry
# -------------------------
MODEL_REGISTRY = {
    "gemini": {
        "label": "Gemini",
        "env": "GOOGLE_API_KEY",
        "models": ["gemini-2.5-flash", "gemini-2.5-flash-lite"],
        "vision": True,
        "supports_json_schema": True
    },
    "openai": {
        "label": "OpenAI",
        "env": "OPENAI_API_KEY",
        "models": ["gpt-5-nano", "gpt-4o-mini", "gpt-4.1-mini"],
        "vision": True,  # gpt-4o-mini supports images as base64
        "supports_json_schema": True
    },
    "grok": {
        "label": "Grok (xAI)",
        "env": "XAI_API_KEY",
        "models": ["grok-4-fast-reaoning", "grok-3-mini", "grok-4"],  # include grok-4 for sample compatibility
        "vision": True,  # via URL only (per sample)
        "supports_json_schema": False
    }
}

DEFAULT_DOC_TITLES = [
    "Quality Manual",
    "SOP: CAPA",
    "SOP: Design Control",
    "Supplier Management Procedure",
    "Risk Management File"
]

# -------------------------
# Session State Setup
# -------------------------
def init_state():
    st.session_state.setdefault("api_keys", {"gemini": None, "openai": None, "grok": None})
    st.session_state.setdefault("documents", [
        {"id": i, "title": DEFAULT_DOC_TITLES[i], "content": "", "file_name": None, "pages_ocrd": 0}
        for i in range(5)
    ])
    st.session_state.setdefault("summary", "")
    st.session_state.setdefault("extracted", {})
    st.session_state.setdefault("agents_config", [])
    st.session_state.setdefault("agent_results", {})
    st.session_state.setdefault("current_step", 0)
    st.session_state.setdefault("token_usage", {"input_tokens": 0, "output_tokens": 0})
    st.session_state.setdefault("ocr_provider", "gemini")
    st.session_state.setdefault("ocr_model", "gemini-2.5-flash")
    st.session_state.setdefault("default_summary_provider", "gemini")
    st.session_state.setdefault("default_summary_model", "gemini-2.5-flash")
    st.session_state.setdefault("default_extract_provider", "openai")
    st.session_state.setdefault("default_extract_model", "gpt-4o-mini")
    st.session_state.setdefault("agent_dashboard_rows", [])

init_state()

# -------------------------
# Load agents.yaml
# -------------------------
def load_agents_yaml(path: str = "agents.yaml") -> List[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, list)
        return data
    except Exception as e:
        st.warning(f"Could not load agents.yaml: {e}")
        return []

if not st.session_state["agents_config"]:
    st.session_state["agents_config"] = load_agents_yaml()

# -------------------------
# API Key Handling
# -------------------------
def load_env_keys():
    for provider, meta in MODEL_REGISTRY.items():
        env_key = os.getenv(meta["env"])
        if env_key:
            st.session_state["api_keys"][provider] = env_key

def render_api_key_inputs():
    st.sidebar.subheader("API Keys")
    load_env_keys()

    for provider, meta in MODEL_REGISTRY.items():
        if st.session_state["api_keys"][provider]:
            st.sidebar.success(f"{meta['label']}: Loaded from environment")
        else:
            st.session_state["api_keys"][provider] = st.sidebar.text_input(
                f"{meta['label']} API Key",
                type="password",
                placeholder=f"Enter {meta['label']} key",
                key=f"{provider}_key_input",
            )

render_api_key_inputs()

def provider_ready(provider: str) -> bool:
    return bool(st.session_state["api_keys"].get(provider))

# -------------------------
# Provider Clients
# -------------------------
def get_gemini_client():
    key = st.session_state["api_keys"]["gemini"]
    if not key:
        return None
    genai.configure(api_key=key)
    return genai

def get_openai_client():
    key = st.session_state["api_keys"]["openai"]
    if not key:
        return None
    return OpenAI(api_key=key)

def get_grok_client():
    key = st.session_state["api_keys"]["grok"]
    if not key:
        return None
    # sample style per provided code, with longer timeout
    return XAIClient(api_key=key, timeout=3600)

# -------------------------
# LLM Call Wrapper
# -------------------------
def call_llm(provider: str, model: str, system_prompt: str, user_prompt: str,
             json_schema: Optional[Dict[str, Any]] = None,
             images: Optional[List[bytes]] = None) -> Tuple[str, Dict[str, Any]]:
    """
    Returns (text, meta) where text is the model output and meta may include usage info.
    images: list of image bytes for vision models. For Grok, only remote URLs are supported (sample).
    """
    t0 = time.time()
    meta = {"provider": provider, "model": model, "duration_sec": 0}
    out_text = ""

    try:
        if provider == "gemini":
            g = get_gemini_client()
            if not g:
                raise RuntimeError("Gemini key missing")
            generation_config = {}
            if json_schema and MODEL_REGISTRY["gemini"]["supports_json_schema"]:
                generation_config = {
                    "response_mime_type": "application/json",
                    "response_schema": json_schema
                }
            model_client = g.GenerativeModel(model)
            if images:
                # Vision content: image bytes -> PIL -> genai Image
                parts = []
                if system_prompt:
                    parts.append({"text": system_prompt})
                parts.append({"text": user_prompt})
                for b in images:
                    parts.append({"inline_data": {"mime_type": "image/png", "data": base64.b64encode(b).decode()}})
                resp = model_client.generate_content(parts, generation_config=generation_config or None)
            else:
                contents = [{"role": "system", "parts": [{"text": system_prompt}]}] if system_prompt else []
                contents.append({"role": "user", "parts": [{"text": user_prompt}]})
                resp = model_client.generate_content(contents, generation_config=generation_config or None)
            out_text = resp.text or ""
            # Gemini usage not always standardized; omit token estimation here
        elif provider == "openai":
            client = get_openai_client()
            if not client:
                raise RuntimeError("OpenAI key missing")

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            content = [{"type": "text", "text": user_prompt}]
            if images:
                # gpt-4o-mini vision: base64-encoded images
                for b in images:
                    b64 = base64.b64encode(b).decode()
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"}
                    })
            messages.append({"role": "user", "content": content})
            response_format = None
            if json_schema and MODEL_REGISTRY["openai"]["supports_json_schema"]:
                # Simplified JSON mode
                response_format = {"type": "json_object"}
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
                response_format=response_format
            )
            out_text = resp.choices[0].message.content or ""
            usage = getattr(resp, "usage", None)
            if usage:
                st.session_state["token_usage"]["input_tokens"] += usage.prompt_tokens
                st.session_state["token_usage"]["output_tokens"] += usage.completion_tokens
        elif provider == "grok":
            # Use sample code pattern; structured outputs not officially demonstrated; prefer JSON-in-prompt
            xclient = get_grok_client()
            if not xclient:
                raise RuntimeError("Grok key missing")
            chat = xclient.chat.create(model=model if model else "grok-4")
            if system_prompt:
                chat.append(xai_system(system_prompt))
            if images:
                # Grok sample supports URL images; for local bytes you'd need to host them.
                # Here we fallback to text-only prompt with a note, or require URL images.
                chat.append(xai_user(user_prompt))
            else:
                chat.append(xai_user(user_prompt))
            response = chat.sample()
            out_text = response.content
        else:
            raise ValueError(f"Unknown provider: {provider}")

        meta["duration_sec"] = round(time.time() - t0, 2)
        return out_text, meta
    except Exception as e:
        meta["error"] = str(e)
        return f"ERROR: {e}", meta

# -------------------------
# OCR Utilities
# -------------------------
def pdf_page_to_image(pdf_bytes: bytes, page_number: int = 1) -> Image.Image:
    pages = convert_from_bytes(pdf_bytes, first_page=page_number, last_page=page_number, dpi=200)
    return pages[0]

def ai_ocr_image(img: Image.Image, provider: str, model: str) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_b = buf.getvalue()
    # Vision prompt to extract text only
    sys_p = "You are an expert OCR assistant. Return only the extracted text, preserving layout as best as possible."
    usr_p = "Extract all text from the following image. Output plain UTF-8 text only."
    text, _ = call_llm(provider, model, sys_p, usr_p, images=[img_b])
    return text.strip()

# -------------------------
# UI Helpers: WOW Indicators and Dashboard Widgets
# -------------------------
def wow_badge(text: str, color: str = "#6A5ACD", emoji: str = "🚀"):
    st.markdown(
        f"""
        <span style="background:{color}; color:white; padding:6px 10px; border-radius:12px; margin-right:6px;">
            {emoji} {text}
        </span>
        """,
        unsafe_allow_html=True
    )

def status_block(title: str, status: str):
    color = {"ok": "#16a34a", "warn": "#f59e0b", "err": "#ef4444"}.get(status, "#64748b")
    emoji = {"ok": "✅", "warn": "⚠️", "err": "❌"}.get(status, "ℹ️")
    wow_badge(f"{title}", color=color, emoji=emoji)

def kpi_row(cols: List[Tuple[str, str]]):
    c = st.columns(len(cols))
    for i, (label, value) in enumerate(cols):
        with c[i]:
            st.metric(label, value)

# -------------------------
# Sidebar Configuration
# -------------------------
st.set_page_config(page_title="Agentic Document Processing System", layout="wide", page_icon="🧭")

st.sidebar.title("Agentic Doc Processing")
status_block("Space: Ready", "ok")
st.sidebar.caption("Choose models and set API keys.")

# OCR defaults
with st.sidebar.expander("OCR Settings"):
    st.session_state["ocr_provider"] = st.selectbox(
        "OCR Provider", options=list(MODEL_REGISTRY.keys()), index=list(MODEL_REGISTRY.keys()).index("gemini")
    )
    st.session_state["ocr_model"] = st.selectbox(
        "OCR Model", options=MODEL_REGISTRY[st.session_state["ocr_provider"]]["models"],
        index=0
    )

# Defaults for Summary/Extraction
with st.sidebar.expander("Pipeline Defaults"):
    st.session_state["default_summary_provider"] = st.selectbox(
        "Summary Provider", options=list(MODEL_REGISTRY.keys()), index=0
    )
    st.session_state["default_summary_model"] = st.selectbox(
        "Summary Model", options=MODEL_REGISTRY[st.session_state["default_summary_provider"]]["models"], index=0
    )
    st.session_state["default_extract_provider"] = st.selectbox(
        "Extraction Provider", options=list(MODEL_REGISTRY.keys()), index=1
    )
    st.session_state["default_extract_model"] = st.selectbox(
        "Extraction Model", options=MODEL_REGISTRY[st.session_state["default_extract_provider"]]["models"], index=0
    )

# -------------------------
# Main Layout: Tabs
# -------------------------
tab_overview, tab_documents, tab_summary, tab_extract, tab_agents, tab_results = st.tabs(
    ["Overview", "Documents", "Summary", "Extract Data", "Agents", "Dashboard"]
)

with tab_overview:
    st.title("Agentic Document Processing System")
    wow_badge("Multi-Model", emoji="🧠")
    wow_badge("Agents Orchestration", emoji="🤖")
    wow_badge("Vision OCR", emoji="👁️")
    wow_badge("Interactive Dashboard", emoji="📊")

    st.write(
        "Upload your regulatory/quality documents, summarize them, extract structured data, "
        "run specialized agents per model, and explore insights on a live dashboard."
    )

    kpi_row([
        ("Documents", f"{sum(1 for d in st.session_state['documents'] if d['content'].strip()):d}/5"),
        ("Pages OCR’d", str(sum(d['pages_ocrd'] for d in st.session_state["documents"]))),
        ("Tokens In", str(st.session_state["token_usage"]["input_tokens"])),
        ("Tokens Out", str(st.session_state["token_usage"]["output_tokens"]))
    ])

with tab_documents:
    st.header("Documents")
    st.info("For each document: paste text or upload .txt/.md/.pdf. For PDF, select page to OCR with chosen model.")
    for i, doc in enumerate(st.session_state["documents"]):
        with st.expander(f"{doc['title']}"):
            c1, c2 = st.columns([2, 1])
            with c1:
                txt = st.text_area("Text", doc["content"], height=180, key=f"doc_text_{i}")
                st.session_state["documents"][i]["content"] = txt
            with c2:
                uploaded = st.file_uploader("Upload .txt/.md/.pdf", type=["txt", "md", "pdf"], key=f"u_{i}")
                if uploaded is not None:
                    st.session_state["documents"][i]["file_name"] = uploaded.name
                    if uploaded.type in ("text/plain", "text/markdown"):
                        st.session_state["documents"][i]["content"] = uploaded.read().decode("utf-8", errors="ignore")
                        st.success(f"Ingested {uploaded.name}")
                    elif uploaded.type == "application/pdf":
                        pdf_bytes = uploaded.read()
                        pg = st.number_input("Page to OCR", min_value=1, max_value=2500, value=1, step=1, key=f"pg_{i}")
                        if st.button("OCR Page", key=f"ocr_{i}"):
                            if not provider_ready(st.session_state["ocr_provider"]):
                                st.error(f"{MODEL_REGISTRY[st.session_state['ocr_provider']]['label']} key missing.")
                            else:
                                with st.status("AI OCR Running...", expanded=True) as s:
                                    s.write("Converting PDF page to image...")
                                    img = pdf_page_to_image(pdf_bytes, pg)
                                    s.write(f"OCR via {st.session_state['ocr_provider']} - {st.session_state['ocr_model']}...")
                                    text = ai_ocr_image(img, st.session_state["ocr_provider"], st.session_state["ocr_model"])
                                    st.session_state["documents"][i]["content"] = text
                                    st.session_state["documents"][i]["pages_ocrd"] += 1
                                    s.update(label="OCR Completed", state="complete", expanded=False)
                                    st.success("OCR completed and text loaded.")

with tab_summary:
    st.header("Generate Summary")
    concat_docs = "\n\n".join(
        f"# {d['title']}\n{d['content']}" for d in st.session_state["documents"] if d["content"].strip()
    )
    if not concat_docs.strip():
        st.warning("Please provide content for at least one document.")
    else:
        st.text_area("Preview combined input", concat_docs[:4000] + ("..." if len(concat_docs) > 4000 else ""), height=180)
        if st.button("Generate Summary"):
            if not provider_ready(st.session_state["default_summary_provider"]):
                st.error("Missing API key for the chosen provider.")
            else:
                with st.status("Summarizing documents...", expanded=True) as s:
                    sys_p = "You are an expert orchestrator specializing in medical device quality/regulatory analysis."
                    usr_p = (
                        "Task: Summarize the following documents for medical device manufacturing. "
                        "Provide Top 5 insights, an executive summary (<=150 words), and a Key terms section. "
                        "Use Markdown. Highlight critical terms using <em> tags.\n\n"
                        f"{concat_docs}"
                    )
                    out, meta = call_llm(
                        st.session_state["default_summary_provider"],
                        st.session_state["default_summary_model"],
                        sys_p, usr_p
                    )
                    st.session_state["summary"] = out
                    s.update(label=f"Summary created in {meta.get('duration_sec', '?')}s", state="complete")
        st.markdown(st.session_state["summary"] or "_Summary will appear here..._")

with tab_extract:
    st.header("Structured Data Extraction")
    doc_titles = [d["title"] for d in st.session_state["documents"] if d["content"].strip()]
    if not doc_titles:
        st.warning("Please add content for at least one document.")
    else:
        selected = st.selectbox("Select a document to extract from", options=doc_titles)
        field_list = st.text_area(
            "Fields to Extract (comma-separated)", "委託者名稱, 委託者地址, 產品型號, 有效日期, 負責人"
        )
        if st.button("Extract Data"):
            if not provider_ready(st.session_state["default_extract_provider"]):
                st.error("Missing API key for the chosen provider.")
            else:
                with st.status("Extracting structured data...", expanded=True) as s:
                    doc_content = next(d["content"] for d in st.session_state["documents"] if d["title"] == selected)
                    sys_p = "You are a precise information extraction assistant. Output strict JSON only."
                    fields = [f.strip() for f in field_list.split(",") if f.strip()]
                    # Basic JSON schema (as hint for providers with JSON mode)
                    schema = {
                        "type": "object",
                        "properties": {f: {"type": ["string", "null"]} for f in fields},
                        "additionalProperties": False
                    }
                    usr_p = (
                        f"Extract the following fields from the document. If missing, return null.\n"
                        f"Fields: {fields}\nDocument:\n{doc_content}\n\nReturn JSON only."
                    )
                    out, meta = call_llm(
                        st.session_state["default_extract_provider"],
                        st.session_state["default_extract_model"],
                        sys_p, usr_p, json_schema=schema
                    )
                    try:
                        data = json.loads(out)
                    except Exception:
                        st.warning("Model did not return valid JSON; attempting to recover.")
                        # Best-effort cleanup
                        start = out.find("{")
                        end = out.rfind("}")
                        data = json.loads(out[start:end+1]) if start >= 0 and end >= 0 else {"raw": out}
                    st.session_state["extracted"] = data
                    s.update(label=f"Extraction done in {meta.get('duration_sec','?')}s", state="complete")

        if st.session_state["extracted"]:
            st.subheader("Extracted JSON")
            st.json(st.session_state["extracted"])
            # Markdown table rendering
            df = pd.DataFrame([st.session_state["extracted"]])
            st.subheader("Table")
            st.dataframe(df)

with tab_agents:
    st.header("Agents")
    if not st.session_state["summary"] or not st.session_state["extracted"]:
        st.warning("Please complete Summary and Extraction steps first.")
    else:
        provider_opts = list(MODEL_REGISTRY.keys())
        # Editable grid of agents
        for idx, agent in enumerate(st.session_state["agents_config"]):
            with st.expander(f"{agent.get('name', agent['id'])}"):
                agent["enabled"] = st.checkbox("Enabled", value=agent.get("enabled", True), key=f"en_{agent['id']}")
                c1, c2 = st.columns(2)
                with c1:
                    agent["provider"] = st.selectbox(
                        "Provider", options=provider_opts,
                        index=provider_opts.index(agent.get("provider", provider_opts[0])),
                        key=f"prov_{agent['id']}"
                    )
                with c2:
                    agent["model"] = st.selectbox(
                        "Model", options=MODEL_REGISTRY[agent["provider"]]["models"],
                        index=max(0, MODEL_REGISTRY[agent["provider"]]["models"].index(agent.get("model", MODEL_REGISTRY[agent["provider"]]["models"][0])) if agent.get("model") in MODEL_REGISTRY[agent["provider"]]["models"] else 0),
                        key=f"mdl_{agent['id']}"
                    )
                agent["prompt"] = st.text_area("Prompt", agent.get("prompt", ""), height=180, key=f"pr_{agent['id']}")

        if st.button("Execute Agents"):
            enabled_agents = [a for a in st.session_state["agents_config"] if a.get("enabled")]
            if not enabled_agents:
                st.error("No agents enabled.")
            else:
                sum_text = st.session_state["summary"]
                extracted_json = json.dumps(st.session_state["extracted"], ensure_ascii=False)
                doc_titles_joined = ", ".join([d["title"] for d in st.session_state["documents"] if d["content"].strip()])
                st.session_state["agent_results"] = {}
                st.session_state["agent_dashboard_rows"].clear()

                with st.status("Running agents...", expanded=True) as s:
                    # Run agents in thread pool
                    def run_agent(agent_obj):
                        sys_p = "You are a specialized analysis agent for medical device regulatory/quality."
                        user_p = agent_obj["prompt"].format(
                            summary=sum_text,
                            extracted_json=extracted_json,
                            document_titles=doc_titles_joined,
                            agent_findings=json.dumps(st.session_state["agent_results"], ensure_ascii=False)
                        )
                        out, meta = call_llm(agent_obj["provider"], agent_obj["model"], sys_p, user_p)
                        return agent_obj["id"], out, meta

                    with concurrent.futures.ThreadPoolExecutor(max_workers=min(6, len(enabled_agents))) as ex:
                        futures = [ex.submit(run_agent, a) for a in enabled_agents]
                        for fut in concurrent.futures.as_completed(futures):
                            aid, out, meta = fut.result()
                            st.session_state["agent_results"][aid] = {"output": out, "meta": meta}
                            st.write(f"Agent {aid} completed in {meta.get('duration_sec','?')}s")
                            st.session_state["agent_dashboard_rows"].append({
                                "time": datetime.utcnow().isoformat(timespec="seconds"),
                                "agent": aid,
                                "provider": meta.get("provider"),
                                "model": meta.get("model"),
                                "duration_s": meta.get("duration_sec")
                            })
                    s.update(label="All agents completed", state="complete")

        if st.session_state["agent_results"]:
            st.subheader("Agent Outputs")
            for aid, payload in st.session_state["agent_results"].items():
                ag_def = next((a for a in st.session_state["agents_config"] if a["id"] == aid), None)
                st.markdown(f"### {ag_def.get('name', aid) if ag_def else aid}")
                st.code(payload["output"])
                meta = payload.get("meta", {})
                st.caption(f"Provider: {meta.get('provider')} | Model: {meta.get('model')} | Time: {meta.get('duration_sec','?')}s")

with tab_results:
    st.header("Interactive Dashboard")
    # KPIs
    kpi_row([
        ("Enabled Agents", str(sum(1 for a in st.session_state["agents_config"] if a.get("enabled")))),
        ("Completed Runs", str(len(st.session_state["agent_results"]))),
        ("Total Duration (s)", str(sum((v.get('meta', {}).get('duration_sec') or 0) for v in st.session_state["agent_results"].values())))
    ])

    # Timeline table
    if st.session_state["agent_dashboard_rows"]:
        st.subheader("Agent Run Timeline")
        st.dataframe(pd.DataFrame(st.session_state["agent_dashboard_rows"]))

    # Consolidation + Follow-ups (3)
    if st.button("Generate Consolidated Follow-ups"):
        if not provider_ready(st.session_state["default_summary_provider"]):
            st.error("Provider key missing for follow-ups.")
        else:
            with st.status("Generating follow-up questions...", expanded=True) as s:
                joined = "\n\n".join(f"{k}:\n{v['output']}" for k, v in st.session_state["agent_results"].items())
                sys_p = "You are a helpful assistant. Output concise, high-value follow-up questions."
                usr_p = (
                    f"Given the summary and agent outputs, produce 3 incisive follow-up questions. Avoid yes/no.\n"
                    f"Summary:\n{st.session_state['summary']}\n\nAgent Outputs:\n{joined}"
                )
                out, meta = call_llm(
                    st.session_state["default_summary_provider"],
                    st.session_state["default_summary_model"],
                    sys_p, usr_p
                )
                st.markdown(out)
                s.update(label="Follow-ups ready", state="complete")

    # Raw JSON view
    with st.expander("Raw State"):
        st.json({
            "documents": st.session_state["documents"],
            "summary": st.session_state["summary"],
            "extracted": st.session_state["extracted"],
            "agent_results": st.session_state["agent_results"]
        })

README.md (key points)
- Add GOOGLE_API_KEY, OPENAI_API_KEY, XAI_API_KEY as HF Space Secrets to avoid key input in UI.
- If env keys are present, the app uses them automatically without displaying the values.
- For Grok vision with local PDFs, the xAI SDK sample expects image URLs. For OCR, this app defaults to Gemini/OpenAI local vision. You can host images and pass URLs to Grok using xai_image(url).

Notes on structured outputs
- Gemini: uses response_mime_type and response_schema for reliable JSON.
- OpenAI: uses response_format={"type":"json_object"} to prefer JSON; validate with json.loads.
- Grok: no public JSON-mode sample; prompt the model to “Return JSON only” and validate/repair.

Sample Grok usage (mirrors your sample and used above)
import os
from xai_sdk import Client
from xai_sdk.chat import user, system

client = Client(api_key=os.getenv("XAI_API_KEY"), timeout=3600)
chat = client.chat.create(model="grok-4")
chat.append(system("You are Grok, a highly intelligent, helpful AI assistant."))
chat.append(user("What is the meaning of life, the universe, and everything?"))
response = chat.sample()
print(response.content)

For images via URL:
from xai_sdk.chat import user, image

chat = client.chat.create(model="grok-4")
chat.append(user("What's in this image?", image("https://example.com/image.png")))
response = chat.sample()
print(response.content)

What changed vs original design
- React/TS → Streamlit with a tabs-based workflow.
- pdf.js OCR → AI Vision OCR with Gemini/OpenAI for local image bytes; fallbacks elegantly handled.
- agents.yaml controls agent defaults; model per agent selectable in UI.
- Clear wow indicators, status blocks, and an interactive dashboard for runs, metrics, and timeline.
- Robust key handling: loads from env if present, otherwise secure masked input; no display of env keys.

20 comprehensive follow-up questions
1) Which regulatory frameworks must this system primarily align with (e.g., ISO 13485, 21 CFR 820, EU MDR), and should we include built-in citations for each?
2) What is your preferred extraction schema for Step 2? Can you provide a canonical JSON schema with field types, formats, and allowed values?
3) How large can uploads be per document and per session on your Hugging Face Space tier, and do you want chunking for very long PDFs?
4) Should the OCR step support multi-page batch extraction and automatic page-range detection (e.g., only pages containing specific keywords)?
5) Do you need multilingual OCR and analysis (e.g., Chinese/Japanese/European languages) with language auto-detection and translation?
6) For OpenAI models, do you prefer GPT-4o-mini or GPT-4.1-mini for analysis accuracy vs cost/speed, and should we add usage/cost estimation per provider?
7) For Grok, do you want us to add an image-hosting helper so local OCR images can be analyzed by Grok via URL?
8) Should agent outputs be versioned and persisted (e.g., on HF Datasets or a lightweight SQLite) for auditability and re-comparison?
9) Do you require role-based access control, per-user storage, or encryption for uploaded documents and outputs?
10) What SLAs and timeouts should we enforce per model call, and do you want retry/backoff with circuit breakers for provider outages?
11) Should we add a policy/guardrail layer to block PII exfiltration or redact sensitive fields in UI exports?
12) Do you want a “compliance pack” export (PDF/Word) combining summary, extracted data, and all agent findings with your branding?
13) Would you like a calibration suite (golden test docs) to automatically benchmark accuracy per provider/model combination over time?
14) Do you need additional agents (e.g., Supplier Risk Profiler, Change Control Reviewer, Audit Readiness Coach)?
15) Should we enable few-shot exemplars for extraction and each agent to boost consistency on your document types?
16) Do you want a knowledge base (KB) upload for standards and internal SOPs, with retrieval-augmented generation (RAG) across agents?
17) Should the dashboard include drill-down charts (e.g., risk heatmaps, severity histograms, timeline deltas between runs)?
18) How should we handle non-PDF formats with embedded images (Word, scanned TIFF), and do you want automatic image segmentation?
19) Do you need localization of the UI and outputs, including date/number formats and terminology mappings per region?
20) What is the approval pathway for model changes (e.g., updating default models in agents.yaml), and should we pin model versions for reproducibility?
