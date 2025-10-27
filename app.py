import os
import io
import json
import time
import base64
import concurrent.futures
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

import streamlit as st
import pandas as pd
import yaml
from PIL import Image

try:
    from pdf2image import convert_from_bytes
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    st.warning("pdf2image not available. PDF OCR disabled.")

# Providers
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from xai_sdk import Client as XAIClient
    from xai_sdk.chat import user as xai_user, system as xai_system
    GROK_AVAILABLE = True
except ImportError:
    GROK_AVAILABLE = False

# -------------------------
# THEME DEFINITIONS (20 Animal Themes)
# -------------------------
THEMES = {
    "🦁 Lion": {
        "primary": "#D4A574",
        "secondary": "#8B6914",
        "accent": "#FFD700",
        "bg": "#FFF8DC",
        "text": "#2C1810",
        "gradient": "linear-gradient(135deg, #D4A574 0%, #FFD700 100%)"
    },
    "🐼 Panda": {
        "primary": "#2C3E50",
        "secondary": "#ECF0F1",
        "accent": "#95A5A6",
        "bg": "#FFFFFF",
        "text": "#000000",
        "gradient": "linear-gradient(135deg, #2C3E50 0%, #ECF0F1 100%)"
    },
    "🦊 Fox": {
        "primary": "#E67E22",
        "secondary": "#D35400",
        "accent": "#F39C12",
        "bg": "#FDF5E6",
        "text": "#5D4037",
        "gradient": "linear-gradient(135deg, #E67E22 0%, #F39C12 100%)"
    },
    "🐺 Wolf": {
        "primary": "#546E7A",
        "secondary": "#37474F",
        "accent": "#78909C",
        "bg": "#ECEFF1",
        "text": "#263238",
        "gradient": "linear-gradient(135deg, #546E7A 0%, #78909C 100%)"
    },
    "🦅 Eagle": {
        "primary": "#795548",
        "secondary": "#5D4037",
        "accent": "#A1887F",
        "bg": "#EFEBE9",
        "text": "#3E2723",
        "gradient": "linear-gradient(135deg, #795548 0%, #A1887F 100%)"
    },
    "🐯 Tiger": {
        "primary": "#FF6F00",
        "secondary": "#E65100",
        "accent": "#FFB300",
        "bg": "#FFF3E0",
        "text": "#BF360C",
        "gradient": "linear-gradient(135deg, #FF6F00 0%, #FFB300 100%)"
    },
    "🐘 Elephant": {
        "primary": "#757575",
        "secondary": "#616161",
        "accent": "#9E9E9E",
        "bg": "#FAFAFA",
        "text": "#212121",
        "gradient": "linear-gradient(135deg, #757575 0%, #9E9E9E 100%)"
    },
    "🦒 Giraffe": {
        "primary": "#F9A825",
        "secondary": "#F57F17",
        "accent": "#FDD835",
        "bg": "#FFFDE7",
        "text": "#827717",
        "gradient": "linear-gradient(135deg, #F9A825 0%, #FDD835 100%)"
    },
    "🐧 Penguin": {
        "primary": "#0277BD",
        "secondary": "#01579B",
        "accent": "#4FC3F7",
        "bg": "#E1F5FE",
        "text": "#01579B",
        "gradient": "linear-gradient(135deg, #0277BD 0%, #4FC3F7 100%)"
    },
    "🦋 Butterfly": {
        "primary": "#8E24AA",
        "secondary": "#6A1B9A",
        "accent": "#CE93D8",
        "bg": "#F3E5F5",
        "text": "#4A148C",
        "gradient": "linear-gradient(135deg, #8E24AA 0%, #CE93D8 100%)"
    },
    "🐬 Dolphin": {
        "primary": "#0288D1",
        "secondary": "#01579B",
        "accent": "#4DD0E1",
        "bg": "#E0F7FA",
        "text": "#006064",
        "gradient": "linear-gradient(135deg, #0288D1 0%, #4DD0E1 100%)"
    },
    "🦎 Chameleon": {
        "primary": "#388E3C",
        "secondary": "#2E7D32",
        "accent": "#81C784",
        "bg": "#E8F5E9",
        "text": "#1B5E20",
        "gradient": "linear-gradient(135deg, #388E3C 0%, #81C784 100%)"
    },
    "🦉 Owl": {
        "primary": "#5D4E37",
        "secondary": "#3E2723",
        "accent": "#8D6E63",
        "bg": "#EFEBE9",
        "text": "#3E2723",
        "gradient": "linear-gradient(135deg, #5D4E37 0%, #8D6E63 100%)"
    },
    "🦩 Flamingo": {
        "primary": "#EC407A",
        "secondary": "#C2185B",
        "accent": "#F48FB1",
        "bg": "#FCE4EC",
        "text": "#880E4F",
        "gradient": "linear-gradient(135deg, #EC407A 0%, #F48FB1 100%)"
    },
    "🐝 Bee": {
        "primary": "#FBC02D",
        "secondary": "#F57F17",
        "accent": "#FFEB3B",
        "bg": "#FFFDE7",
        "text": "#F57F17",
        "gradient": "linear-gradient(135deg, #FBC02D 0%, #FFEB3B 100%)"
    },
    "🦜 Parrot": {
        "primary": "#00BFA5",
        "secondary": "#00897B",
        "accent": "#64FFDA",
        "bg": "#E0F2F1",
        "text": "#004D40",
        "gradient": "linear-gradient(135deg, #00BFA5 0%, #64FFDA 100%)"
    },
    "🐨 Koala": {
        "primary": "#A1887F",
        "secondary": "#8D6E63",
        "accent": "#BCAAA4",
        "bg": "#EFEBE9",
        "text": "#4E342E",
        "gradient": "linear-gradient(135deg, #A1887F 0%, #BCAAA4 100%)"
    },
    "🦈 Shark": {
        "primary": "#455A64",
        "secondary": "#37474F",
        "accent": "#78909C",
        "bg": "#ECEFF1",
        "text": "#263238",
        "gradient": "linear-gradient(135deg, #455A64 0%, #78909C 100%)"
    },
    "🦌 Deer": {
        "primary": "#8D6E63",
        "secondary": "#6D4C41",
        "accent": "#A1887F",
        "bg": "#EFEBE9",
        "text": "#4E342E",
        "gradient": "linear-gradient(135deg, #8D6E63 0%, #A1887F 100%)"
    },
    "🦓 Zebra": {
        "primary": "#424242",
        "secondary": "#212121",
        "accent": "#757575",
        "bg": "#FAFAFA",
        "text": "#000000",
        "gradient": "linear-gradient(135deg, #424242 0%, #757575 100%)"
    }
}

# -------------------------
# Model Registry
# -------------------------
MODEL_REGISTRY = {
    "gemini": {
        "label": "Gemini",
        "env": "GOOGLE_API_KEY",
        "models": ["gemini-2.0-flash-exp", "gemini-1.5-flash", "gemini-1.5-pro"],
        "vision": True,
        "supports_json_schema": True,
        "available": GEMINI_AVAILABLE
    },
    "openai": {
        "label": "OpenAI",
        "env": "OPENAI_API_KEY",
        "models": ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"],
        "vision": True,
        "supports_json_schema": True,
        "available": OPENAI_AVAILABLE
    },
    "grok": {
        "label": "Grok (xAI)",
        "env": "XAI_API_KEY",
        "models": ["grok-beta", "grok-vision-beta"],
        "vision": True,
        "supports_json_schema": False,
        "available": GROK_AVAILABLE
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
    st.session_state.setdefault("theme", "🦁 Lion")
    st.session_state.setdefault("api_keys", {"gemini": None, "openai": None, "grok": None})
    st.session_state.setdefault("documents", [
        {"id": i, "title": DEFAULT_DOC_TITLES[i], "content": "", "file_name": None, "pages_ocrd": 0}
        for i in range(5)
    ])
    st.session_state.setdefault("summary", "")
    st.session_state.setdefault("extracted", {})
    st.session_state.setdefault("agents_config", [])
    st.session_state.setdefault("agent_results", {})
    st.session_state.setdefault("token_usage", {"input_tokens": 0, "output_tokens": 0})
    st.session_state.setdefault("ocr_provider", "gemini")
    st.session_state.setdefault("ocr_model", "gemini-1.5-flash")
    st.session_state.setdefault("default_summary_provider", "gemini")
    st.session_state.setdefault("default_summary_model", "gemini-1.5-flash")
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
        if not isinstance(data, list):
            st.error("agents.yaml must contain a list of agents")
            return []
        return data
    except FileNotFoundError:
        st.warning(f"agents.yaml not found at {path}")
        return []
    except Exception as e:
        st.error(f"Error loading agents.yaml: {e}")
        return []

if not st.session_state["agents_config"]:
    st.session_state["agents_config"] = load_agents_yaml()

# -------------------------
# Apply Theme CSS
# -------------------------
def apply_theme(theme_name: str):
    theme = THEMES[theme_name]
    st.markdown(f"""
    <style>
        .stApp {{
            background: {theme['bg']};
        }}
        .css-1d391kg, .css-18e3th9 {{
            background: {theme['gradient']};
        }}
        h1, h2, h3, h4, h5, h6 {{
            color: {theme['primary']} !important;
        }}
        .stButton>button {{
            background: {theme['primary']};
            color: white;
            border-radius: 10px;
            border: none;
            padding: 0.5rem 1rem;
            font-weight: 600;
        }}
        .stButton>button:hover {{
            background: {theme['secondary']};
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}
        .stSelectbox, .stTextInput, .stTextArea {{
            border-radius: 8px;
        }}
        .metric-card {{
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            border-left: 4px solid {theme['accent']};
        }}
        .wow-badge {{
            background: {theme['primary']};
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            display: inline-block;
            margin: 4px;
            font-weight: 600;
            box-shadow: 0 2px 6px rgba(0,0,0,0.2);
        }}
    </style>
    """, unsafe_allow_html=True)

# -------------------------
# API Key Handling
# -------------------------
def load_env_keys():
    for provider, meta in MODEL_REGISTRY.items():
        if not meta.get("available", True):
            continue
        env_key = os.getenv(meta["env"])
        if env_key:
            st.session_state["api_keys"][provider] = env_key

def render_api_key_inputs():
    st.sidebar.subheader("🔐 API Keys")
    load_env_keys()

    for provider, meta in MODEL_REGISTRY.items():
        if not meta.get("available", True):
            st.sidebar.warning(f"⚠️ {meta['label']} SDK not installed")
            continue
            
        if st.session_state["api_keys"][provider]:
            st.sidebar.success(f"✅ {meta['label']}: Loaded")
        else:
            key = st.sidebar.text_input(
                f"{meta['label']} API Key",
                type="password",
                placeholder=f"Enter {meta['label']} key",
                key=f"{provider}_key_input",
            )
            if key:
                st.session_state["api_keys"][provider] = key

def provider_ready(provider: str) -> bool:
    meta = MODEL_REGISTRY.get(provider, {})
    return meta.get("available", False) and bool(st.session_state["api_keys"].get(provider))

# -------------------------
# Provider Clients
# -------------------------
def get_gemini_client():
    if not GEMINI_AVAILABLE:
        return None
    key = st.session_state["api_keys"]["gemini"]
    if not key:
        return None
    genai.configure(api_key=key)
    return genai

def get_openai_client():
    if not OPENAI_AVAILABLE:
        return None
    key = st.session_state["api_keys"]["openai"]
    if not key:
        return None
    return OpenAI(api_key=key)

def get_grok_client():
    if not GROK_AVAILABLE:
        return None
    key = st.session_state["api_keys"]["grok"]
    if not key:
        return None
    return XAIClient(api_key=key, timeout=3600)

# -------------------------
# LLM Call Wrapper
# -------------------------
def call_llm(provider: str, model: str, system_prompt: str, user_prompt: str,
             json_schema: Optional[Dict[str, Any]] = None,
             images: Optional[List[bytes]] = None) -> Tuple[str, Dict[str, Any]]:
    t0 = time.time()
    meta = {"provider": provider, "model": model, "duration_sec": 0}
    out_text = ""

    try:
        if provider == "gemini":
            g = get_gemini_client()
            if not g:
                raise RuntimeError("Gemini client unavailable")
            
            generation_config = {}
            if json_schema and MODEL_REGISTRY["gemini"]["supports_json_schema"]:
                generation_config = {
                    "response_mime_type": "application/json",
                    "response_schema": json_schema
                }
            
            model_client = g.GenerativeModel(model)
            
            if images:
                parts = []
                if system_prompt:
                    parts.append(system_prompt)
                parts.append(user_prompt)
                for img_bytes in images:
                    pil_img = Image.open(io.BytesIO(img_bytes))
                    parts.append(pil_img)
                resp = model_client.generate_content(parts, generation_config=generation_config or None)
            else:
                prompt = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
                resp = model_client.generate_content(prompt, generation_config=generation_config or None)
            
            out_text = resp.text or ""
            
        elif provider == "openai":
            client = get_openai_client()
            if not client:
                raise RuntimeError("OpenAI client unavailable")

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            content = [{"type": "text", "text": user_prompt}]
            if images:
                for img_bytes in images:
                    b64 = base64.b64encode(img_bytes).decode()
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"}
                    })
            messages.append({"role": "user", "content": content})
            
            kwargs = {"model": model, "messages": messages, "temperature": 0.2}
            if json_schema and MODEL_REGISTRY["openai"]["supports_json_schema"]:
                kwargs["response_format"] = {"type": "json_object"}
            
            resp = client.chat.completions.create(**kwargs)
            out_text = resp.choices[0].message.content or ""
            
            if hasattr(resp, "usage") and resp.usage:
                st.session_state["token_usage"]["input_tokens"] += resp.usage.prompt_tokens
                st.session_state["token_usage"]["output_tokens"] += resp.usage.completion_tokens
                
        elif provider == "grok":
            xclient = get_grok_client()
            if not xclient:
                raise RuntimeError("Grok client unavailable")
            
            chat = xclient.chat.create(model=model)
            if system_prompt:
                chat.append(xai_system(system_prompt))
            chat.append(xai_user(user_prompt))
            response = chat.sample()
            out_text = response.content
        else:
            raise ValueError(f"Unknown provider: {provider}")

        meta["duration_sec"] = round(time.time() - t0, 2)
        return out_text, meta
        
    except Exception as e:
        meta["error"] = str(e)
        meta["duration_sec"] = round(time.time() - t0, 2)
        return f"ERROR: {e}", meta

# -------------------------
# OCR Utilities
# -------------------------
def pdf_page_to_image(pdf_bytes: bytes, page_number: int = 1) -> Image.Image:
    if not PDF_AVAILABLE:
        raise RuntimeError("pdf2image not available")
    pages = convert_from_bytes(pdf_bytes, first_page=page_number, last_page=page_number, dpi=200)
    return pages[0]

def ai_ocr_image(img: Image.Image, provider: str, model: str) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_b = buf.getvalue()
    
    sys_p = "You are an expert OCR assistant. Extract all text accurately, preserving layout and formatting."
    usr_p = "Extract all text from this image. Return plain UTF-8 text only, preserving structure."
    text, _ = call_llm(provider, model, sys_p, usr_p, images=[img_b])
    return text.strip()

# -------------------------
# UI Helpers
# -------------------------
def wow_badge(text: str, emoji: str = "🚀"):
    st.markdown(f'<div class="wow-badge">{emoji} {text}</div>', unsafe_allow_html=True)

def kpi_card(label: str, value: str):
    st.markdown(f"""
    <div class="metric-card">
        <h4 style="margin:0; color: #666;">{label}</h4>
        <h2 style="margin:0.5rem 0 0 0;">{value}</h2>
    </div>
    """, unsafe_allow_html=True)

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="Agentic Document Processing System",
    layout="wide",
    page_icon="🧭",
    initial_sidebar_state="expanded"
)

# Apply selected theme
apply_theme(st.session_state["theme"])

# -------------------------
# Sidebar
# -------------------------
st.sidebar.title("🧭 Agentic Doc Processing")

# Theme Selector
st.sidebar.subheader("🎨 Choose Your Theme")
theme_choice = st.sidebar.selectbox(
    "Select Animal Theme",
    options=list(THEMES.keys()),
    index=list(THEMES.keys()).index(st.session_state["theme"]),
    key="theme_selector"
)
if theme_choice != st.session_state["theme"]:
    st.session_state["theme"] = theme_choice
    st.rerun()

render_api_key_inputs()

# Settings
with st.sidebar.expander("⚙️ OCR Settings"):
    available_providers = [p for p, m in MODEL_REGISTRY.items() if m.get("available", False)]
    if available_providers:
        st.session_state["ocr_provider"] = st.selectbox(
            "OCR Provider",
            options=available_providers,
            index=0 if "gemini" not in available_providers else available_providers.index("gemini")
        )
        st.session_state["ocr_model"] = st.selectbox(
            "OCR Model",
            options=MODEL_REGISTRY[st.session_state["ocr_provider"]]["models"],
            index=0
        )

with st.sidebar.expander("🔧 Pipeline Defaults"):
    if available_providers:
        st.session_state["default_summary_provider"] = st.selectbox(
            "Summary Provider", options=available_providers, index=0, key="sum_prov"
        )
        st.session_state["default_summary_model"] = st.selectbox(
            "Summary Model",
            options=MODEL_REGISTRY[st.session_state["default_summary_provider"]]["models"],
            index=0, key="sum_mod"
        )
        st.session_state["default_extract_provider"] = st.selectbox(
            "Extraction Provider", options=available_providers, index=0, key="ext_prov"
        )
        st.session_state["default_extract_model"] = st.selectbox(
            "Extraction Model",
            options=MODEL_REGISTRY[st.session_state["default_extract_provider"]]["models"],
            index=0, key="ext_mod"
        )

# -------------------------
# Main Tabs
# -------------------------
tabs = st.tabs(["🏠 Overview", "📄 Documents", "📝 Summary", "🔍 Extract", "🤖 Agents", "📊 Dashboard"])

with tabs[0]:
    st.title("🧭 Agentic Document Processing System")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        wow_badge("Multi-Model AI", "🧠")
    with col2:
        wow_badge("31 Agent Analysis", "🤖")
    with col3:
        wow_badge("Vision OCR", "👁️")
    with col4:
        wow_badge("20 Themes", "🎨")
    
    st.markdown("---")
    st.markdown("""
    ### 🚀 Advanced Document Intelligence Platform
    
    Upload regulatory/quality documents, perform AI-powered OCR, generate executive summaries,
    extract structured data, and run 31 specialized agents for comprehensive text analysis and data mining.
    
    **Features:**
    - 🎨 20 beautiful animal-themed UIs
    - 🤖 31 specialized analysis agents
    - 🧠 Multi-provider AI (Gemini, OpenAI, Grok)
    - 👁️ Advanced Vision OCR
    - 📊 Real-time analytics dashboard
    """)
    
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        kpi_card("Documents", f"{sum(1 for d in st.session_state['documents'] if d['content'].strip())}/5")
    with col2:
        kpi_card("Pages OCR'd", str(sum(d['pages_ocrd'] for d in st.session_state['documents'])))
    with col3:
        kpi_card("Agents Ready", str(len([a for a in st.session_state['agents_config'] if a.get('enabled', True)])))
    with col4:
        kpi_card("Tokens Used", f"{st.session_state['token_usage']['input_tokens'] + st.session_state['token_usage']['output_tokens']:,}")

with tabs[1]:
    st.header("📄 Document Management")
    st.info("💡 Paste text directly or upload .txt/.md/.pdf files. For PDFs, use AI-powered OCR.")
    
    for i, doc in enumerate(st.session_state["documents"]):
        with st.expander(f"📄 {doc['title']}", expanded=(i == 0)):
            col1, col2 = st.columns([3, 2])
            
            with col1:
                new_title = st.text_input("Document Title", doc["title"], key=f"title_{i}")
                st.session_state["documents"][i]["title"] = new_title
                txt = st.text_area("Content", doc["content"], height=200, key=f"doc_text_{i}")
                st.session_state["documents"][i]["content"] = txt
                
            with col2:
                uploaded = st.file_uploader("Upload File", type=["txt", "md", "pdf"], key=f"upload_{i}")
                if uploaded:
                    st.session_state["documents"][i]["file_name"] = uploaded.name
                    
                    if uploaded.type in ("text/plain", "text/markdown"):
                        content = uploaded.read().decode("utf-8", errors="ignore")
                        st.session_state["documents"][i]["content"] = content
                        st.success(f"✅ Loaded {uploaded.name}")
                        
                    elif uploaded.type == "application/pdf":
                        if not PDF_AVAILABLE:
                            st.error("❌ PDF support not available. Install pdf2image.")
                        else:
                            pdf_bytes = uploaded.read()
                            page_num = st.number_input("Page to OCR", 1, 9999, 1, 1, key=f"page_{i}")
                            
                            if st.button("🔍 OCR This Page", key=f"ocr_btn_{i}"):
                                if not provider_ready(st.session_state["ocr_provider"]):
                                    st.error(f"❌ {MODEL_REGISTRY[st.session_state['ocr_provider']]['label']} API key required")
                                else:
                                    with st.spinner("🔄 Processing..."):
                                        try:
                                            img = pdf_page_to_image(pdf_bytes, page_num)
                                            text = ai_ocr_image(img, st.session_state["ocr_provider"], st.session_state["ocr_model"])
                                            st.session_state["documents"][i]["content"] = text
                                            st.session_state["documents"][i]["pages_ocrd"] += 1
                                            st.success(f"✅ OCR completed! Extracted {len(text)} characters")
                                        except Exception as e:
                                            st.error(f"❌ OCR failed: {e}")
                
                if doc["pages_ocrd"] > 0:
                    st.metric("Pages OCR'd", doc["pages_ocrd"])

with tabs[2]:
    st.header("📝 Generate Executive Summary")
    
    concat_docs = "\n\n".join(
        f"# {d['title']}\n{d['content']}" for d in st.session_state["documents"] if d["content"].strip()
    )
    
    if not concat_docs.strip():
        st.warning("⚠️ Please add content to at least one document first.")
    else:
        with st.expander("📖 Preview Combined Documents", expanded=False):
            preview = concat_docs[:5000] + ("..." if len(concat_docs) > 5000 else "")
            st.text_area("Combined Content", preview, height=300, disabled=True)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("✨ Generate Summary", use_container_width=True):
                if not provider_ready(st.session_state["default_summary_provider"]):
                    st.error(f"❌ {MODEL_REGISTRY[st.session_state['default_summary_provider']]['label']} API key required")
                else:
                    with st.spinner("🔄 AI is analyzing your documents..."):
                        sys_p = "You are an expert document analyst specializing in regulatory and quality systems."
                        usr_p = f"""Analyze these documents and provide:

1. **Top 5 Key Insights** (bullet points)
2. **Executive Summary** (≤200 words)
3. **Key Terms** (important terminology)

Documents:
{concat_docs}"""
                        
                        out, meta = call_llm(
                            st.session_state["default_summary_provider"],
                            st.session_state["default_summary_model"],
                            sys_p, usr_p
                        )
                        st.session_state["summary"] = out
                        st.success(f"✅ Summary generated in {meta.get('duration_sec', '?')}s")
                        st.rerun()
        
        with col2:
            st.metric("Provider", st.session_state["default_summary_provider"].title())
            st.metric("Model", st.session_state["default_summary_model"])
        
        if st.session_state["summary"]:
            st.markdown("---")
            st.markdown("### 📋 Generated Summary")
            st.markdown(st.session_state["summary"])
            
            if st.button("📋 Copy Summary"):
                st.code(st.session_state["summary"], language=None)

with tabs[3]:
    st.header("🔍 Structured Data Extraction")
    
    doc_titles = [d["title"] for d in st.session_state["documents"] if d["content"].strip()]
    
    if not doc_titles:
        st.warning("⚠️ Please add content to at least one document first.")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_doc = st.selectbox("📄 Select Document", options=doc_titles)
            field_list = st.text_area(
                "📝 Fields to Extract (comma-separated)",
                "Document Number, Effective Date, Revision, Approver, Department",
                height=100
            )
        
        with col2:
            st.metric("Provider", st.session_state["default_extract_provider"].title())
            st.metric("Model", st.session_state["default_extract_model"])
        
        if st.button("🔍 Extract Data", use_container_width=True):
            if not provider_ready(st.session_state["default_extract_provider"]):
                st.error(f"❌ {MODEL_REGISTRY[st.session_state['default_extract_provider']]['label']} API key required")
            else:
                with st.spinner("🔄 Extracting structured data..."):
                    doc_content = next(d["content"] for d in st.session_state["documents"] if d["title"] == selected_doc)
                    fields = [f.strip() for f in field_list.split(",") if f.strip()]
                    
                    sys_p = "You are a precise data extraction specialist. Extract requested fields and return valid JSON only."
                    usr_p = f"""Extract these fields from the document. If a field is not found, use null.

Fields: {', '.join(fields)}

Document:
{doc_content}

Return ONLY valid JSON with the extracted fields."""
                    
                    schema = {
                        "type": "object",
                        "properties": {f: {"type": ["string", "null"]} for f in fields}
                    }
                    
                    out, meta = call_llm(
                        st.session_state["default_extract_provider"],
                        st.session_state["default_extract_model"],
                        sys_p, usr_p, json_schema=schema
                    )
                    
                    try:
                        # Clean potential markdown formatting
                        clean_out = out.strip()
                        if clean_out.startswith("```"):
                            clean_out = clean_out.split("```")[1]
                            if clean_out.startswith("json"):
                                clean_out = clean_out[4:]
                        clean_out = clean_out.strip()
                        
                        data = json.loads(clean_out)
                        st.session_state["extracted"] = data
                        st.success(f"✅ Extraction completed in {meta.get('duration_sec', '?')}s")
                        st.rerun()
                    except json.JSONDecodeError as e:
                        st.error(f"❌ Invalid JSON response: {e}")
                        st.code(out)
        
        if st.session_state["extracted"]:
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📊 Extracted Data (JSON)")
                st.json(st.session_state["extracted"])
            
            with col2:
                st.markdown("### 📋 Data Table")
                df = pd.DataFrame([st.session_state["extracted"]])
                st.dataframe(df, use_container_width=True)
            
            st.download_button(
                "💾 Download as JSON",
                data=json.dumps(st.session_state["extracted"], indent=2, ensure_ascii=False),
                file_name="extracted_data.json",
                mime="application/json"
            )

with tabs[4]:
    st.header("🤖 Agent Orchestration")
    
    if not st.session_state["summary"] or not st.session_state["extracted"]:
        st.warning("⚠️ Please complete Summary and Data Extraction steps first.")
    else:
        st.info(f"📋 {len(st.session_state['agents_config'])} agents loaded from agents.yaml")
        
        # Agent configuration
        enabled_count = sum(1 for a in st.session_state["agents_config"] if a.get("enabled", True))
        st.metric("Enabled Agents", f"{enabled_count}/{len(st.session_state['agents_config'])}")
        
        with st.expander("⚙️ Configure Agents", expanded=False):
            for idx, agent in enumerate(st.session_state["agents_config"]):
                with st.container():
                    col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
                    
                    with col1:
                        st.markdown(f"**{agent.get('name', agent['id'])}**")
                        st.caption(agent.get('description', 'No description'))
                    
                    with col2:
                        providers = [p for p, m in MODEL_REGISTRY.items() if m.get("available", False)]
                        if providers:
                            current_prov = agent.get("provider", providers[0])
                            if current_prov not in providers:
                                current_prov = providers[0]
                            agent["provider"] = st.selectbox(
                                "Provider",
                                options=providers,
                                index=providers.index(current_prov),
                                key=f"prov_{agent['id']}",
                                label_visibility="collapsed"
                            )
                    
                    with col3:
                        models = MODEL_REGISTRY[agent["provider"]]["models"]
                        current_model = agent.get("model", models[0])
                        if current_model not in models:
                            current_model = models[0]
                        agent["model"] = st.selectbox(
                            "Model",
                            options=models,
                            index=models.index(current_model),
                            key=f"model_{agent['id']}",
                            label_visibility="collapsed"
                        )
                    
                    with col4:
                        agent["enabled"] = st.checkbox(
                            "Enable",
                            value=agent.get("enabled", True),
                            key=f"enable_{agent['id']}",
                            label_visibility="collapsed"
                        )
                    
                    st.markdown("---")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            if st.button("🚀 Execute All Agents", use_container_width=True):
                enabled_agents = [a for a in st.session_state["agents_config"] if a.get("enabled", True)]
                
                if not enabled_agents:
                    st.error("❌ No agents enabled")
                else:
                    # Check if required providers are ready
                    missing_providers = []
                    for agent in enabled_agents:
                        if not provider_ready(agent["provider"]):
                            missing_providers.append(MODEL_REGISTRY[agent["provider"]]["label"])
                    
                    if missing_providers:
                        st.error(f"❌ Missing API keys: {', '.join(set(missing_providers))}")
                    else:
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        sum_text = st.session_state["summary"]
                        extracted_json = json.dumps(st.session_state["extracted"], ensure_ascii=False)
                        doc_titles_joined = ", ".join([d["title"] for d in st.session_state["documents"] if d["content"].strip()])
                        
                        st.session_state["agent_results"] = {}
                        st.session_state["agent_dashboard_rows"].clear()
                        
                        def run_agent(agent_obj):
                            sys_p = "You are a specialized analysis agent. Provide detailed, actionable insights."
                            user_p = agent_obj["prompt"].format(
                                summary=sum_text,
                                extracted_json=extracted_json,
                                document_titles=doc_titles_joined,
                                agent_findings=json.dumps(st.session_state["agent_results"], ensure_ascii=False)
                            )
                            out, meta = call_llm(agent_obj["provider"], agent_obj["model"], sys_p, user_p)
                            return agent_obj["id"], agent_obj.get("name", agent_obj["id"]), out, meta
                        
                        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                            futures = [executor.submit(run_agent, a) for a in enabled_agents]
                            
                            for idx, future in enumerate(concurrent.futures.as_completed(futures)):
                                try:
                                    aid, aname, out, meta = future.result()
                                    st.session_state["agent_results"][aid] = {"output": out, "meta": meta, "name": aname}
                                    
                                    st.session_state["agent_dashboard_rows"].append({
                                        "timestamp": datetime.utcnow().isoformat(),
                                        "agent": aname,
                                        "provider": meta.get("provider"),
                                        "model": meta.get("model"),
                                        "duration_s": meta.get("duration_sec"),
                                        "status": "✅" if "error" not in meta else "❌"
                                    })
                                    
                                    progress = (idx + 1) / len(enabled_agents)
                                    progress_bar.progress(progress)
                                    status_text.text(f"Completed: {aname} ({idx + 1}/{len(enabled_agents)})")
                                except Exception as e:
                                    st.error(f"❌ Agent execution failed: {e}")
                        
                        status_text.text(f"✅ All {len(enabled_agents)} agents completed!")
                        st.success("🎉 Agent execution completed successfully!")
                        st.rerun()
        
        with col2:
            if st.button("🔄 Reset Results", use_container_width=True):
                st.session_state["agent_results"] = {}
                st.session_state["agent_dashboard_rows"].clear()
                st.rerun()
        
        with col3:
            st.metric("Completed", len(st.session_state["agent_results"]))
        
        # Display results
        if st.session_state["agent_results"]:
            st.markdown("---")
            st.markdown("### 📊 Agent Results")
            
            for aid, payload in st.session_state["agent_results"].items():
                with st.expander(f"🤖 {payload.get('name', aid)}", expanded=False):
                    meta = payload.get("meta", {})
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Provider", meta.get("provider", "N/A").title())
                    with col2:
                        st.metric("Model", meta.get("model", "N/A"))
                    with col3:
                        st.metric("Duration", f"{meta.get('duration_sec', 0):.2f}s")
                    
                    st.markdown("#### Output")
                    st.markdown(payload["output"])
                    
                    if st.button(f"📋 Copy Output", key=f"copy_{aid}"):
                        st.code(payload["output"], language=None)

with tabs[5]:
    st.header("📊 Analytics Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        kpi_card("Total Agents", str(len(st.session_state["agents_config"])))
    with col2:
        kpi_card("Enabled", str(sum(1 for a in st.session_state["agents_config"] if a.get("enabled", True))))
    with col3:
        kpi_card("Completed", str(len(st.session_state["agent_results"])))
    with col4:
        total_time = sum(v.get("meta", {}).get("duration_sec", 0) for v in st.session_state["agent_results"].values())
        kpi_card("Total Time", f"{total_time:.1f}s")
    
    st.markdown("---")
    
    # Token usage
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🔢 Token Usage")
        token_data = pd.DataFrame([
            {"Type": "Input", "Tokens": st.session_state["token_usage"]["input_tokens"]},
            {"Type": "Output", "Tokens": st.session_state["token_usage"]["output_tokens"]}
        ])
        st.dataframe(token_data, use_container_width=True)
    
    with col2:
        st.markdown("### 📈 Agent Performance")
        if st.session_state["agent_dashboard_rows"]:
            perf_df = pd.DataFrame(st.session_state["agent_dashboard_rows"])
            st.dataframe(perf_df, use_container_width=True)
        else:
            st.info("No agent runs yet")
    
    # Timeline visualization
    if st.session_state["agent_dashboard_rows"]:
        st.markdown("---")
        st.markdown("### ⏱️ Execution Timeline")
        timeline_df = pd.DataFrame(st.session_state["agent_dashboard_rows"])
        st.bar_chart(timeline_df.set_index("agent")["duration_s"])
    
    # Consolidated insights
    if st.button("💡 Generate Consolidated Insights", use_container_width=True):
        if not st.session_state["agent_results"]:
            st.warning("⚠️ No agent results to consolidate")
        elif not provider_ready(st.session_state["default_summary_provider"]):
            st.error(f"❌ {MODEL_REGISTRY[st.session_state['default_summary_provider']]['label']} API key required")
        else:
            with st.spinner("🔄 Generating insights..."):
                joined = "\n\n".join(
                    f"**{v.get('name', k)}**:\n{v['output']}" 
                    for k, v in st.session_state["agent_results"].items()
                )
                
                sys_p = "You are an expert analyst. Synthesize insights and generate actionable follow-up questions."
                usr_p = f"""Based on the summary and agent analyses, provide:

1. **Key Consolidated Findings** (3-5 main points)
2. **Critical Issues** (prioritized list)
3. **3 Strategic Follow-up Questions** (incisive, actionable)

Summary:
{st.session_state['summary']}

Agent Results:
{joined}"""
                
                out, meta = call_llm(
                    st.session_state["default_summary_provider"],
                    st.session_state["default_summary_model"],
                    sys_p, usr_p
                )
                
                st.markdown("---")
                st.markdown("### 💡 Consolidated Insights")
                st.markdown(out)
                st.success(f"✅ Generated in {meta.get('duration_sec', '?')}s")
    
    # Export options
    st.markdown("---")
    st.markdown("### 💾 Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📥 Export Summary", use_container_width=True):
            if st.session_state["summary"]:
                st.download_button(
                    "Download Summary",
                    data=st.session_state["summary"],
                    file_name="summary.md",
                    mime="text/markdown"
                )
    
    with col2:
        if st.button("📥 Export Extracted Data", use_container_width=True):
            if st.session_state["extracted"]:
                st.download_button(
                    "Download JSON",
                    data=json.dumps(st.session_state["extracted"], indent=2, ensure_ascii=False),
                    file_name="extracted_data.json",
                    mime="application/json"
                )
    
    with col3:
        if st.button("📥 Export Agent Results", use_container_width=True):
            if st.session_state["agent_results"]:
                export_data = {
                    "summary": st.session_state["summary"],
                    "extracted": st.session_state["extracted"],
                    "agent_results": {
                        k: {"name": v.get("name"), "output": v["output"], "meta": v.get("meta")}
                        for k, v in st.session_state["agent_results"].items()
                    },
                    "metrics": {
                        "token_usage": st.session_state["token_usage"],
                        "execution_timeline": st.session_state["agent_dashboard_rows"]
                    }
                }
                st.download_button(
                    "Download Complete Report",
                    data=json.dumps(export_data, indent=2, ensure_ascii=False),
                    file_name="complete_analysis.json",
                    mime="application/json"
                )

# Footer
st.markdown("---")
st.markdown(
    f"""
    <div style="text-align: center; color: #666; padding: 1rem;">
        🧭 Agentic Document Processing System | Theme: {st.session_state['theme']} | 
        Powered by Gemini, OpenAI & Grok
    </div>
    """,
    unsafe_allow_html=True
)
