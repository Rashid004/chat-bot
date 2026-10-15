import base64
import html
import importlib.util
import os
import re
import sys
from pathlib import Path

import streamlit as st

# ************************* agent load *************************
# file ke naam me "-" hai, isliye normal `import` nahi chalega -> importlib se load karte hain.
# Import par agent kuch run nahi karta (run sirf __main__ me hai), to yahan koi token kharch nahi hota.
_spec = importlib.util.spec_from_file_location("blog_agent", Path(__file__).parent / "ai-blog-writer-agent.py")
agent = importlib.util.module_from_spec(_spec)
sys.modules["blog_agent"] = agent  # LangGraph/pydantic type hints resolve karne ke liye module registered hona chahiye
_spec.loader.exec_module(agent)

MAX_RUNS_PER_SESSION = 5      # ek browser session me max itni baar generate (galti se paise na jayein)
MAX_TOPIC_CHARS = 200

EXAMPLES = [
    "Self Attention in Transformer Architecture",
    "RAG vs Fine-tuning: when to use which",
    "Designing a rate limiter for an API",
    "Latest open-source LLM releases this month",
]

STEP_LABELS = {
    "router": "Deciding if web research is needed",
    "research": "Searching the web",
    "orchestrator": "Planning the outline",
    "worker": "Writing sections",
    "reducer": "Merging the final blog",
}

st.set_page_config(page_title="AI Blog Writer", page_icon=":material/edit_note:", layout="centered")

ICON_PEN = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" '
    'stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    '<path d="M12 20h9"/><path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4Z"/></svg>'
)

# Design system: Dark Mode (OLED) + IBM Plex Sans / JetBrains Mono + green accent (ui-ux-pro-max)
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap');
:root {
    --bg: #0F172A; --card: #1B2336; --muted-bg: #272F42; --border: #334155;
    --text: #F8FAFC; --muted: #94A3B8; --accent: #22C55E; --accent-soft: #4ADE80;
    --on-accent: #0F172A; --mono: 'JetBrains Mono', ui-monospace, monospace;
}
[data-testid="stAppViewContainer"] { background: var(--bg); }
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stMainBlockContainer"] { padding: 2.5rem 1rem 3rem; max-width: 52rem; }
code, pre, [data-testid="stCode"] * { font-family: var(--mono) !important; }

/* ---------- hero ---------- */
.hero { margin: 0 0 1.75rem; }
.hero .badge {
    display: inline-flex; align-items: center; gap: .5rem; padding: .35rem .75rem; border-radius: 999px;
    background: rgba(34,197,94,.12); border: 1px solid rgba(34,197,94,.35); color: var(--accent-soft);
    font: 500 .78rem var(--mono); margin-bottom: .9rem;
}
.hero h1 { font-size: clamp(1.8rem, 5vw, 2.5rem); font-weight: 700; line-height: 1.15; margin: 0 0 .5rem; color: var(--text); }
.hero h1 em { font-style: normal; color: var(--accent); }
.hero p { color: var(--muted); margin: 0; font-size: 1rem; line-height: 1.6; max-width: 40rem; }

/* ---------- controls (>=44px touch targets, visible focus) ---------- */
.stButton > button, .stDownloadButton > button {
    min-height: 44px; cursor: pointer; transition: background-color .18s ease, border-color .18s ease, transform .18s ease;
}
.stButton > button[kind="secondary"] {
    background: var(--card); border: 1px solid var(--border); color: var(--text); font-size: .9rem; text-align: left;
}
.stButton > button[kind="secondary"]:hover { border-color: var(--accent); background: var(--muted-bg); }
.stButton > button[kind="primary"], .stDownloadButton > button {
    background: var(--accent); color: var(--on-accent); border: 0; font-weight: 600;
}
.stButton > button[kind="primary"]:hover, .stDownloadButton > button:hover { background: var(--accent-soft); color: var(--on-accent); }
.stButton > button:disabled { opacity: .45; cursor: not-allowed; }
button:focus-visible, textarea:focus-visible, [role="tab"]:focus-visible, a:focus-visible {
    outline: 2px solid var(--accent-soft) !important; outline-offset: 2px;
}
textarea { font-size: 1rem !important; line-height: 1.5 !important; }
.field-hint { color: var(--muted); font-size: .8rem; margin: -.25rem 0 .75rem; }

/* ---------- step indicator ---------- */
.steps { display: flex; gap: .5rem; flex-wrap: wrap; margin: .25rem 0 .75rem; padding: 0; list-style: none; }
.steps li {
    display: inline-flex; align-items: center; gap: .4rem; padding: .3rem .7rem; border-radius: 999px;
    font: 500 .78rem var(--mono); color: var(--muted); background: var(--card); border: 1px solid var(--border);
}
.steps li.active { color: var(--on-accent); background: var(--accent); border-color: var(--accent); }
.steps li.done { color: var(--accent-soft); border-color: rgba(34,197,94,.45); }
.steps li.skip { opacity: .5; text-decoration: line-through; }

/* ---------- stats ---------- */
.stat-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: .75rem; margin: 1.25rem 0; }
.stat { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: .75rem .9rem; }
.stat b { display: block; font: 600 1.25rem var(--mono); color: var(--text); }
.stat span { font-size: .72rem; color: var(--muted); text-transform: uppercase; letter-spacing: .06em; }

/* ---------- sources ---------- */
.src {
    display: block; padding: .65rem .9rem; margin-bottom: .5rem; border-radius: 10px; min-height: 44px;
    background: var(--card); border: 1px solid var(--border); text-decoration: none !important; transition: border-color .18s ease;
}
.src:hover { border-color: var(--accent); }
.src b { color: var(--text); font-weight: 600; font-size: .9rem; }
.src small { display: block; color: var(--muted); font-family: var(--mono); overflow-wrap: anywhere; }

@media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    f'<div class="hero"><span class="badge">{ICON_PEN}gpt-4.1-mini · LangGraph</span>'
    "<h1>Write technical blogs with <em>AI</em></h1>"
    "<p>Give a topic. The agent researches, plans the outline and writes every section in parallel, "
    "then hands you clean Markdown.</p></div>",
    unsafe_allow_html=True,
)

# ************************* sidebar: cost controls *************************
st.session_state.setdefault("runs", 0)
st.session_state.setdefault("result", None)

with st.sidebar:
    st.subheader("Settings")
    allow_research = st.toggle("Web research (Tavily)", value=True, help="Off = no search, no router call. Cheapest.")
    max_sections = st.slider("Sections", 3, agent.MAX_SECTIONS, 5, help="Each section is one OpenAI call.")
    enable_images = st.toggle(
        "AI images (Gemini)", value=False,
        help="Off by default. On = one extra big OpenAI call plus 3 Gemini image calls.",
    )
    st.divider()
    left = MAX_RUNS_PER_SESSION - st.session_state.runs
    st.caption(f"Runs left in this session: **{left}** / {MAX_RUNS_PER_SESSION}")
    st.caption(
        f"Token guards: max {max_sections} sections, {agent.SECTION_MAX_TOKENS} tokens per section, "
        f"{agent.MAX_QUERIES} searches, 1 retry."
    )

# ************************* input *************************
if "topic" not in st.session_state:
    st.session_state.topic = ""

def _use_example(text):
    st.session_state.topic = text

topic = st.text_area(
    "Blog topic", key="topic", max_chars=MAX_TOPIC_CHARS, height=100,
    placeholder="e.g. How does KV caching speed up LLM inference?",
)
st.markdown('<p class="field-hint">Or try one of these:</p>', unsafe_allow_html=True)

cols = st.columns(2)
for i, ex in enumerate(EXAMPLES):
    cols[i % 2].button(ex, key=f"ex{i}", on_click=_use_example, args=(ex,), use_container_width=True)

generate = st.button(
    "Generate blog", type="primary", use_container_width=True,
    disabled=not topic.strip() or left <= 0,
)
if left <= 0:
    st.warning("Session limit reached. Refresh the page to reset it.")

# ************************* run *************************
PIPELINE = [("router", "Route"), ("research", "Research"), ("orchestrator", "Plan"), ("worker", "Write"), ("reducer", "Merge")]

def _steps_html(done, active, skipped=()):
    items = []
    for key, label in PIPELINE:
        cls = "skip" if key in skipped else "done" if key in done else "active" if key == active else ""
        items.append(f'<li class="{cls}">{label}</li>')
    return f'<ol class="steps" aria-label="Progress" aria-busy="true">{"".join(items)}</ol>'

def _inline_images(md, saved_path):
    """Markdown me images/xyz.png relative path hota hai; Streamlit ko dikhane ke liye base64 me badal do."""
    base = Path(saved_path).parent

    def repl(m):
        p = base / m.group(2)
        if not p.is_file():
            return m.group(0)
        data = base64.b64encode(p.read_bytes()).decode()
        return f"![{m.group(1)}](data:image/png;base64,{data})"

    return re.sub(r"!\[([^\]]*)\]\((images/[^)]+)\)", repl, md)

if generate:
    if not os.environ.get("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY not found. Add it to the `.env` file in the project root.")
        st.stop()
    if allow_research and not os.environ.get("TAVILY_API_KEY"):
        st.warning("TAVILY_API_KEY missing: web research will return nothing.")

    st.session_state.runs += 1  # fail ho ya pass, count hoga (safe side)
    state = agent.initial_state(topic, allow_research, enable_images, max_sections)
    result = {"plan": None, "evidence": [], "mode": "", "final": "", "saved_path": ""}
    written = 0

    with st.status("Starting...", expanded=True) as status:
        steps_box = st.empty()
        steps_box.markdown(_steps_html(set(), "router"), unsafe_allow_html=True)
        bar = st.progress(0.0)
        done = set()
        try:
            for chunk in agent.app.stream(state, config={"recursion_limit": 25}, stream_mode="updates"):
                if not isinstance(chunk, dict):
                    continue
                for node, upd in chunk.items():
                    upd = upd or {}
                    status.update(label=STEP_LABELS.get(node, node))
                    if node != "worker":
                        done.add(node)
                    if node == "router":
                        result["mode"] = upd.get("mode", "")
                        st.write(f"Mode: **{result['mode'] or 'closed_book'}**")
                    elif node == "research":
                        result["evidence"] = upd.get("evidence", [])
                        st.write(f"Found **{len(result['evidence'])}** sources")
                    elif node == "orchestrator":
                        result["plan"] = upd.get("plan")
                        st.write(f"Planned **{len(result['plan'].tasks)}** sections: {result['plan'].blog_title}")
                    elif node == "worker":
                        written += 1
                        total = len(result["plan"].tasks) if result["plan"] else max_sections
                        if written >= total:
                            done.add("worker")
                        bar.progress(min(written / total, 0.95))
                        st.write(f"Section {written}/{total} written")
                    elif node == "reducer":
                        result["final"] = upd.get("final", "")
                        result["saved_path"] = upd.get("saved_path", "")
                    skipped = ("research",) if "router" in done and result["mode"] == "closed_book" else ()
                    nxt = next((k for k, _ in PIPELINE if k not in done and k not in skipped), None)
                    steps_box.markdown(_steps_html(done, nxt, skipped), unsafe_allow_html=True)
            bar.progress(1.0)
            steps_box.empty()
            status.update(label="Done", state="complete", expanded=False)
            st.session_state.result = result
        except Exception as e:
            status.update(label="Failed", state="error")
            st.error(f"Generation failed: {e}")

# ************************* output *************************
res = st.session_state.result
if res and res["final"]:
    md = res["final"]
    words = len(md.split())
    plan = res["plan"]

    st.markdown(
        f'<div class="stat-row">'
        f'<div class="stat"><b>{words}</b><span>Words</span></div>'
        f'<div class="stat"><b>{len(plan.tasks) if plan else 0}</b><span>Sections</span></div>'
        f'<div class="stat"><b>{len(res["evidence"])}</b><span>Sources</span></div>'
        f'<div class="stat"><b>{html.escape(res["mode"] or "closed_book")}</b><span>Mode</span></div>'
        f"</div>",
        unsafe_allow_html=True,
    )

    tab_preview, tab_md, tab_plan = st.tabs(["Preview", "Markdown", "Plan & sources"])

    with tab_preview:
        with st.container(border=True):
            st.markdown(_inline_images(md, res["saved_path"]) if res["saved_path"] else md)

    with tab_md:
        st.code(md, language="markdown")

    with tab_plan:
        if plan:
            st.caption(f"Audience: {plan.audience} · Tone: {plan.tone} · Kind: {plan.blog_kind}")
            for t in plan.tasks:
                with st.expander(f"{t.id}. {t.title} ({t.target_words} words)"):
                    st.write(t.goal)
                    for b in t.bullets:
                        st.markdown(f"- {b}")
        if res["evidence"]:
            st.markdown("**Sources**")
            for e in res["evidence"]:
                # URL web se aata hai: escape karke hi HTML me daalte hain
                st.markdown(
                    f'<a class="src" href="{html.escape(e.url)}" target="_blank" rel="noopener noreferrer">'
                    f"<b>{html.escape(e.title or e.url)}</b><small>{html.escape(e.url)}</small></a>",
                    unsafe_allow_html=True,
                )

    c1, c2 = st.columns([1, 2])
    c1.download_button(
        "Download .md", md, file_name=Path(res["saved_path"]).name or "blog.md",
        mime="text/markdown", use_container_width=True,
    )
    if res["saved_path"]:
        c2.caption(f"Saved at `{res['saved_path']}`")
