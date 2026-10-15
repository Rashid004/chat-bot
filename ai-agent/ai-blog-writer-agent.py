from __future__ import annotations

import operator
import os
import re
from pathlib import Path
from typing import TypedDict, List, Optional, Literal, Annotated

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_tavily import TavilySearch

load_dotenv()  # root ki .env se OPENAI_API_KEY / TAVILY_API_KEY / GEMINI_API_KEY uthata hai

# -----------------------------
# 0) Token / cost limits (yahin se sab kuch control hota hai)
# -----------------------------
MAX_SECTIONS = 6          # plan me kitne sections (har section = 1 LLM call)
MAX_QUERIES = 3           # Tavily searches per run
RESULTS_PER_QUERY = 4     # Tavily results per search
SNIPPET_CHARS = 200       # evidence snippet ki max length
PLAN_MAX_TOKENS = 2000    # router/planner output cap
SECTION_MAX_TOKENS = 1000 # ek section ka output cap (~550 words ~ 800 tokens)
IMAGE_PLAN_MAX_TOKENS = 8000  # image step poora blog wapas likhta hai, isliye bada cap
MAX_IMAGES = 3            # Gemini image calls per run
OUT_DIR = Path(__file__).parent / "blogs"

# -----------------------------
# 1) Schemas
# -----------------------------

class Task(BaseModel):
    id: int
    title: str

    goal: str = Field(
        ...,
        description="One sentence describing what the reader should be able to do/understand after this section.",
    )
    bullets: List[str] = Field(
        ...,
        min_length=3,
        max_length=6,
        description="3–6 concrete, non-overlapping subpoints to cover in this section.",
    )
    target_words: int = Field(..., description="Target word count for this section (120–550).")

    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False

class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal["explainer", "tutorial", "news_roundup", "comparison", "system_design"] = "explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]

class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None  # keep if Tavily provides; DO NOT rely on it
    snippet: Optional[str] = None
    source: Optional[str] = None

class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    queries: List[str] = Field(default_factory=list)

class ImageSpec(BaseModel):
    placeholder: str = Field(..., description="e.g. [[IMAGE_1]]")
    filename: str = Field(..., description="Save under images/, e.g. qkv_flow.png")
    alt: str
    caption: str
    prompt: str = Field(..., description="Prompt to send to the image model.")
    size: Literal["1024x1024", "1024x1536", "1536x1024"] = "1024x1024"
    quality: Literal["low", "medium", "high"] = "medium"


class GlobalImagePlan(BaseModel):
    md_with_placeholders: str
    images: List[ImageSpec] = Field(default_factory=list)

class State(TypedDict):
    topic: str

    # UI / user ke switches (cost control)
    allow_research: bool
    enable_images: bool
    max_sections: int

    # routing / research
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]

    # workers
    sections: Annotated[List[tuple[int, str]], operator.add]  # (task_id,section_md)

    # reducer/image
    merged_md: str
    md_with_placeholders: str
    image_specs: List[dict]

    final: str
    saved_path: str

# -----------------------------
# 2) LLM
# -----------------------------

# max_retries=1 + timeout: fail hone par bar-bar retry karke tokens waste nahi honge
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.3, max_tokens=PLAN_MAX_TOKENS, max_retries=1, timeout=60)
writer_llm = llm.bind(max_tokens=SECTION_MAX_TOKENS)

# -----------------------------
# 3) Router (decide upfront)
# -----------------------------

ROUTER_SYSTEM = """You are a routing module for a technical blog planner.

Decide whether web research is needed BEFORE planning.

Modes:
- closed_book (needs_research=false):
  Evergreen topics where correctness does not depend on recent facts (concepts, fundamentals).
- hybrid (needs_research=true):
  Mostly evergreen but needs up-to-date examples/tools/models to be useful.
- open_book (needs_research=true):
  Mostly volatile: weekly roundups, "this week", "latest", rankings, pricing, policy/regulation.

If needs_research=true:
- Output 3 high-signal queries (never more than 3).
- Queries should be scoped and specific (avoid generic queries like just "AI" or "LLM").
- If user asked for "last week/this week/latest", reflect that constraint IN THE QUERIES.
"""

def router_node(state: State) -> dict:
    # user ne research band kiya hai -> LLM call bhi bachao
    if not state.get("allow_research", True):
        return {"needs_research": False, "mode": "closed_book", "queries": []}

    decider = llm.with_structured_output(RouterDecision)
    decision = decider.invoke(
        [
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(content=f"Topic: {state['topic']}"),
        ]
    )

    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": decision.queries[:MAX_QUERIES],
    }

def route_next(state: State) -> str:
    return "research" if state["needs_research"] else "orchestrator"


# -----------------------------
# 4) Research (Tavily)
# -----------------------------

def _tavily_search(query: str) -> List[EvidenceItem]:
    tool = TavilySearch(max_results=RESULTS_PER_QUERY)
    res = tool.invoke({"query": query})
    items = res.get("results", []) if isinstance(res, dict) else []

    return [
        EvidenceItem(
            title=r.get("title") or "",
            url=r["url"],
            snippet=(r.get("content") or "")[:SNIPPET_CHARS],
            published_at=r.get("published_date"),
        )
        for r in items
        if r.get("url")
    ]

def research_node(state: State) -> dict:
    dedup: dict[str, EvidenceItem] = {}  # url -> item (duplicate URLs hata deta hai)

    for q in (state.get("queries") or [])[:MAX_QUERIES]:
        try:
            for e in _tavily_search(q):
                dedup.setdefault(e.url, e)
        except Exception:
            continue  # ek query fail ho to baaki chalti rahein

    return {"evidence": list(dedup.values())}


# -----------------------------
# 5) Orchestrator (Plan)
# -----------------------------

ORCH_SYSTEM = """You are a senior technical writer and developer advocate.
Your job is to produce a highly actionable outline for a technical blog post.

Hard requirements:
- Create 5–6 sections (tasks) suitable for the topic and audience. Never more than 6.
- Each task must include:
  1) goal (1 sentence)
  2) 3–6 bullets that are concrete, specific, and non-overlapping
  3) target word count (120–450)

Quality bar:
- Assume the reader is a developer; use correct terminology.
- Bullets must be actionable: build/compare/measure/verify/debug.
- Ensure the overall plan includes at least 2 of these somewhere:
  * minimal code sketch / MWE (set requires_code=True for that section)
  * edge cases / failure modes
  * performance/cost considerations
  * security/privacy considerations (if relevant)
  * debugging/observability tips

Grounding rules:
- Mode closed_book: keep it evergreen; do not depend on evidence.
- Mode hybrid:
  - Use evidence for up-to-date examples (models/tools/releases) in bullets.
  - Mark sections using fresh info as requires_research=True and requires_citations=True.
- Mode open_book:
  - Set blog_kind = "news_roundup".
  - Every section is about summarizing events + implications.
  - DO NOT include tutorial/how-to sections unless user explicitly asked for that.
  - If evidence is empty or insufficient, create a plan that transparently says "insufficient sources"
    and includes only what can be supported.

Output must strictly match the Plan schema.
"""

def orchestrator_node(state: State) -> dict:
    planner = llm.with_structured_output(Plan)

    evidence = state.get("evidence", [])
    mode = state.get("mode", "closed_book")

    plan = planner.invoke(
        [
            SystemMessage(content=ORCH_SYSTEM),
            HumanMessage(
                content=(
                    f"Topic: {state['topic']}\n"
                    f"Mode: {mode}\n\n"
                    f"Evidence (ONLY use for fresh claims; may be empty):\n"
                    f"{[e.model_dump() for e in evidence][:12]}"
                )
            ),
        ]
    )

    # LLM ne zyada sections diye to bhi hard cap: har section = 1 paid call
    plan.tasks = plan.tasks[: min(state.get("max_sections", MAX_SECTIONS), MAX_SECTIONS)]
    return {"plan": plan}


# -----------------------------
# 6) Fanout
# -----------------------------
def fanout(state: State):
    return [
        Send(
            "worker",
            {
                "task": task.model_dump(),
                "topic": state["topic"],
                "mode": state["mode"],
                "plan": state["plan"].model_dump(),
                "evidence": [e.model_dump() for e in state.get("evidence", [])],
            },
        )
        for task in state["plan"].tasks
    ]


# -----------------------------
# 7) Worker (write one section)
# -----------------------------
WORKER_SYSTEM = """You are a senior technical writer and developer advocate.
Write ONE section of a technical blog post in Markdown.

Hard constraints:
- Follow the provided Goal and cover ALL Bullets in order (do not skip or merge bullets).
- Stay close to Target words (±15%).
- Output ONLY the section content in Markdown (no blog title H1, no extra commentary).
- Start with a '## <Section Title>' heading.

Scope guard:
- If blog_kind == "news_roundup": do NOT turn this into a tutorial/how-to guide.
  Do NOT teach web scraping, RSS, automation, or "how to fetch news" unless bullets explicitly ask for it.
  Focus on summarizing events and implications.

Grounding policy:
- If mode == open_book:
  - Do NOT introduce any specific event/company/model/funding/policy claim unless it is supported by provided Evidence URLs.
  - For each event claim, attach a source as a Markdown link: ([Source](URL)).
  - Only use URLs provided in Evidence. If not supported, write: "Not found in provided sources."
- If requires_citations == true:
  - For outside-world claims, cite Evidence URLs the same way.
- Evergreen reasoning is OK without citations unless requires_citations is true.

Code:
- If requires_code == true, include at least one minimal, correct code snippet relevant to the bullets.

Style:
- Short paragraphs, bullets where helpful, code fences for code.
- Avoid fluff/marketing. Be precise and implementation-oriented.
"""

def worker_node(payload: dict) -> dict:

    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])
    evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]
    topic = payload["topic"]
    mode = payload.get("mode", "closed_book")

    bullets_text = "\n- " + "\n- ".join(task.bullets)

    evidence_text = ""
    if evidence:
        evidence_text = "\n".join(
            f"- {e.title} | {e.url} | {e.published_at or 'date:unknown'} | {e.snippet or ''}".strip()
            for e in evidence[:12]
        )

    section_md = writer_llm.invoke(
        [
            SystemMessage(content=WORKER_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog title: {plan.blog_title}\n"
                    f"Audience: {plan.audience}\n"
                    f"Tone: {plan.tone}\n"
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Constraints: {plan.constraints}\n"
                    f"Topic: {topic}\n"
                    f"Mode: {mode}\n\n"
                    f"Section title: {task.title}\n"
                    f"Goal: {task.goal}\n"
                    f"Target words: {task.target_words}\n"
                    f"Tags: {task.tags}\n"
                    f"requires_research: {task.requires_research}\n"
                    f"requires_citations: {task.requires_citations}\n"
                    f"requires_code: {task.requires_code}\n"
                    f"Bullets:{bullets_text}\n\n"
                    f"Evidence (ONLY use these URLs when citing):\n{evidence_text}\n"
                )
            ),
        ]
    ).content.strip()

    return {"sections": [(task.id, section_md)]}


# ============================================================
# 8) ReducerWithImages (subgraph)
#    merge_content -> decide_images -> generate_and_place_images
# ============================================================
def merge_content(state: State) -> dict:

    plan = state["plan"]

    ordered_sections = [md for _, md in sorted(state["sections"], key=lambda x: x[0])]
    body = "\n\n".join(ordered_sections).strip()
    merged_md = f"# {plan.blog_title}\n\n{body}\n"
    return {"merged_md": merged_md}

DECIDE_IMAGES_SYSTEM = """You are an expert technical editor.
Decide if images/diagrams are needed for THIS blog.

Rules:
- Create exactly 3 images (a blog about a visual/technical topic always benefits from 3 diagrams).
- Each image must materially improve understanding (diagram/flow/table-like visual).
- Insert placeholders exactly: [[IMAGE_1]], [[IMAGE_2]], [[IMAGE_3]] (each on its own line, right after the section it explains).
- Avoid decorative images; prefer technical diagrams with short labels.
Return strictly GlobalImagePlan.
"""

def decide_images(state: State) -> dict:
    merged_md = state["merged_md"]

    # images off (default) -> ye step poora skip: isme poora blog dobara LLM se likhwana padta hai (sabse mehenga step)
    if not state.get("enable_images", False):
        return {"md_with_placeholders": merged_md, "image_specs": []}

    planner = llm.bind(max_tokens=IMAGE_PLAN_MAX_TOKENS).with_structured_output(GlobalImagePlan)
    plan = state["plan"]
    assert plan is not None

    image_plan = planner.invoke(
        [
            SystemMessage(content=DECIDE_IMAGES_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Topic: {state['topic']}\n\n"
                    "Insert placeholders + propose image prompts.\n\n"
                    f"{merged_md}"
                )
            ),
        ]
    )

    return {
        "md_with_placeholders": image_plan.md_with_placeholders,
        "image_specs": [img.model_dump() for img in image_plan.images[:MAX_IMAGES]],
    }

def _gemini_generate_image_bytes(prompt: str) -> bytes:
    """
    Returns raw image bytes generated by Gemini.
    Requires: pip install google-genai
    Env var: GEMINI_API_KEY (ya GOOGLE_API_KEY)
    """
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    client = genai.Client(api_key=api_key)

    resp = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            safety_settings=[
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="BLOCK_ONLY_HIGH",
                )
            ],
        ),
    )

    # Depending on SDK version, parts may hang off resp.candidates[0].content.parts
    parts = getattr(resp, "parts", None)
    if not parts and getattr(resp, "candidates", None):
        try:
            parts = resp.candidates[0].content.parts
        except Exception:
            parts = None

    if not parts:
        raise RuntimeError("No image content returned (safety/quota/SDK change).")

    for part in parts:
        inline = getattr(part, "inline_data", None)
        if inline and getattr(inline, "data", None):
            return inline.data

    raise RuntimeError("No inline image bytes found in response.")

def _save_markdown(plan: Plan, md: str) -> str:
    # title me "/" ya ":" ho sakta hai, isliye safe filename banate hain
    slug = re.sub(r"[^\w\-]+", "_", plan.blog_title).strip("_")[:80] or "blog"
    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / f"{slug}.md"
    path.write_text(md, encoding="utf-8")
    return str(path)

def generate_and_place_images(state: State) -> dict:

    plan = state["plan"]
    assert plan is not None

    md = state.get("md_with_placeholders") or state["merged_md"]
    image_specs = (state.get("image_specs", []) or [])[:MAX_IMAGES]

    images_dir = OUT_DIR / "images"
    if image_specs:
        images_dir.mkdir(parents=True, exist_ok=True)

    for spec in image_specs:
        placeholder = spec["placeholder"]
        filename = Path(spec["filename"]).name  # LLM ka diya path trust nahi karte (../ se bachne ke liye)
        out_path = images_dir / filename

        # generate only if needed
        if not out_path.exists():
            try:
                img_bytes = _gemini_generate_image_bytes(spec["prompt"])
                out_path.write_bytes(img_bytes)
            except Exception as e:
                # graceful fallback: keep doc usable
                prompt_block = (
                    f"> **[IMAGE GENERATION FAILED]** {spec.get('caption','')}\n>\n"
                    f"> **Alt:** {spec.get('alt','')}\n>\n"
                    f"> **Prompt:** {spec.get('prompt','')}\n>\n"
                    f"> **Error:** {e}\n"
                )
                md = md.replace(placeholder, prompt_block)
                continue

        img_md = f"![{spec['alt']}](images/{filename})\n*{spec['caption']}*"
        md = md.replace(placeholder, img_md)

    # cap ki wajah se jo placeholders bach gaye unhe hata do
    md = re.sub(r"\[\[IMAGE_\d+\]\]", "", md)

    return {"final": md, "saved_path": _save_markdown(plan, md)}

# build reducer subgraph
reducer_graph = StateGraph(State)
reducer_graph.add_node("merge_content", merge_content)
reducer_graph.add_node("decide_images", decide_images)
reducer_graph.add_node("generate_and_place_images", generate_and_place_images)
reducer_graph.add_edge(START, "merge_content")
reducer_graph.add_edge("merge_content", "decide_images")
reducer_graph.add_edge("decide_images", "generate_and_place_images")
reducer_graph.add_edge("generate_and_place_images", END)
reducer_subgraph = reducer_graph.compile()


# -----------------------------
# 9) Build main graph
# -----------------------------
g = StateGraph(State)
g.add_node("router", router_node)
g.add_node("research", research_node)
g.add_node("orchestrator", orchestrator_node)
g.add_node("worker", worker_node)
g.add_node("reducer", reducer_subgraph)

g.add_edge(START, "router")
g.add_conditional_edges("router", route_next, {"research": "research", "orchestrator": "orchestrator"})
g.add_edge("research", "orchestrator")

g.add_conditional_edges("orchestrator", fanout, ["worker"])
g.add_edge("worker", "reducer")
g.add_edge("reducer", END)

app = g.compile()


# -----------------------------
# 10) Runner
# -----------------------------
def initial_state(
    topic: str,
    allow_research: bool = True,
    enable_images: bool = False,
    max_sections: int = MAX_SECTIONS,
) -> dict:
    return {
        "topic": topic.strip()[:300],  # bahut lamba topic = faltu input tokens
        "allow_research": allow_research,
        "enable_images": enable_images,
        "max_sections": max_sections,
        "mode": "",
        "needs_research": False,
        "queries": [],
        "evidence": [],
        "plan": None,
        "sections": [],
        "merged_md": "",
        "md_with_placeholders": "",
        "image_specs": [],
        "final": "",
        "saved_path": "",
    }

def run(topic: str, **options):
    return app.invoke(initial_state(topic, **options), config={"recursion_limit": 25})


if __name__ == "__main__":
    # sirf `python ai-blog-writer-agent.py` chalane par run hoga (import par nahi: warna UI khulte hi paise kharch)
    out = run("Self Attention in Transformer Architecture")
    print(f"Saved: {out['saved_path']}")
