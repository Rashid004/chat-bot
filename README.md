# JPT — AI Chat Assistant with Web Search

JPT is a conversational AI assistant with a dark, ChatGPT-style web interface. It remembers every conversation, streams answers token by token, and can search the web, do math and check the time. When it searches, it shows image galleries and source links next to its answer.

It is built with **LangGraph** (agent workflow), **Groq** (LLM inference), **Tavily** (web search) and **Streamlit** (UI).

## Features

- **Streaming replies.** Answers appear token by token, with a typing indicator while the model starts up.
- **Tool use.** The LLM decides on its own when to call a tool:
  - `web_search` searches the web through Tavily. It supports a `news` or `general` topic and can optionally fetch images.
  - `calculator` does add, subtract, multiply and divide, and handles division by zero.
  - `get_current_datetime` returns the current local date and time.
- **Live status pill.** While a tool runs, the UI shows what is happening: "Searching the web", "Reading sources" or "Writing the answer".
- **Image gallery.** Image URLs from search results are checked in parallel, and only the ones that load are shown (up to 4). Broken or blocked images never reach the UI.
- **Source chips.** Each answer that used search links to its sources (up to 5), shown as small numbered chips.
- **Persistent conversations.** Every chat is saved to SQLite through a LangGraph checkpointer. The sidebar lists past chats, titled by their first message, and you can reopen any of them.
- **Reply cleanup.** `【...】` citation markers and odd Unicode spaces (for example narrow no-break spaces that glue words together) are stripped from model output, both while streaming and when loading saved chats.
- **Observability.** LangSmith tracing is supported through environment variables, with the `thread_id` attached as metadata.
- **Custom theme.** Dark violet theme with the Inter font, set in `.streamlit/config.toml` and `streamlit_frontend.py`. It respects `prefers-reduced-motion`.

## How it works

The backend is a small LangGraph agent loop:

```
START ──► chat_node ──(tool call?)──► tools ──┐
              ▲                               │
              └───────────────────────────────┘
              │
              └──(no tool call)──► END
```

1. `chat_node` sends the system prompt and the full message history to the LLM (`openai/gpt-oss-120b` on Groq, with the tools bound).
2. If the model asks for a tool, the `tools` node runs it and sends the result back to `chat_node`. This repeats until the model answers in plain text.
3. After every step, `SqliteSaver` writes the state to `chatbot.db`, keyed by `thread_id`. This is what makes conversations persistent.

`web_search` returns two things (LangChain's `content_and_artifact` format):

- **Content:** a compact, text-only summary of up to 4 results. Each snippet is capped at 350 characters to keep token use low on Groq's free tier. This is all the LLM sees.
- **Artifact:** the images and source links, passed straight to the UI. The LLM never sees them, so it can't mangle URLs or write broken markdown images.

## Project structure

| File | Purpose |
|---|---|
| [chatbot_backend.py](chatbot_backend.py) | LangGraph agent: tools, LLM setup, system prompt, graph, SQLite checkpointer, `retrieve_all_threads()` |
| [streamlit_frontend.py](streamlit_frontend.py) | **Main UI.** Streaming, tool status pill, image gallery, source chips, sidebar chat history, suggestion cards |
| [streamlit_frontend_streming.py](streamlit_frontend_streming.py) | Earlier, minimal UI with streaming only (no tool UI or styling). Kept as a simple reference |
| [.streamlit/config.toml](.streamlit/config.toml) | Streamlit theme (colors, font, radius) |
| `chatbot.db` | SQLite database holding the saved conversations (created automatically, git-ignored) |

## Getting started

### Prerequisites

- Python 3.10 or newer
- A [Groq](https://console.groq.com) API key (LLM)
- A [Tavily](https://tavily.com) API key (web search)
- Optional: a [LangSmith](https://smith.langchain.com) API key for tracing

### Install

```bash
git clone <your-repo-url>
cd chat-boat

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install streamlit python-dotenv requests \
    langchain-core langchain-groq langchain-tavily \
    langgraph langgraph-checkpoint-sqlite
```

### Configure

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_key
TAVILY_API_KEY=your_tavily_key

# Optional: LangSmith tracing
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_PROJECT=your_project_name
```

`.env` is git-ignored. Never commit your keys.

### Run

```bash
streamlit run streamlit_frontend.py
```

Then open the URL Streamlit prints, usually http://localhost:8501.

To try the minimal version instead, run `streamlit run streamlit_frontend_streming.py`.

## Usage examples

| You ask | What happens |
|---|---|
| "Write a Python function to reverse a string" | Answered directly, no tool |
| "What's the latest news on SpaceX?" | `web_search` with `topic='news'`, then an answer with source chips |
| "Show me photos of the Eiffel Tower" | `web_search` with `with_images=True`, then a gallery and sources |
| "What is 1234 × 56?" | `calculator` |
| "What day is it today?" | `get_current_datetime` |

## Customizing

- **Change the model.** Edit the `ChatGroq(model=...)` line in [chatbot_backend.py](chatbot_backend.py). Any Groq model that supports tool calling works. To use another provider, swap in its LangChain chat class.
- **Add a tool.** Write a function with type hints and a docstring, decorate it with `@tool`, and add it to the `tools` list. The docstring tells the LLM when to use it. Add a friendly label in `TOOL_LABELS` in the frontend to show it in the status pill.
- **Change the persona.** Edit `SYSTEM_PROMPT` in the backend. It is added on every call and is not stored in the database.
- **Restyle the UI.** The color tokens are at the top of `CUSTOM_CSS` in [streamlit_frontend.py](streamlit_frontend.py), and match [.streamlit/config.toml](.streamlit/config.toml). Restart Streamlit after changing the theme file.

## Tech stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Agent orchestration | LangGraph, LangChain |
| LLM | `openai/gpt-oss-120b` via Groq (`langchain-groq`) |
| Web search | Tavily (`langchain-tavily`) |
| Persistence | SQLite with LangGraph `SqliteSaver` |
| Tracing (optional) | LangSmith |

## Notes and limitations

- Conversations live in a local SQLite file, so this is a single-user, local setup. It has no authentication.
- The Groq free tier has rate limits, which is why search snippets are kept short.
- Image checks add up to about 3 seconds of latency when images are requested.
