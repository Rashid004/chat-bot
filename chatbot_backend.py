from __future__ import annotations
import os
import tempfile

from datetime import datetime
from concurrent.futures import ThreadPoolExecutor  
from typing import Annotated, Any, Dict, Optional, TypedDict,Literal,Tuple
from urllib.parse import urlparse  
import sqlite3

import requests

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_tavily import TavilySearch
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()


# ************************* Step 1: tools *************************
# @tool function ko LLM ke liye "tool" bana deta hai.
# Docstring + type hints hi LLM ko batate hain ki tool kab aur kaise use karna hai.

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """Do basic arithmetic on two numbers. operation must be one of: add, sub, mul, div."""
    if operation == 'add':
        result = first_num + second_num
    elif operation == 'sub':
        result = first_num - second_num
    elif operation == 'mul':
        result = first_num * second_num
    elif operation == 'div':
        if second_num == 0:
            return {'error': 'Division by zero is not allowed'}
        result = first_num / second_num
    else:
        return {'error': f'Unsupported operation: {operation}'}
    return {'first_num': first_num, 'second_num': second_num, 'operation': operation, 'result': result}


@tool
def get_current_datetime() -> str:
    """Get the current local date and time. Use this for any question about today, now, or the time."""
    return datetime.now().strftime('%A, %d %B %Y, %I:%M %p')

# TavilySearch ke ~10 optional args se gpt-oss confuse hota hai (galat tool call bhejta hai).
# Isliye chhota wrapper: LLM ko sirf 3 args dikhte hain.
_tavily = TavilySearch(max_results=4)

MAX_IMAGES = 4
SNIPPET_CHARS = 350  # LLM ko chhota snippet do: kam tokens = Groq free tier ki limit safe


def _is_image(url):
    """Image URL sach mein khulta hai? (broken/blocked images UI se bahar rakhne ke liye)"""
    try:
        with requests.get(url, stream=True, timeout=3, headers={'User-Agent': 'Mozilla/5.0'}) as r:
            return r.ok and r.headers.get('content-type', '').startswith('image/')
    except requests.RequestException:
        return False


def _working_images(raw_images):
    """Tavily ke image urls mein se sirf chalne wale (max MAX_IMAGES), same order mein."""
    urls = []
    for item in raw_images or []:
        url = item.get('url') if isinstance(item, dict) else item
        if isinstance(url, str) and url.startswith('http') and url not in urls:
            urls.append(url)
    urls = urls[:MAX_IMAGES * 2]  # kuch broken honge, isliye thode extra check karo
    with ThreadPoolExecutor(max_workers=8) as pool:
        ok = list(pool.map(_is_image, urls))
    return [u for u, good in zip(urls, ok) if good][:MAX_IMAGES]


@tool(response_format='content_and_artifact')
def web_search(query: str, topic: Literal['general', 'news'] = 'general', with_images: bool = False) -> Tuple[str, dict]:
    """Search the web for current information. Use topic='news' for news and current events,
    topic='general' for everything else (facts, prices, sports, weather).
    Set with_images=True only when the user asks for pictures/photos or the subject is
    visual (a place, person, product, animal, landmark)."""
    try:
        raw = _tavily.invoke({'query': query, 'topic': topic, 'include_images': with_images})
    except Exception as error:
        return f'Search failed: {error}', {'images': [], 'sources': []}

    results = raw.get('results', []) if isinstance(raw, dict) else []
    if not results:
        return f"No results found. {raw.get('error', '') if isinstance(raw, dict) else ''}".strip(), {'images': [], 'sources': []}

    # content -> LLM ko jata hai (text only); artifact -> sirf UI ke liye (LLM ko nahi dikhta)
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r.get('title', '')} ({urlparse(r.get('url', '')).netloc})\n{r.get('content', '')[:SNIPPET_CHARS]}")
    artifact = {
        'images': _working_images(raw.get('images')) if with_images else [],
        'sources': [{'title': r.get('title', ''), 'url': r.get('url', '')} for r in results],
    }
    return '\n\n'.join(lines), artifact


# ************************* RAG: javascript.pdf *************************
# Groq embeddings nahi deta, isliye local free model (pehli baar ~130MB download, phir offline)
embeddings = FastEmbedEmbeddings()
PDF_PATH = 'javascript.pdf'

# PDF retriever store (per thread)
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """Build a FAISS retriever for the uploaded PDF and store it for the thread.

    Returns a summary dict that can be surfaced in the UI.
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
    finally:
        # The FAISS store keeps copies of the text, so the temp file is safe to remove.
        try:
            os.remove(temp_path)
        except OSError:
            pass


INDEX_DIR = 'faiss_index'


def _default_retriever():
    """Book ka retriever: pehle disk se, nahi to PDF embed karke disk par save (sirf pehli baar slow)."""
    if 'default' not in _THREAD_RETRIEVERS:
        if os.path.isdir(INDEX_DIR):
            store = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)  # file hamari apni hai
            _THREAD_RETRIEVERS['default'] = store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        else:
            with open(PDF_PATH, 'rb') as f:
                ingest_pdf(f.read(), thread_id='default', filename=PDF_PATH)
            _THREAD_RETRIEVERS['default'].vectorstore.save_local(INDEX_DIR)
    return _THREAD_RETRIEVERS['default']


@tool
def search_pdf(query: str) -> str:
    """Search the JavaScript book (javascript.pdf). Call this FIRST for any JavaScript question."""
    docs = _default_retriever().invoke(query)
    return '\n\n'.join(f"[page {d.metadata.get('page', 0) + 1}]\n{d.page_content}" for d in docs)


tools = [calculator, get_current_datetime, web_search, search_pdf]


# ************************* Step 2: LLM ko tools batao *************************
# OpenAI key revoked hai, abhi Groq use kar rahe hain (GROQ_API_KEY .env se aata hai)
llm = ChatGroq(model='openai/gpt-oss-120b')
# bind_tools: LLM ko tools ki list (naam + docstring + args) bhejta hai.
# LLM khud tool nahi chalata, sirf bolta hai "ye tool in args ke saath chalao".
llm_with_tools = llm.bind_tools(tools)


# ************************* Step 3: graph *************************

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# system prompt har call par judta hai, par state/DB mein save nahi hota
SYSTEM_PROMPT = SystemMessage(content=(
    "You are JPT, a helpful assistant. "
    "Use web_search when the answer may depend on recent or changing information "
    "(news, current events, prices, sports, weather, recent facts) or when you are unsure. "
    "For 'today', 'latest', 'breaking' or 'current' questions use topic='news'. "
    "Set with_images=True when the user asks for pictures/photos or the subject is visual. "
    "Images and source links are shown to the user automatically, so never write image URLs "
    "or markdown images yourself. "
    "For any JavaScript question call search_pdf first and answer from its text, mentioning the page; "
    "only if the PDF has nothing relevant, answer normally or use web_search. "
    "Answer directly, without searching, for general knowledge, coding, or casual chat. "
    "Never say you lack real-time access; search instead. "
    "Name the source in plain words (e.g. 'according to Reuters'); never use 【】 citation markers."
))


def chat_node(state: ChatState):
    """LLM ko poori history bhejo; jawab ya to text hoga ya tool call."""
    response = llm_with_tools.invoke([SYSTEM_PROMPT] + state['messages'])
    return {'messages': [response]}


# ToolNode: LLM ne jo tool calls maange, unhe chala ke ToolMessage return karta hai
tool_node = ToolNode(tools)

connection = sqlite3.connect(database='chatbot.db', check_same_thread=False)

# Check Pointer
checkpointer = SqliteSaver(conn=connection)

graph = StateGraph(ChatState)

graph.add_node('chat_node', chat_node)
graph.add_node('tools', tool_node)

graph.add_edge(START, 'chat_node')
# tools_condition: last message mein tool call hai to 'tools' par jao, warna END
graph.add_conditional_edges('chat_node', tools_condition)
# tool ka result wapas LLM ko, taaki wo final jawab likh sake (yahi loop hai)
graph.add_edge('tools', 'chat_node')

chatbot = graph.compile(checkpointer=checkpointer)


def retrieve_all_threads():
    """DB mein saved saare thread ids, purane se naye order mein."""
    all_threads = []
    # list() naye checkpoints pehle deta hai; set() order kho deta, isliye list
    for checkpoint in checkpointer.list(None):
        thread_id = checkpoint.config['configurable']['thread_id']
        if thread_id not in all_threads:
            all_threads.append(thread_id)

    # frontend [::-1] karta hai, isliye yahan purane pehle rakho
    return all_threads[::-1]
