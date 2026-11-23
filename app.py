"""
IntelliResearch AI — Multi-Agent Research Assistant
Based on project.ipynb:
  PDF RAG (FAISS + MMR) + Tavily web search + CrewAI agents + local Ollama LLM.

New in this version
  • Upload many PDFs at once, and keep adding more later (incremental index)
  • Knowledge base is saved to disk and reloaded on the next launch
  • Persistent search history (searchable, re-runnable, deletable, exportable)
  • Two modes: full multi-agent report, or the notebook's quick RAG-only answer

Run:  streamlit run app.py
"""

import json
import os
import shutil
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

import ollama
import streamlit as st
from crewai import LLM, Agent, Crew, Process, Task
from crewai.tools import tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tavily import TavilyClient

# ----------------------------------------------------------------------------
# Storage locations (persist across restarts)
# ----------------------------------------------------------------------------
DATA_DIR = Path("data")
INDEX_DIR = DATA_DIR / "faiss_index"
DOCS_FILE = DATA_DIR / "documents.json"
HISTORY_FILE = DATA_DIR / "history.json"
MAX_HISTORY = 200
DATA_DIR.mkdir(exist_ok=True)

VIEW_RESEARCH, VIEW_HISTORY = "🔬 Research", "🕘 Search history"
MODE_FULL, MODE_QUICK = "Full multi-agent report", "Quick answer (RAG only)"

# ----------------------------------------------------------------------------
# Page setup & styling
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="IntelliResearch AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container {padding-top: 2rem; max-width: 1200px;}
        .hero {
            padding: 1.6rem 2rem; border-radius: 14px; margin-bottom: 1.2rem;
            background: linear-gradient(135deg, #1e3a8a 0%, #4f46e5 60%, #7c3aed 100%);
            color: #fff;
        }
        .hero h1 {margin: 0; font-size: 2rem; font-weight: 700; color: #fff;}
        .hero p {margin: .35rem 0 0; opacity: .9; font-size: 1rem;}
        div[data-testid="stMetric"] {
            background: rgba(128,128,128,.08); border: 1px solid rgba(128,128,128,.2);
            padding: .8rem 1rem; border-radius: 10px;
        }
        .stButton > button[kind="primary"] {font-weight: 600; border-radius: 8px;}
        footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Persistence helpers
# ----------------------------------------------------------------------------
def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _write_json(path: Path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def save_history():
    _write_json(HISTORY_FILE, st.session_state.history[:MAX_HISTORY])


def save_docs():
    _write_json(DOCS_FILE, st.session_state.docs)


# ----------------------------------------------------------------------------
# Cached resources
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model…")
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_data(ttl=10, show_spinner=False)
def list_ollama_models(base_url: str):
    """Return (models, error). Cached briefly so reruns stay fast."""
    try:
        client = ollama.Client(host=base_url)
        return [m["model"] for m in client.list()["models"]], None
    except Exception as exc:  # noqa: BLE001
        return [], str(exc)


# ----------------------------------------------------------------------------
# Session state (history + knowledge base are restored from disk)
# ----------------------------------------------------------------------------
def init_state():
    ss = st.session_state
    if "initialized" in ss:
        return
    ss.history = _read_json(HISTORY_FILE, [])
    ss.docs = _read_json(DOCS_FILE, [])
    ss.vectorstore = None
    if ss.docs and (INDEX_DIR / "index.faiss").exists():
        try:
            ss.vectorstore = FAISS.load_local(
                str(INDEX_DIR), get_embeddings(),
                allow_dangerous_deserialization=True,  # index was created by this app
            )
        except Exception:  # noqa: BLE001
            ss.docs = []
    ss.current_id = None
    ss.view = VIEW_RESEARCH
    ss.question = ""
    ss.initialized = True


init_state()
ss = st.session_state


# ----------------------------------------------------------------------------
# Knowledge-base functions (notebook: PyPDFLoader -> splitter -> FAISS -> MMR)
# ----------------------------------------------------------------------------
def add_pdfs(files, chunk_size, chunk_overlap, progress):
    """Index new PDFs and merge them into the existing FAISS store."""
    known = {d["name"] for d in ss.docs}
    new_files = [f for f in files if f.name not in known]
    skipped = [f.name for f in files if f.name in known]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    failed = []

    for i, f in enumerate(new_files, 1):
        progress.progress((i - 1) / max(len(new_files), 1), text=f"Processing {f.name}…")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(f.getbuffer())
            path = tmp.name
        try:
            pages = PyPDFLoader(path).load()
            for p in pages:
                p.metadata["source"] = f.name  # real filename, not the temp path
            chunks = splitter.split_documents(pages)
            if not chunks:
                raise ValueError("no extractable text (scanned PDF?)")
            if ss.vectorstore is None:
                ss.vectorstore = FAISS.from_documents(chunks, get_embeddings())
            else:
                ss.vectorstore.add_documents(chunks)
            ss.docs.append({
                "name": f.name, "pages": len(pages), "chunks": len(chunks),
                "added": datetime.now().strftime("%Y-%m-%d %H:%M"),
            })
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{f.name} ({exc})")
        finally:
            os.remove(path)

    if ss.vectorstore is not None and new_files:
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        ss.vectorstore.save_local(str(INDEX_DIR))
        save_docs()
    progress.progress(1.0, text="Done")
    return len(new_files) - len(failed), skipped, failed


def clear_knowledge_base():
    ss.vectorstore = None
    ss.docs = []
    shutil.rmtree(INDEX_DIR, ignore_errors=True)
    DOCS_FILE.unlink(missing_ok=True)


def make_retriever(k, fetch_k, lambda_mult):
    return ss.vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": max(fetch_k, k), "lambda_mult": lambda_mult},
    )


def snapshot_evidence(docs):
    return [
        {
            "source": os.path.basename(d.metadata.get("source", "Unknown")),
            "page": d.metadata.get("page", 0) + 1,
            "text": d.page_content,
        }
        for d in docs
    ]


# ----------------------------------------------------------------------------
# CrewAI tools / crew (notebook agents & tasks)
# ----------------------------------------------------------------------------
def make_tools(retriever, tavily_client):
    @tool("PDF Knowledge Search")
    def pdf_search(query: str) -> str:
        """Search the uploaded PDFs using MMR retrieval; returns text with source and page."""
        results = retriever.invoke(query)
        if not results:
            return "No relevant information found in the PDF."
        return "\n".join(
            f"Result: {i}\nSource: {os.path.basename(d.metadata.get('source', 'Unknown'))}\n"
            f"Page: {d.metadata.get('page', 0) + 1}\n\nContent:\n{d.page_content}\n{'-' * 32}"
            for i, d in enumerate(results, 1)
        )

    tools = {"pdf": pdf_search, "web": None}

    if tavily_client:

        @tool("Web Research Search")
        def web_search(query: str) -> str:
            """Search the web using Tavily and return findings with source URLs."""
            resp = tavily_client.search(query=query, search_depth="advanced", max_results=5)
            items = resp.get("results", [])
            if not items:
                return "No relevant web information found."
            return "\n".join(
                f"Result: {i}\nTitle: {r['title']}\nURL: {r['url']}\nContent: {r['content']}\n"
                for i, r in enumerate(items, 1)
            )

        tools["web"] = web_search
    return tools


def build_crew(llm, tools, report_words):
    use_web = tools["web"] is not None

    doc_agent = Agent(
        role="Document Research Specialist",
        goal="Retrieve accurate information from PDF documents with page references.",
        backstory="You analyze technical PDFs and never invent document information.",
        llm=llm, tools=[tools["pdf"]], verbose=False, allow_delegation=False,
    )
    analyst = Agent(
        role="Senior Research Analyst",
        goal="Analyze findings, compare sources and identify insights and research gaps.",
        backstory="You maintain source references and avoid unsupported claims.",
        llm=llm, verbose=False, allow_delegation=False,
    )
    writer = Agent(
        role="Professional Research Report Writer",
        goal="Produce a well-structured research report from the collected findings.",
        backstory="You preserve PDF page references and URLs and never invent citations.",
        llm=llm, verbose=False, allow_delegation=False,
    )

    doc_task = Task(
        description=(
            "Answer this research question:\n{question}\n\n"
            "Use the PDF Knowledge Search tool. The knowledge base may contain several "
            "PDFs, so include the source file name and page for every finding. "
            "Do not invent information not present in the documents."
        ),
        expected_output=(
            "Structured answer: introduction, key findings, technical explanations, "
            "PDF source names and correct page references."
        ),
        agent=doc_agent,
    )
    tasks, agents, context = [doc_task], [doc_agent], [doc_task]

    if use_web:
        web_agent = Agent(
            role="Web Research Specialist",
            goal="Find reliable, recent information and preserve source URLs.",
            backstory="You use Tavily to find papers, articles and reports. Never invent facts.",
            llm=llm, tools=[tools["web"]], verbose=False, allow_delegation=False,
        )
        web_task = Task(
            description=(
                "Conduct web research on:\n{question}\n\n"
                "Use the Web Research Search tool, focus on recent technical "
                "developments, avoid duplicates, include titles and URLs."
            ),
            expected_output="Web research summary with findings, source titles and URLs.",
            agent=web_agent,
        )
        agents.insert(1, web_agent)
        tasks.append(web_task)
        context.append(web_task)

    analysis_task = Task(
        description=(
            "Analyze the collected findings for:\n{question}\n\n"
            "Compare sources, highlight similarities/differences and research gaps, "
            "preserve page references and URLs, and keep PDF-based findings clearly "
            "separate from web-based findings."
        ),
        expected_output="Comparative analysis with insights, gaps and source references.",
        agent=analyst,
        context=list(context),
    )
    web_section = "5. Recent Developments from Web Research " if use_web else ""
    report_task = Task(
        description=(
            "Write a research report for:\n{question}\n\n"
            "Sections: 1. Title 2. Executive Summary 3. Introduction "
            f"4. Key Findings from PDF {web_section}"
            "6. Comparative Analysis 7. Conclusion 8. References\n\n"
            f"Keep it around {report_words} words, use clear academic language, "
            "preserve page numbers and URLs, use only information from previous "
            "tasks and never invent facts or references."
        ),
        expected_output="A complete, concise report with all sections and accurate references.",
        agent=writer,
        context=[*context, analysis_task],
    )

    tasks += [analysis_task, report_task]
    agents += [analyst, writer]
    return Crew(agents=agents, tasks=tasks, process=Process.sequential, verbose=False)


def quick_answer(llm_cfg, question, evidence):
    """Notebook's direct RAG prompt (no agents) — fast, grounded in the PDFs."""
    context = "".join(
        f"\nSource: {e['source']} | Page: {e['page']}\n{e['text']}\n" for e in evidence
    )
    prompt = f"""
You are an academic research assistant.

Answer the question using ONLY the provided PDF context.

Instructions:
1. Use clear and simple academic language.
2. Do not add information from outside the context.
3. Organize the answer using bullet points.
4. Mention the PDF file name and page number for each relevant point.
5. If the context does not contain enough information, say so.

PDF CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
    model = ChatOllama(model=llm_cfg["model"], base_url=llm_cfg["base_url"],
                       temperature=llm_cfg["temperature"])
    return model.invoke(prompt).content


# ----------------------------------------------------------------------------
# History actions (used as button callbacks)
# ----------------------------------------------------------------------------
def open_entry(entry_id):
    ss.current_id = entry_id
    ss.view = VIEW_RESEARCH


def rerun_question(entry_id):
    entry = next((e for e in ss.history if e["id"] == entry_id), None)
    if entry:
        ss.question = entry["question"]
        ss.current_id = None
        ss.view = VIEW_RESEARCH


def delete_entry(entry_id):
    ss.history = [e for e in ss.history if e["id"] != entry_id]
    if ss.current_id == entry_id:
        ss.current_id = None
    save_history()


def clear_history():
    ss.history = []
    ss.current_id = None
    save_history()


def get_entry(entry_id):
    return next((e for e in ss.history if e["id"] == entry_id), None)


def render_entry(entry, prefix):
    """Show a stored research result (report, agent steps, evidence)."""
    badge = "PDF + Web" if entry.get("web") else "PDF only"
    mode = "Quick answer" if entry.get("mode") == "quick" else "Full report"
    st.caption(f"{entry['time']} · {mode} · {badge} · {entry['model']}")
    if entry.get("docs"):
        st.caption("Documents: " + " · ".join(f"`{d}`" for d in entry["docs"]))

    tabs = ["📄 Report", "🔎 Evidence"] + (["🧩 Agent outputs"] if entry.get("steps") else [])
    objs = st.tabs(tabs)

    with objs[0]:
        st.markdown(entry["report"])
        d1, d2, _ = st.columns([1, 1, 4])
        d1.download_button("⬇️ Markdown", entry["report"], "IntelliResearch_Report.md",
                           "text/markdown", key=f"{prefix}_md")
        d2.download_button("⬇️ Text", entry["report"], "IntelliResearch_Report.txt",
                           "text/plain", key=f"{prefix}_txt")
    with objs[1]:
        st.caption("Passages retrieved from your documents for this question.")
        for i, e in enumerate(entry.get("evidence", []), 1):
            with st.expander(f"Passage {i} · {e['source']} · page {e['page']}"):
                st.write(e["text"])
    if entry.get("steps"):
        with objs[2]:
            for i, s in enumerate(entry["steps"], 1):
                with st.expander(f"Step {i} · {s['agent']}"):
                    st.markdown(s["output"])


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    with st.expander("🤖 Language model", expanded=True):
        base_url = st.text_input("Ollama URL", "http://localhost:11434")
        models, err = list_ollama_models(base_url)
        if err:
            st.error("Ollama not reachable. Is `ollama serve` running?")
            model_name = st.text_input("Model", "gemma4:latest")
        else:
            st.success(f"Connected · {len(models)} model(s)")
            idx = models.index("gemma4:latest") if "gemma4:latest" in models else 0
            model_name = st.selectbox("Model", models, index=idx)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)

    with st.expander("🌐 Web research (Tavily)", expanded=False):
        default_key = os.getenv("TAVILY_API_KEY", "")
        try:
            default_key = st.secrets.get("TAVILY_API_KEY", default_key)
        except Exception:  # noqa: BLE001
            pass
        tavily_key = st.text_input("Tavily API key", value=default_key, type="password")
        use_web = st.toggle("Enable web research", value=bool(tavily_key))

    with st.expander("📚 Retrieval settings"):
        st.caption("Chunk settings apply to PDFs indexed from now on.")
        chunk_size = st.slider("Chunk size", 300, 2000, 800, 50)
        chunk_overlap = st.slider("Chunk overlap", 0, 400, 150, 10)
        k = st.slider("Chunks returned (k)", 2, 12, 4)
        fetch_k = st.slider("MMR candidates (fetch_k)", 4, 40, 12)
        lambda_mult = st.slider("MMR diversity (λ)", 0.0, 1.0, 0.7, 0.05,
                                help="Lower = more diverse, higher = more relevant")

    with st.expander("📝 Report"):
        report_words = st.select_slider("Target length (words)", [250, 400, 500, 800, 1200], 500)

    st.divider()
    st.markdown("### 🕘 Recent searches")
    if ss.history:
        for e in ss.history[:6]:
            st.button(f"↩ {e['question'][:42]}{'…' if len(e['question']) > 42 else ''}",
                      key=f"side_{e['id']}", on_click=open_entry, args=(e["id"],),
                      use_container_width=True)
    else:
        st.caption("Your searches will appear here and are saved between sessions.")

# ----------------------------------------------------------------------------
# Header & status
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🔬 IntelliResearch AI</h1>
        <p>Multi-agent research assistant — grounded in your PDFs, enriched with live web research,
        powered by a local LLM.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Documents", len(ss.docs))
c2.metric("Pages", sum(d["pages"] for d in ss.docs))
c3.metric("Chunks indexed", sum(d["chunks"] for d in ss.docs))
c4.metric("Searches saved", len(ss.history))

st.radio("View", [VIEW_RESEARCH, VIEW_HISTORY], key="view", horizontal=True,
         label_visibility="collapsed")

# ============================================================================
# VIEW: Research
# ============================================================================
if ss.view == VIEW_RESEARCH:

    # ---- Step 1: knowledge base -------------------------------------------
    st.subheader("1 · Knowledge base")
    uploader_key = f"uploader_{len(ss.docs)}"  # resets after each successful indexing
    uploaded = st.file_uploader(
        "Upload one or more PDFs (select several at once, or add more any time)",
        type="pdf", accept_multiple_files=True, key=uploader_key,
    )

    b1, b2, _ = st.columns([1.3, 1.3, 3])
    if b1.button("📥 Index documents", disabled=not uploaded, type="primary"):
        if chunk_overlap >= chunk_size:
            st.error("Chunk overlap must be smaller than chunk size.")
        else:
            bar = st.progress(0.0, text="Starting…")
            added, skipped, failed = add_pdfs(uploaded, chunk_size, chunk_overlap, bar)
            if added:
                st.toast(f"Indexed {added} new document(s)", icon="✅")
            if skipped:
                st.warning("Already indexed, skipped: " + ", ".join(skipped))
            if failed:
                st.error("Could not index: " + "; ".join(failed))
            if added and not failed and not skipped:
                st.rerun()
    if b2.button("🧹 Clear knowledge base", disabled=not ss.docs):
        clear_knowledge_base()
        st.rerun()

    if ss.docs:
        with st.expander(f"📚 Indexed documents ({len(ss.docs)})", expanded=False):
            st.dataframe(
                [{"File": d["name"], "Pages": d["pages"], "Chunks": d["chunks"], "Added": d["added"]}
                 for d in ss.docs],
                use_container_width=True, hide_index=True,
            )
        st.caption("The index is saved in `data/` and will be reloaded next time you open the app.")

    # ---- Step 2: question ---------------------------------------------------
    st.subheader("2 · Ask a research question")
    mode = st.radio("Mode", [MODE_FULL, MODE_QUICK], horizontal=True,
                    help="Full: 4 agents + optional web search (slower). "
                         "Quick: one grounded answer from your PDFs (fast).")
    question = st.text_area(
        "Research question", key="question", height=110, label_visibility="collapsed",
        placeholder="e.g. What are the structural limitations of existing RIS technologies "
                    "and what recent developments may address them?",
    )

    if ss.vectorstore is None:
        st.info("Upload and index at least one PDF to begin.")

    if st.button("🚀 Run research", type="primary",
                 disabled=ss.vectorstore is None or not question.strip()):
        q = question.strip()
        retriever = make_retriever(k, fetch_k, lambda_mult)
        evidence = snapshot_evidence(retriever.invoke(q))

        tavily_client = None
        if use_web and mode == MODE_FULL:
            if tavily_key:
                tavily_client = TavilyClient(api_key=tavily_key)
            else:
                st.warning("Web research is enabled but no Tavily key was given — running PDF-only.")

        try:
            steps = []
            if mode == MODE_QUICK:
                with st.spinner("Generating grounded answer…"):
                    report = quick_answer(
                        {"model": model_name, "base_url": base_url, "temperature": temperature},
                        q, evidence,
                    )
            else:
                llm = LLM(model=f"ollama/{model_name}", base_url=base_url, temperature=temperature)
                crew = build_crew(llm, make_tools(retriever, tavily_client), report_words)
                with st.status("Research crew is working…", expanded=True) as status:
                    st.write("📄 Document agent retrieving evidence from your PDFs")
                    if tavily_client:
                        st.write("🌐 Web agent searching recent sources")
                    st.write("🧠 Analyst comparing findings")
                    st.write("✍️ Writer drafting the final report")
                    result = crew.kickoff(inputs={"question": q})
                    status.update(label="Research complete", state="complete", expanded=False)
                report = str(result)
                steps = [
                    {"agent": str(getattr(t, "agent", "Agent")), "output": getattr(t, "raw", str(t))}
                    for t in getattr(result, "tasks_output", [])
                ]

            entry = {
                "id": uuid.uuid4().hex,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "question": q,
                "mode": "quick" if mode == MODE_QUICK else "full",
                "web": bool(tavily_client),
                "model": model_name,
                "docs": [d["name"] for d in ss.docs],
                "report": report,
                "steps": steps,
                "evidence": evidence,
            }
            ss.history.insert(0, entry)
            ss.history = ss.history[:MAX_HISTORY]
            ss.current_id = entry["id"]
            save_history()
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.error(f"The research run failed: {exc}")

    # ---- Step 3: results ----------------------------------------------------
    current = get_entry(ss.current_id) if ss.current_id else None
    if current:
        st.divider()
        st.subheader("3 · Results")
        st.markdown(f"**Q:** {current['question']}")
        render_entry(current, prefix=f"cur_{current['id']}")

# ============================================================================
# VIEW: Search history
# ============================================================================
else:
    st.subheader("Search history")
    if not ss.history:
        st.info("No searches yet. Run a research question and it will be saved here.")
    else:
        h1, h2, h3 = st.columns([3, 1, 1])
        needle = h1.text_input("Filter history", placeholder="🔍 Search past questions and reports…",
                               label_visibility="collapsed").strip().lower()
        h2.download_button("⬇️ Export all (JSON)",
                           json.dumps(ss.history, ensure_ascii=False, indent=2),
                           "intelliresearch_history.json", "application/json")
        if h3.button("🗑️ Clear history"):
            clear_history()
            st.rerun()

        shown = [e for e in ss.history
                 if not needle or needle in e["question"].lower() or needle in e["report"].lower()]
        st.caption(f"Showing {len(shown)} of {len(ss.history)} saved searches")

        for e in shown:
            tag = "⚡" if e.get("mode") == "quick" else "🧠"
            with st.expander(f"{tag} {e['time']} — {e['question'][:100]}"):
                a, b, c, _ = st.columns([1, 1, 1, 3])
                a.button("📂 Open", key=f"open_{e['id']}", on_click=open_entry, args=(e["id"],))
                b.button("🔁 Re-ask", key=f"re_{e['id']}", on_click=rerun_question, args=(e["id"],))
                c.button("🗑️ Delete", key=f"del_{e['id']}", on_click=delete_entry, args=(e["id"],))
                st.markdown(e["report"])

st.markdown("---")
st.caption("IntelliResearch AI · CrewAI · LangChain · FAISS · Ollama · Tavily")