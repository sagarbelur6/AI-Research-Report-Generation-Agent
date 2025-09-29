# agents.py
import os
import time
import json
from typing import List, Dict, Any, Tuple
from tools import ddg_search, arxiv_search, wikipedia_search, load_local_documents, chunk_documents, embed_texts, semantic_search
from memory import EpisodicMemory, LongTermMemory
from report_generator import save_pdf_report, save_pptx, save_visual
import uuid
import logging
import matplotlib.pyplot as plt

import openai

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# -----------------------------
# Low-level LLM helpers
# -----------------------------
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in environment (.env or env var)")
openai.api_key = OPENAI_API_KEY


def llm_chat(prompt: str, temperature=0.0, max_tokens=800) -> str:
    """Simple wrapper around OpenAI Chat completions for a single-turn completion."""
    resp = openai.ChatCompletion.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message["content"].strip()


# -----------------------------
# Searcher Agent
# -----------------------------
class SearcherAgent:
    """
    Responsible for gathering data from:
     - local documents (PDF/TXT)
     - web (DuckDuckGo)
     - academic (arXiv)
     - knowledge base (Wikipedia)
    Returns a list of raw items (dicts with 'source','title','text','url' optional)
    """

    def __init__(self, materials_dir: str = "materials", max_web=8, max_arxiv=10):
        self.materials_dir = materials_dir
        self.max_web = max_web
        self.max_arxiv = max_arxiv

    def search(self, topic: str) -> List[Dict[str, Any]]:
        results = []
        logger.info("SearcherAgent: searching local materials...")
        local_docs = load_local_documents(self.materials_dir)
        # local_docs: list of dict {'source','text','metadata'}
        for d in local_docs:
            # quick simple relevance filter by substring match; we'll use semantic search later too
            if topic.lower() in d["text"][:5000].lower() or topic.split()[0].lower() in d["text"][:5000].lower():
                results.append({"source": d["source"], "title": d.get("title", d["source"]), "text": d["text"], "url": None, "type": "local"})

        logger.info("SearcherAgent: performing web search...")
        try:
            web_hits = ddg_search(topic, max_results=self.max_web)
            for w in web_hits:
                results.append({"source": "web", "title": w.get("title"), "text": w.get("snippet") or "", "url": w.get("link"), "type": "web"})
        except Exception as e:
            logger.warning("DuckDuckGo search failed: %s", e)

        logger.info("SearcherAgent: performing arXiv search...")
        try:
            arxiv_hits = arxiv_search(topic, max_results=self.max_arxiv)
            for a in arxiv_hits:
                results.append({"source": "arxiv", "title": a.get("title"), "text": a.get("summary"), "url": a.get("url"), "type": "arxiv"})
        except Exception as e:
            logger.warning("arXiv search failed: %s", e)

        logger.info("SearcherAgent: fetching Wikipedia summary...")
        try:
            wiki = wikipedia_search(topic)
            if wiki:
                results.append({"source": "wikipedia", "title": wiki.get("title"), "text": wiki.get("summary"), "url": wiki.get("url"), "type": "wiki"})
        except Exception as e:
            logger.warning("Wikipedia search failed: %s", e)

        logger.info("SearcherAgent: finished; found %d candidate pieces", len(results))
        return results


# -----------------------------
# Summarizer Agent
# -----------------------------
class SummarizerAgent:
    """
    Distills raw content into structured notes. Uses the LLM to summarize each item and produce key points.
    """

    def __init__(self, max_tokens=500):
        self.max_tokens = max_tokens

    def summarize_items(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        summaries = []
        for i, item in enumerate(items):
            title = item.get("title") or f"item_{i}"
            text = (item.get("text") or "")[:12000]  # clamp
            prompt = f"""Summarize the following source in bullet points (3-8 bullets). Provide 1-sentence summary, key findings, and any potential citation formatted as [source: {title} | {item.get('url')}].\n\nSOURCE:\n{text}"""
            try:
                out = llm_chat(prompt, temperature=0.0, max_tokens=self.max_tokens)
            except Exception as e:
                out = f"(LLM failed to summarize: {e})"
            summaries.append({"source": item.get("source"), "title": title, "summary": out, "meta": {"type": item.get("type"), "url": item.get("url")}})
        return summaries


# -----------------------------
# Visualizer Agent
# -----------------------------
class VisualizerAgent:
    """
    Create simple visuals (charts) from structured data.
    For this project we will support: frequency charts (term counts), simple bar charts from extracted numeric tables.
    """

    def __init__(self, out_dir="outputs/visuals"):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    def term_frequency_chart(self, docs: List[Dict[str, Any]], top_n: int = 10) -> str:
        # simple term frequency from summaries
        from collections import Counter
        import re

        counter = Counter()
        for d in docs:
            text = d.get("summary", "") + " " + d.get("title", "")
            tokens = re.findall(r"\w{3,}", text.lower())
            counter.update(tokens)
        most = counter.most_common(top_n)
        if not most:
            return ""
        labels, vals = zip(*most)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(range(len(vals))[::-1], vals, tick_label=list(labels)[::-1])
        ax.set_title("Top terms across summarized sources")
        out_path = os.path.join(self.out_dir, f"term_freq_{int(time.time())}.png")
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def save_table_chart(self, name: str, categories: List[str], values: List[float]) -> str:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(categories, values)
        ax.set_title(name)
        out_path = os.path.join(self.out_dir, f"chart_{name.replace(' ','_')}_{int(time.time())}.png")
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return out_path


# -----------------------------
# Writer Agent
# -----------------------------
class WriterAgent:
    """
    Assemble structured report sections into a final draft. Uses LLM to generate human-friendly text with inline citations.
    """

    def __init__(self, model=None):
        self.model = model or OPENAI_MODEL

    def produce_report(self, topic: str, structured_notes: List[Dict[str, Any]], visuals: List[str]) -> Dict[str, Any]:
        """
        Create a structured report object with sections:
          - title, executive_summary, methodology, findings (with citations), charts, references
        """
        # build combined context snippet
        context = "\n\n".join([f"{i+1}. {n['title']} -- {n['meta'].get('type','')}\n{n['summary']}" for i, n in enumerate(structured_notes)])
        prompt = f"""
You are a research writer. Produce a structured report on the topic: "{topic}".
Use the INTERNAL NOTES below (these are summaries with source tags). Create sections:
- Title
- Executive Summary (3-6 sentences)
- Methodology (how the agent gathered and filtered info)
- Findings (bullet points; for each bullet cite sources in square brackets)
- Discussion & Limitations
- Actionable Recommendations (3)
- References (list with URLs if provided)

INTERNAL NOTES:
{context}

Include references for each factual claim by referencing the source titles in INTERNAL NOTES.
"""
        report_text = llm_chat(prompt, temperature=0.0, max_tokens=1200)
        # Build report object
        report = {
            "id": str(uuid.uuid4()),
            "topic": topic,
            "text": report_text,
            "notes": structured_notes,
            "visuals": visuals,
            "generated_at": time.time(),
        }
        return report


# -----------------------------
# Reflection Agent
# -----------------------------
class ReflectionAgent:
    """
    Reviews a draft report and suggests improvements (clarity, missing citations, contradictions).
    It can produce a revised_report_text or a list of edit suggestions.
    """

    def __init__(self):
        pass

    def critique_and_revise(self, report_text: str, notes: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
        # Ask LLM to critique then produce revised version
        prompt = f"""
You are an expert editor and subject-matter verifier. Below is a draft report and the internal notes (summaries with sources).
1) Provide a short critique listing up to 6 specific issues (missing citations, unclear claims, contradictions, lack of limitations).
2) Produce a revised improved report text addressing those critiques and improving clarity. Ensure claims are marked with source references where available.

DRAFT REPORT:
{report_text}

INTERNAL NOTES:
{chr(10).join([f'- {n["title"]}: {n["summary"]}' for n in notes])}
"""
        crit = llm_chat("List issues only:\n\n" + prompt, temperature=0.0, max_tokens=400)
        revised = llm_chat("Revise the report:\n\n" + prompt, temperature=0.0, max_tokens=1200)
        # return list of critique items (split lines) and revised text
        crit_list = [line.strip("- ").strip() for line in crit.splitlines() if line.strip()]
        return revised, crit_list


# -----------------------------
# Controller/Orchestrator
# -----------------------------
class ResearchOrchestrator:
    """
    Orchestrates the agents: search -> summarize -> (embedding & RAG) -> writer -> visualize -> reflect -> finalize
    """

    def __init__(self, materials_dir="materials", outputs_dir="outputs"):
        self.searcher = SearcherAgent(materials_dir=materials_dir)
        self.summarizer = SummarizerAgent()
        self.visualizer = VisualizerAgent(out_dir=os.path.join(outputs_dir, "visuals"))
        self.writer = WriterAgent()
        self.reflection = ReflectionAgent()
        self.episodic = EpisodicMemory()
        self.longterm = LongTermMemory()
        os.makedirs(outputs_dir, exist_ok=True)
        self.outputs_dir = outputs_dir

    def run_full_pipeline(self, topic: str, persist: bool = True) -> Dict[str, Any]:
        # Step 1: search
        raw_items = self.searcher.search(topic)

        # Step 2: summarization
        notes = self.summarizer.summarize_items(raw_items)

        # Step 3: build small in-memory embedding/semantic index for RAG (so writer can ask targeted questions)
        texts = [n["summary"] for n in notes]
        if texts:
            embedding_vectors, ids = embed_texts(texts)  # wrapper in tools.py
            # store temporarily in episodic memory for this run
            eid = f"run_{int(time.time())}"
            self.episodic.write(eid, {"topic": topic, "notes": notes})
        else:
            embedding_vectors, ids = [], []

        # Step 4: visualization
        term_chart = self.visualizer.term_frequency_chart(notes, top_n=12)
        visuals = [term_chart] if term_chart else []

        # Step 5: writer
        report = self.writer.produce_report(topic, notes, visuals)

        # Step 6: reflection and revision loop
        revised_text, critiques = self.reflection.critique_and_revise(report["text"], notes)
        report["revised_text"] = revised_text
        report["critiques"] = critiques

        # Step 7: save outputs
        # PDF (report text + visuals)
        pdf_path = save_pdf_report(report, out_dir=self.outputs_dir)
        pptx_path = save_pptx(report, out_dir=self.outputs_dir)
        report["pdf_path"] = pdf_path
        report["pptx_path"] = pptx_path

        # Step 8: long-term memory store of report metadata
        self.longterm.write(topic, {"report_id": report["id"], "topic": topic, "pdf": pdf_path, "pptx": pptx_path, "summary": report["text"]})

        return report
