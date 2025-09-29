# tools.py
import os
from typing import List, Dict, Any
from duckduckgo_search import ddg
import arxiv
import wikipedia
import glob
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
import logging

logger = logging.getLogger(__name__)

# ---------- Search wrappers ----------
def ddg_search(query: str, max_results: int = 8) -> List[Dict[str, Any]]:
    hits = ddg(query, max_results=max_results)
    # ddg returns list of dicts with 'title','href','body' usually
    results = []
    for h in hits:
        results.append({"title": h.get("title"), "snippet": h.get("body"), "link": h.get("href")})
    return results

def arxiv_search(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
    out = []
    for r in search.results():
        out.append({"title": r.title, "summary": r.summary, "url": r.entry_id})
    return out

def wikipedia_search(query: str) -> Dict[str, Any]:
    try:
        s = wikipedia.search(query, results=1)
        if not s:
            return {}
        page = wikipedia.page(s[0])
        return {"title": page.title, "summary": page.summary[:2000], "url": page.url}
    except Exception:
        return {}

# ---------- Local documents loader ----------
def load_local_documents(materials_dir: str = "materials") -> List[Dict[str, Any]]:
    docs = []
    # PDFs
    for p in glob.glob(os.path.join(materials_dir, "*.pdf")):
        try:
            loader = PyPDFLoader(p)
            pages = loader.load()
            for i, pg in enumerate(pages):
                text = pg.page_content
                docs.append({"source": os.path.basename(p), "text": text, "metadata": {"page": i}})
        except Exception as e:
            logger.warning("Failed to load PDF %s: %s", p, e)
    # TXT, MD
    for p in glob.glob(os.path.join(materials_dir, "*.txt")) + glob.glob(os.path.join(materials_dir, "*.md")):
        try:
            loader = TextLoader(p, encoding="utf-8")
            pages = loader.load()
            for pg in pages:
                docs.append({"source": os.path.basename(p), "text": pg.page_content, "metadata": {}})
        except Exception as e:
            logger.warning("Failed to load text %s: %s", p, e)
    return docs

# ---------- Chunking & Embedding helpers ----------
def chunk_documents(local_docs: List[Dict[str, Any]], chunk_size: int = 700, overlap: int = 100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    docs_for_split = []
    from langchain.schema import Document
    for d in local_docs:
        docs_for_split.append(Document(page_content=d["text"], metadata={"source": d["source"], **d.get("metadata", {})}))
    chunks = splitter.split_documents(docs_for_split)
    return chunks

def embed_texts(texts: List[str]):
    """Return embeddings using OpenAIEmbeddings and ids (this is a thin wrapper)."""
    embeddings_model = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    vectors = []
    for t in texts:
        vectors.append(embeddings_model.embed_query(t))
    ids = list(range(len(vectors)))
    return vectors, ids

def semantic_search(chunks, query: str, k: int = 5):
    """Use FAISS quickly on provided chunks (chunks are langchain Documents)."""
    embeddings_model = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    vs = FAISS.from_documents(chunks, embeddings_model)
    retr = vs.as_retriever(search_kwargs={"k": k})
    return retr.get_relevant_documents(query)
