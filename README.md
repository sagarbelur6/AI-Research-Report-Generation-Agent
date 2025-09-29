# AI Research & Report Generation Agent

An autonomous multi-agent system that researches a topic across web, academic and internal sources and produces a structured report with citations, summaries and visuals.

## Features
- Multi-agent orchestration (Searcher, Summarizer, Writer, Visualizer, Reflection)
- Episodic & long-term memory
- Multi-modal outputs: PDF report, PPTX slides, dashboard (Streamlit)
- Uses web (DuckDuckGo), arXiv, Wikipedia + local materials
- RAG-style retrieval using local doc chunking and embeddings
- Reflection loop for iterative improvement of the report
- OpenAI GPT as the LLM backend

## Setup

1. Clone repo and cd into it.

2. Create a `.env` with:
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini # optional override


3. Install dependencies:


pip install -r requirements.txt


4. Add local materials (PDF/TXT/MD/code/images) to `materials/` folder.

5. Run example:


python run_agent.py


6. Optional UI:


streamlit run streamlit_app.py


## Files
- `agents.py` — agents & orchestrator
- `tools.py` — search & local doc helpers
- `report_generator.py` — PDF & PPTX generation
- `memory.py` — episodic & long-term memory
- `run_agent.py` — example end-to-end run
- `streamlit_app.py` — minimal dashboard
- `requirements.txt`, `README.md`

## Notes & Extensions
- The project is intentionally modular so you can swap search tools (SerpAPI), use LangGraph orchestration, add advanced multi-modal embeddings (CLIP/HuggingFace), persist FAISS indices, or add provenance/citation formatting improvements.
- For large-scale use, persist FAISS index and implement streaming updates to avoid re-embedding everything.

AI_Research_Agent/
│
├── agents.py                 # all agents + orchestrator
├── tools.py                  # search helpers, local docs, embeddings
├── memory.py                 # episodic & long-term memory
├── report_generator.py       # PDF & PPTX generation
├── run_agent.py              # example script to run full pipeline
├── streamlit_app.py          # optional Streamlit UI
├── requirements.txt          # all Python dependencies
├── README.md                 # setup and usage instructions
├── .env                      # your OpenAI API key (create this file)
│
├── materials/                # folder to put all PDFs, TXT, MD docs you want to research
│   ├── example1.pdf
│   ├── example2.txt
│   └── ...
│
└── outputs/                  # generated outputs will be saved here automatically
    ├── report_*.pdf
    ├── slides_*.pptx
    └── visuals/
        └── *.png

RUN -> python run_agent.py
RUN Use the Streamlit UI -> streamlit run streamlit_app.py
