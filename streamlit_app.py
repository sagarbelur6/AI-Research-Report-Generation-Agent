# streamlit_app.py
import streamlit as st
from agents import ResearchOrchestrator
import os

st.set_page_config(page_title="AI Research Agent", layout="wide")
st.title("AI Research & Report Generation Agent")

topic = st.text_input("Enter research topic", value="climate change and agriculture")
materials_dir = st.text_input("Materials folder (local docs)", value="materials")
if st.button("Run research"):
    orchestrator = ResearchOrchestrator(materials_dir=materials_dir, outputs_dir="outputs")
    with st.spinner("Running multi-agent research pipeline... this may take a while"):
        report = orchestrator.run_full_pipeline(topic)
    st.success("Report generated")
    st.write("Report ID:", report["id"])
    st.download_button("Download PDF", data=open(report["pdf_path"], "rb").read(), file_name=os.path.basename(report["pdf_path"]))
    st.download_button("Download PPTX", data=open(report["pptx_path"], "rb").read(), file_name=os.path.basename(report["pptx_path"]))
    st.subheader("Executive Summary (excerpt)")
    st.write((report.get("revised_text") or report.get("text"))[:2000])
