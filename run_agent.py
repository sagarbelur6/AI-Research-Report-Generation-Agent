# run_agent.py
"""
Example run script that shows how to use the ResearchOrchestrator to generate a report.
Put PDF/TXT materials into ./materials before running.
"""
from agents import ResearchOrchestrator
import json
import os

if __name__ == "__main__":
    topic = "machine learning fairness"  # example topic; change to whatever you want
    orchestrator = ResearchOrchestrator(materials_dir="materials", outputs_dir="outputs")
    print("Running full research pipeline for topic:", topic)
    report = orchestrator.run_full_pipeline(topic)
    print("Report generated:")
    print("  id:", report["id"])
    print("  pdf:", report["pdf_path"])
    print("  pptx:", report["pptx_path"])
    # Save a short metadata summary
    meta_path = os.path.join("outputs", f"report_meta_{report['id']}.json")
    with open(meta_path, "w") as f:
        json.dump({"id": report["id"], "topic": report["topic"], "pdf": report["pdf_path"], "pptx": report["pptx_path"]}, f, indent=2)
    print("All done. Check outputs/ folder.")
