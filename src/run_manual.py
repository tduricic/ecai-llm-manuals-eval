#!/usr/bin/env python3
# ==============================================================
# run_manual.py  –  Run one technical manual through the 36-run grid
# ==============================================================

import os
import json
import csv
import re
import time
import pathlib
import argparse
import pickle
import random
import logging
from collections import defaultdict
from typing import List, Dict, Tuple, Any # Added Any

# Attempt to import necessary libraries
try:
    import numpy as np
    import pandas as pd
    from tqdm import tqdm # Progress bars

    # Retrieval Imports (Install pyserini following its instructions if needed)
    # pip install pyserini faiss-cpu sentence-transformers # faiss-gpu if applicable
    from pyserini.search.lucene import LuceneSearcher
    from sentence_transformers import SentenceTransformer, util
    import faiss

    # LLM API Imports (Install required clients)
    # pip install openai google-generativeai groq
    import openai
    import google.generativeai as genai
    from groq import Groq as GroqClient # Renamed to avoid conflict

    # Metrics Imports (Install if not already present)
    # pip install datasets ragas scikit-learn # bert-score rouge-score
    from datasets import Dataset, Features, Value, Sequence
    from ragas import evaluate as ragas_eval
    from ragas.metrics import faithfulness, answer_correctness
    # Import other metric libraries later when needed (e.g., BERTScore, ROUGE)

except ImportError as e:
    print(f"Import Error: {e}. Please ensure all dependencies are installed.")
    print("Check imports and install: pyserini, faiss-cpu/faiss-gpu, sentence-transformers, openai, google-generativeai, groq, datasets, ragas, scikit-learn, numpy, pandas, tqdm")
    sys.exit(1)

# ------------------ CONFIGURATION -------------------------------------------

# --- Models ---
# Dictionary: key -> (api_provider, model_id_for_api, context_window_size)
# Verify model IDs are correct and accessible!
MODELS = {
    # User needs to confirm if gpt-4.1 is the intended/correct ID for their API access
    "gpt41-judge": ("openai",  "gpt-4.1",            128_000), # Assuming 128k unless user confirms 1M for this ID
    "gemini15p":   ("google",  "models/gemini-1.5-pro-latest",  1_000_000), # Using latest stable 1.5 Pro
    "llama3-70b":  ("groq",    "llama3-70b-8192",        8_192), # Groq model ID convention
    "mixtral":     ("groq",    "mixtral-8x7b-32768",   32_768), # Groq model ID convention
}
# Define the embedder model for dense retrieval (stronger model)
DENSE_RETRIEVAL_EMBEDDER = "BAAI/bge-large-en-v1.5"
# Define embedder for semantic similarity checks (can be smaller/faster)
SEMANTIC_SIMILARITY_EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"


# --- Retrieval Strategies ---
RETRIEVERS = ["none", "bm25", "hybrid"]

# --- Prompting Strategies ---
# We decided on Zero-Shot, Few-Shot (Direct JSON), Few-Shot CoT
PROMPTS = ["zero", "few_shot", "few_shot_cot"]

# --- Parameters ---
PARAMS = dict(
    # Retrieval params
    bm25_k=10,          # Number of initial candidates from BM25
    dense_k=10,         # Number of initial candidates from Dense
    hybrid_k=10,        # Number of final candidates after MMR
    mmr_lambda=0.5,     # MMR diversity parameter (0.5 = balance)
    # Scoring params
    sem_em_threshold=0.90,  # Cosine threshold for semantic answer match
    page_acc_tolerance=1,   # Allow page number +/- 1
    # RAGAS params (can configure later if needed)
    # ...
)

# --- Output Schema Expected from LLMs (Phase 2) ---
# Based on response #58 (single page, added category/persona)
# NOTE: Models will be prompted to produce this structure.
# The `answer` type depends on the gold standard's `answer_type` for the question.
EXPECTED_JSON_SCHEMA = """
{
  "answer": <value matching input answer_type: bool|str|list[str]>,
  "page": <page number as integer OR null>,
  "predicted_category": "<category_label_from_list>",
  "predicted_persona": "<persona_label_from_list>"
}
OR for Unanswerable:
{
  "answer": null,
  "page": null,
  "predicted_category": "Unanswerable",
  "predicted_persona": null
}
"""

# --- API Call Configuration ---
# Add delays, retry logic if needed later
API_DELAY_SECONDS = 0.2 # Small delay between general API calls

# -----------------------------------------------------------------------------

# --- Setup basic logging ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# =============================================================================
# --- Function Definitions (Will be added step-by-step) ---
# =============================================================================

# Placeholder for functions we will define later
def load_gold_qa(qa_file_path: Path) -> pd.DataFrame: pass
def setup_retrievers(pages: List[Dict], embedder: SentenceTransformer, index_dir: Path) -> Dict[str, Any]: pass
def get_context(query: str, retriever_type: str, retrievers: Dict[str, Any], pages: List[Dict], full_manual_text: str) -> str: pass
def get_prompt(question: str, answer_type: str, context: str, prompt_type: str) -> str: pass
def call_llm(model_key: str, prompt: str) -> str: pass
def parse_llm_json_output(llm_output_str: str) -> Tuple[bool, Dict]: pass
def score_prediction(gold_row: pd.Series, parsed_output: Dict, retrieved_context: str, ragas_metrics: List, sem_embedder: SentenceTransformer) -> Dict[str, float]: pass
# ... other helpers ...


# =============================================================================
# --- Main Execution Block ---
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run one technical manual through the 36-run RAG/LLM benchmark grid.")
    parser.add_argument("-p", "--pages", type=pathlib.Path, required=True,
                        help="Path to the manual's pages input file (*_pages.jsonl)")
    parser.add_argument("-q", "--qa", type=pathlib.Path, required=True,
                        help="Path to the GOLD standard QA file (*_gold.csv prepared for Phase 2)")
    parser.add_argument("-o", "--outdir", type=pathlib.Path, required=True,
                        help="Base output directory where run results subdirectory will be created.")
    # Add other options later if needed (e.g., specific model, resume)
    args = parser.parse_args()

    # --- Setup Output Directory ---
    manual_stem = args.pages.stem.replace('_pages', '') # Get base name
    run_output_dir = args.outdir / manual_stem
    try:
        run_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory for this run: {run_output_dir}")
    except OSError as e:
         logging.error(f"Could not create output directory {run_output_dir}: {e}"); sys.exit(1)

    # --- Initial Argument Validation ---
    if not args.pages.is_file(): logging.error(f"Pages file not found: {args.pages}"); sys.exit(1)
    if not args.qa.is_file(): logging.error(f"Gold QA file not found: {args.qa}"); sys.exit(1)

    print(f"\nStarting benchmark run for manual: {manual_stem}")
    print(f"Pages source: {args.pages}")
    print(f"Gold QA source: {args.qa}")
    print(f"Output dir: {run_output_dir}")
    print("-" * 30)
    print(f"Models: {list(MODELS.keys())}")
    print(f"Retrievers: {RETRIEVERS}")
    print(f"Prompts: {PROMPTS}")
    print(f"Total Conditions: {len(MODELS) * len(RETRIEVERS) * len(PROMPTS)}")
    print("-" * 30)

    # --- Next steps will go here: ---
    # 1. Load pages & gold QA data
    # 2. Initialize embedders
    # 3. Build/Load retrieval indices
    # 4. Initialize API clients (maybe better done inside loop?)
    # 5. Start the main loops (model, retriever, prompt, question)
    # 6. Inside loop: get context, build prompt, call LLM, parse, score, store result
    # 7. Save results per condition to pickle
    # 8. Update manifest

    print("\nSetup complete. Ready to implement data loading and retriever setup.")