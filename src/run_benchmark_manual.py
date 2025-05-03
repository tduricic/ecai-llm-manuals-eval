#!/usr/bin/env python3
# ==============================================================
# run_benchmark_manual.py
# ==============================================================
# Runs the RAG/LLM benchmark grid (Models x Retrievers x Prompts)
# for a single technical manual using a pre-processed gold QA dataset.
# Saves detailed results per condition to a pickle file.
# ==============================================================
import os
import sys
from pathlib import Path
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
from typing import List, Dict, Tuple, Any

# --- Attempt to import necessary libraries ---
# Core Processing
try:
    import numpy as np
    import pandas as pd
    from tqdm import tqdm # Progress bars
except ImportError as e:
     print(f"Core library import error: {e}. pip install numpy pandas tqdm"); sys.exit(1)

# Configuration Loading
try:
    import yaml
except ImportError:
     print("PyYAML not found. Please install it (`pip install pyyaml`) to load config file."); sys.exit(1)

# Retrieval (Optional components)
try:
    from pyserini.search.lucene import LuceneSearcher
    HAS_PYSERINI = True
except ImportError:
    logging.warning("Pyserini not found. BM25 and Hybrid retrieval will not be available.")
    HAS_PYSERINI = False; LuceneSearcher = None

try:
    import faiss
    HAS_FAISS = True
except ImportError:
     logging.warning("FAISS not found. Dense and Hybrid retrieval will not be available.")
     HAS_FAISS = False; faiss = None

try:
    from sentence_transformers import SentenceTransformer, util
    HAS_SBERT = True
except ImportError:
     logging.warning("Sentence Transformers not found. Dense/Hybrid retrieval and Semantic Metrics unavailable.")
     HAS_SBERT = False; SentenceTransformer = None; util = None

# LLM APIs
try:
    import openai
    import google.generativeai as genai
    from groq import Groq as GroqClient
    from dotenv import load_dotenv
    HAS_LLM_APIS = True
except ImportError:
     logging.warning("LLM API clients (openai, google-generativeai, groq) or python-dotenv not found. LLM calls will fail.")
     HAS_LLM_APIS = False; openai = None; genai = None; GroqClient = None; load_dotenv = None

# Metrics (Optional components)
try:
    from datasets import Dataset, Features, Value, Sequence
    from ragas import evaluate as ragas_eval
    from ragas.metrics import faithfulness, answer_correctness
    # Import other metrics later e.g.
    # from bert_score import score as bert_score_calc
    # from rouge_score import rouge_scorer
    HAS_METRIC_LIBS = True
except ImportError:
     logging.warning("Metrics libraries (datasets, ragas) not found. Metric calculation will be limited.")
     HAS_METRIC_LIBS = False; Dataset = None; ragas_eval = None; faithfulness = None; answer_correctness = None

# ------------------ CONFIGURATION LOADING ------------------------------------
# Load benchmark-specific settings
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
    CONFIG_PATH = SCRIPT_DIR.parent / "config" / "settings_benchmark.yaml" # Use benchmark config file
    logging.info(f"Loading benchmark configuration from: {CONFIG_PATH}")
    with open(CONFIG_PATH, 'r') as f:
        config_bench = yaml.safe_load(f)
    logging.info("Benchmark configuration loaded successfully.")
except FileNotFoundError:
    logging.error(f"Benchmark configuration file not found: {CONFIG_PATH}"); sys.exit(1)
except yaml.YAMLError as e:
     logging.error(f"Error parsing benchmark configuration file {CONFIG_PATH}: {e}"); sys.exit(1)
except Exception as e:
     logging.error(f"An unexpected error occurred during benchmark configuration loading: {e}", exc_info=True); sys.exit(1)

# --- Extract config values into global constants/variables ---
try:
    # Models to Test (Defined in config)
    MODELS_TO_TEST = config_bench['models_to_test']
    # Embedding Models (Defined in config)
    DENSE_RETRIEVAL_EMBEDDER = config_bench['dense_retriever_embedder']
    SEMANTIC_SIMILARITY_EMBEDDER = config_bench['semantic_similarity_embedder']
    # Retrieval & Scoring Params (Defined in config)
    RETRIEVAL_PARAMS = config_bench['retrieval_params']
    SCORING_PARAMS = config_bench['scoring_params']
    # API Params (Defined in config)
    API_DELAY_SECONDS = config_bench['api_params']['api_delay_seconds']

    # Constants defined directly in script (could also be moved to config)
    RETRIEVERS = ["none", "bm25", "hybrid"]
    PROMPTS = ["zero_shot", "few_shot", "few_shot_cot"]
    VALID_CATEGORIES = [ # Should match Phase 1 CATEGORY_TARGETS keys
        "Specification Lookup", "Tool/Material Identification",
        "Procedural Step Inquiry", "Location/Definition",
        "Conditional Logic/Causal Reasoning", "Safety Information Lookup",
        "Unanswerable"
    ]
    VALID_PERSONAS = ["Novice User", "Technician", "SafetyOfficer"]
    VALID_ANSWER_TYPES = ["binary", "single_label", "multi_label", "open_ended", "procedural_steps"]

    # Phase 2 Output Schema Reminder (for prompts)
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
    # System message for Phase 2 LLM calls (needs refinement)
    BASE_SYSTEM_MSG = """You are a helpful assistant answering questions based ONLY on the provided context from a technical manual.
    You MUST return ONLY a single, valid JSON object matching the schema shown below. Do not include any preamble, explanations, or markdown formatting around the JSON.
    If the provided context does not contain the information to answer the question, output the specific JSON for unanswerable questions.

    JSON Output Schema:
    """ + EXPECTED_JSON_SCHEMA

except KeyError as e:
     logging.error(f"Missing key in benchmark configuration file {CONFIG_PATH}: {e}"); sys.exit(1)
# -----------------------------------------------------------------------------

# --- Setup basic logging ---
# (Consider adjusting level or adding file handler for long runs)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# =============================================================================
# --- Function Definitions (Placeholders) ---
# =============================================================================

def load_gold_qa(qa_file_path: Path) -> pd.DataFrame:
    """Loads and validates the Phase 2 Gold Standard QA CSV file."""
    logging.info(f"Loading Gold QA data from: {qa_file_path}")
    expected_cols = ['question', 'answer_type', 'gold_answer', 'gold_page', 'gold_category', 'gold_persona'] # Minimum expected
    try:
         df = pd.read_csv(qa_file_path)
         logging.info(f"Loaded {len(df)} gold QA pairs.")
         # Basic validation
         missing = [col for col in expected_cols if col not in df.columns]
         if missing:
              raise ValueError(f"Gold QA file missing required columns: {missing}")
         # TODO: Add validation for answer_type values, format of gold_answer/gold_page?
         return df
    except Exception as e:
         logging.error(f"Failed to load or validate Gold QA file {qa_file_path}: {e}", exc_info=True)
         sys.exit(1)

def load_pages(page_file_path: Path) -> List[Dict[str, Any]]:
    """Loads pages from JSONL file."""
    logging.info(f"Loading Pages data from: {page_file_path}")
    pages = []
    try:
        with open(page_file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try: pages.append(json.loads(line.strip()))
                except: logging.warning(f"Skipping invalid JSON line {i+1} in {page_file_path}")
        if not pages: raise ValueError("No pages loaded.")
        logging.info(f"Loaded {len(pages)} pages.")
        # Basic validation of page structure
        if not all("page_num" in p and "markdown_content" in p for p in pages):
             logging.warning("Some loaded pages missing 'page_num' or 'markdown_content'.")
        return pages
    except Exception as e:
         logging.error(f"Failed to load pages from {page_file_path}: {e}", exc_info=True); sys.exit(1)

# Placeholders for other functions (signatures may need refinement)
def setup_retrievers(pages: List[Dict], embedder_name: str, index_dir: Path) -> Dict[str, Any]: pass
def get_context(query: str, retriever_type: str, retrievers: Dict[str, Any], pages_dict: Dict[int, str], full_manual_text: str, query_embedding: np.ndarray = None) -> Tuple[str, List[Dict]]: pass # Return context string and source chunks/docs
def build_prompt_messages(question: str, answer_type: str, context: str, prompt_type: str, system_message: str) -> List[Dict[str, str]]: pass # Returns list of messages for API
def call_llm(model_key: str, prompt_messages: List[Dict], models_config: Dict) -> str: pass
def parse_llm_json_output(llm_output_str: str, expected_answer_type: str) -> Tuple[bool, Dict]: pass
def score_prediction(gold_row: pd.Series, model_output_json: Dict, retrieved_context: str, retrieved_sources: List[Dict], sem_embedder: SentenceTransformer) -> Dict[str, Any]: pass # Return dict of metric scores

# =============================================================================
# --- Main Execution Block ---
# =============================================================================

if __name__ == "__main__":
    # --- Argument Parsing (REVISED) ---
    parser = argparse.ArgumentParser(description="Run one technical manual through the RAG/LLM benchmark grid.")
    parser.add_argument("-i", "--input_dir", type=pathlib.Path, required=True,
                        help="Path to the directory containing the manual's processed files (e.g., data/processed/heat_pump_dryer)")
    parser.add_argument("-o", "--outdir", type=pathlib.Path, required=True,
                        help="Base output directory where the results subdirectory for this manual will be created (e.g., results/).")
    # Example: python src/run_benchmark_manual.py -i data/processed/heat_pump_dryer -o results/benchmark_runs
    args = parser.parse_args()

    # --- Validate Input Directory ---
    if not args.input_dir.is_dir():
        logging.error(f"Input directory not found or is not a directory: {args.input_dir}"); sys.exit(1)

    # --- Derive Manual Name and File Paths ---
    manual_name = args.input_dir.name # e.g., "heat_pump_dryer"
    pages_file_path = args.input_dir / f"{manual_name}.jsonl"
    qa_file_path = args.input_dir / f"{manual_name}_gold.csv" # Assuming this is the Phase 2 ready gold file

    # --- Validate Derived File Paths ---
    if not pages_file_path.is_file(): logging.error(f"Pages file not found at expected location: {pages_file_path}"); sys.exit(1)
    if not qa_file_path.is_file(): logging.error(f"Gold QA file not found at expected location: {qa_file_path}"); sys.exit(1)

    # --- Setup Output Directory (using manual_name) ---
    run_output_dir = args.outdir / manual_name # e.g., results/benchmark_runs/heat_pump_dryer
    index_dir = run_output_dir / "indices"
    results_dir = run_output_dir / "results"
    try:
        run_output_dir.mkdir(parents=True, exist_ok=True)
        index_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory for this run: {run_output_dir}")
    except OSError as e:
         logging.error(f"Could not create output directories under {args.outdir}: {e}"); sys.exit(1)

    # --- Load Environment Variables for API Keys ---
    if HAS_LLM_APIS and load_dotenv:
        print("Loading environment variables from .env file...")
        if load_dotenv(): print(" -> .env file loaded.")
        else: print(" -> .env file not found, ensure keys are set via environment.")
    elif not HAS_LLM_APIS:
         logging.warning("LLM API libraries not found. Script may fail.")

    # --- Log Configuration ---
    print(f"\nStarting benchmark run for manual: {manual_name}") # Use derived name
    print(f"Pages source: {pages_file_path}") # Use derived path
    print(f"Gold QA source: {qa_file_path}")   # Use derived path
    print(f"Output dir: {run_output_dir}")
    print("-" * 30)
    print(f"Models to test: {list(MODELS_TO_TEST.keys())}")
    print(f"Retrievers to test: {RETRIEVERS}")
    print(f"Prompts to test: {PROMPTS}")
    print(f"Total Conditions: {len(MODELS_TO_TEST) * len(RETRIEVERS) * len(PROMPTS)}")
    print(f"Dense Retriever Embedder: {DENSE_RETRIEVAL_EMBEDDER}")
    print("-" * 30)

    # --- Placeholder for Next Steps ---
    print("\nSetup complete. Ready to implement data loading and retriever setup.")
    # TODO: Implement load_gold_qa using qa_file_path
    # TODO: Implement load_pages using pages_file_path
    # TODO: Initialize embedders
    # TODO: Call setup_retrievers using index_dir
    # TODO: Implement main loops and call functions, saving results to results_dir

    print("\n--- run_benchmark_manual.py finished initial setup ---")