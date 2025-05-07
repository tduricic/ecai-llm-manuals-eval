#!/usr/bin/env python3
# ==============================================================
# run_benchmark_manual.py (v2 - Ollama, Token Count, Refined Truncation)
# ==============================================================
# Runs the RAG/LLM benchmark grid (Models x Retrievers x Prompts)
# for a single technical manual using a pre-processed gold QA dataset.
# Saves detailed results per condition (input, context, raw output, token counts)
# to allow for separate scoring later.
#
# Includes enhanced logging, refactored truncation, Ollama support,
# stores final prompt token counts, and improves JSON parsing debugging.
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
import ast # For safely parsing list strings
import subprocess # For running pyserini indexing
from collections import defaultdict
from typing import List, Dict, Tuple, Any, Optional
import shutil # For removing directories

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

# Text Splitting & Tokenization
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from transformers import AutoTokenizer
    HAS_SPLITTER_LIBS = True
except ImportError:
     logging.warning("Libraries 'langchain-text-splitters' or 'transformers' not found. pip install langchain-text-splitters transformers tiktoken")
     HAS_SPLITTER_LIBS = False; RecursiveCharacterTextSplitter = None; AutoTokenizer = None

# Retrieval (Optional components)
try:
    # Pyserini requires Java 11+ installed
    from pyserini.search.lucene import LuceneSearcher
    HAS_PYSERINI = True
except ImportError:
    logging.warning("Pyserini not found or Java 11+ not configured correctly. BM25 and Hybrid retrieval will not be available.")
    HAS_PYSERINI = False; LuceneSearcher = None
except Exception as e: # Catch potential Java errors too
    logging.warning(f"Error importing Pyserini (check Java 11+ installation/path): {e}")
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
    # Import specific exceptions for better handling
    import google.api_core.exceptions
    from google.generativeai.types import GenerationConfig # Correct import path
    from groq import Groq as GroqClient, RateLimitError as GroqRateLimitError, APIError as GroqAPIError, APITimeoutError as GroqTimeoutError, APIConnectionError as GroqConnectionError # Renamed client, added timeout/connection
    from dotenv import load_dotenv
    import ollama # Use official Ollama library
    HAS_LLM_APIS = True
except ImportError as e:
     logging.warning(f"LLM API clients (openai, google-generativeai, groq, ollama) or python-dotenv not found. LLM calls will fail. Error: {e}")
     HAS_LLM_APIS = False; openai = None; genai = None; GroqClient = None; load_dotenv = None; google = None; GroqRateLimitError = None; GroqAPIError = None; GenerationConfig = None; GroqTimeoutError = None; GroqConnectionError = None; ollama = None

# Metrics (Optional components - Imports removed as scoring is deferred)

# ------------------ CONFIGURATION LOADING ------------------------------------
# Setup basic logging FIRST to capture config loading issues
log_level = logging.INFO # Change to logging.DEBUG for more verbose output
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
# Reduce noise from underlying libraries unless debugging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("google.generativeai").setLevel(logging.WARNING)
logging.getLogger("pyserini").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("ollama").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

try:
    SCRIPT_DIR = Path(__file__).resolve().parent
    CONFIG_PATH = SCRIPT_DIR.parent / "config" / "settings_benchmark.yaml"
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
    logging.info("Extracting configuration parameters...")
    MODELS_TO_TEST = config_bench['models_to_test']
    logging.info(f" -> Models to test: {list(MODELS_TO_TEST.keys())}")
    RETRIEVERS_TO_TEST = config_bench['retrievers_to_test']
    logging.info(f" -> Retrievers to test: {RETRIEVERS_TO_TEST}")
    PROMPTS_TO_TEST = config_bench['prompts_to_test']
    logging.info(f" -> Prompts to test: {PROMPTS_TO_TEST}")
    DENSE_RETRIEVAL_EMBEDDER = config_bench['dense_retriever_embedder']
    logging.info(f" -> Dense embedder: {DENSE_RETRIEVAL_EMBEDDER}")
    SEMANTIC_SIMILARITY_EMBEDDER = config_bench['semantic_similarity_embedder']
    logging.info(f" -> Semantic scoring embedder: {SEMANTIC_SIMILARITY_EMBEDDER}")

    RETRIEVAL_PARAMS = config_bench['retrieval_params']
    CHUNK_SIZE_TOKENS = RETRIEVAL_PARAMS.get('chunk_size_tokens', 512)
    CHUNK_OVERLAP_TOKENS = RETRIEVAL_PARAMS.get('chunk_overlap_tokens', 64)
    logging.info(f" -> Chunk Size: {CHUNK_SIZE_TOKENS} tokens, Overlap: {CHUNK_OVERLAP_TOKENS} tokens")
    BM25_K = RETRIEVAL_PARAMS.get('bm25_k', 10)
    DENSE_K = RETRIEVAL_PARAMS.get('dense_k', 10)
    HYBRID_K = RETRIEVAL_PARAMS.get('hybrid_k', 10)
    MMR_LAMBDA = RETRIEVAL_PARAMS.get('mmr_lambda', 0.5)
    logging.info(f" -> Retrieval K values (BM25: {BM25_K}, Dense: {DENSE_K}, Hybrid: {HYBRID_K})")
    logging.info(f" -> Hybrid MMR Lambda: {MMR_LAMBDA}")
    PYSERINI_THREADS = RETRIEVAL_PARAMS.get('pyserini_threads', 4)
    EMBEDDING_BATCH_SIZE = RETRIEVAL_PARAMS.get('embedding_batch_size', 32)
    LLM_RESPONSE_BUFFER_TOKENS = RETRIEVAL_PARAMS.get('llm_response_buffer_tokens', 4096) # Buffer for LLM response generation
    logging.info(f" -> LLM Response Buffer: {LLM_RESPONSE_BUFFER_TOKENS} tokens")
    NUM_FEW_SHOT_EXAMPLES = RETRIEVAL_PARAMS.get('num_few_shot_examples', 3)
    FEW_SHOT_STRATEGY = RETRIEVAL_PARAMS.get('few_shot_strategy', 'stratified')
    logging.info(f" -> Few-shot examples: {NUM_FEW_SHOT_EXAMPLES} (Strategy: {FEW_SHOT_STRATEGY})")
    FORCE_REINDEX = RETRIEVAL_PARAMS.get('force_reindex', False)
    logging.info(f" -> Force Re-index: {FORCE_REINDEX}")
    RANDOM_SEED = config_bench.get('random_seed', 42)
    logging.info(f" -> Random Seed: {RANDOM_SEED}")

    SCORING_PARAMS = config_bench['scoring_params'] # Used by analysis script
    API_DELAY_SECONDS = config_bench['api_params']['api_delay_seconds']
    API_MAX_RETRIES = config_bench['api_params'].get('api_max_retries', 3)
    API_RETRY_DELAY = config_bench['api_params'].get('api_retry_delay', 60)
    logging.info(f" -> API Delay: {API_DELAY_SECONDS}s, Max Retries: {API_MAX_RETRIES}, Retry Delay: {API_RETRY_DELAY}s")

    SYSTEM_PROMPT_PATH_STR = config_bench['files']['system_prompt']
    SYSTEM_PROMPT_PATH = SCRIPT_DIR.parent / SYSTEM_PROMPT_PATH_STR
    logging.info(f" -> System Prompt Path: {SYSTEM_PROMPT_PATH}")

    UNANSWERABLE_CATEGORY_NAME = "Unanswerable"
    PROCEDURAL_CATEGORY_NAME = "Procedural Step Inquiry"
    VALID_CATEGORIES = [
        "Specification Lookup", "Tool/Material Identification",
        PROCEDURAL_CATEGORY_NAME, "Location/Definition",
        "Conditional Logic/Causal Reasoning", "Safety Information Lookup",
        UNANSWERABLE_CATEGORY_NAME
    ]
    VALID_PERSONAS = ["Novice User", "Technician", "SafetyOfficer"]
    logging.info("Configuration parameters extracted.")

except KeyError as e:
     logging.error(f"Missing key in benchmark configuration file {CONFIG_PATH}: {e}"); sys.exit(1)
# -----------------------------------------------------------------------------

# =============================================================================
# --- Helper Function for Parsing List Strings ---
# =============================================================================

def safe_parse_list_string(list_str: str, default_value=None):
    """Safely parses a string representation of a list (e.g., "['a', 'b']") into a Python list."""
    if pd.isna(list_str) or not isinstance(list_str, str) or not list_str.strip():
        return default_value
    try:
        if list_str.startswith('[') and list_str.endswith(']'):
            parsed = ast.literal_eval(list_str)
            if isinstance(parsed, list):
                 if all(isinstance(item, str) for item in parsed):
                     return parsed
                 else:
                     logging.warning(f"Parsed list contains non-string elements: {list_str[:100]}... Returning default.")
                     return default_value
            else:
                 logging.warning(f"Parsed literal is not a list: {list_str[:100]}... Returning default.")
                 return default_value
        else:
            if list_str == '[]': return []
            logging.warning(f"String does not look like a list literal: {list_str[:100]}... Returning default.")
            return default_value
    except (ValueError, SyntaxError, TypeError, MemoryError) as e:
        logging.warning(f"Could not parse list string using ast.literal_eval: '{list_str[:100]}...'. Error: {e}. Returning default.")
        return default_value

# =============================================================================
# --- Initialization Functions ---
# =============================================================================

def initialize_clients_and_models(models_config: Dict, dense_embedder_name: str, semantic_embedder_name: str) -> Dict:
    """
    Initializes API clients (OpenAI, Google, Groq, Ollama) and Sentence Transformer models.
    Loads API keys from environment variables.
    """
    logging.info("--- Initializing Clients, Models, and Tokenizer ---")
    initialized_components = {
        "openai_client": None, "google_client": None, "groq_client": None,
        "ollama_client": None, "dense_embedder": None, "semantic_embedder": None,
        "system_prompt": None, "tokenizer": None
    }
    api_keys_loaded = {}

    # --- Load API Keys ---
    if HAS_LLM_APIS and load_dotenv:
        logging.info("Attempting to load API keys from .env file...")
        if load_dotenv(): logging.info(" -> .env file loaded.")
        else: logging.info(" -> .env file not found. Relying on environment variables.")
        api_keys_loaded['openai'] = os.getenv("OPENAI_API_KEY")
        api_keys_loaded['google'] = os.getenv("GOOGLE_API_KEY")
        api_keys_loaded['groq'] = os.getenv("GROQ_API_KEY")
    else: logging.warning("LLM API libraries or python-dotenv not available. Cannot load keys.")

    providers_needed = set(details['provider'] for details in models_config.values())
    logging.info(f"Providers needed based on config: {providers_needed}")

    # --- Initialize Clients (OpenAI, Google, Groq, Ollama) ---
    for provider in providers_needed:
        logging.info(f"Initializing client for provider: {provider}")
        if provider == "openai":
            if HAS_LLM_APIS and openai:
                if api_keys_loaded.get('openai'):
                    try:
                        initialized_components["openai_client"] = openai.OpenAI(api_key=api_keys_loaded['openai'], timeout=60.0, max_retries=0)
                        logging.info(" -> OpenAI client initialized successfully.")
                    except Exception as e: logging.error(f" -> Failed to initialize OpenAI client: {e}", exc_info=True)
                else: logging.warning(" -> OpenAI provider needed, but OPENAI_API_KEY not found.")
            else: logging.warning(" -> OpenAI provider needed, but 'openai' library not imported.")
        elif provider == "google":
            if HAS_LLM_APIS and genai:
                if api_keys_loaded.get('google'):
                    try:
                        genai.configure(api_key=api_keys_loaded['google'])
                        initialized_components["google_client"] = genai
                        logging.info(" -> Google GenAI configured successfully.")
                    except Exception as e: logging.error(f" -> Failed to configure Google GenAI: {e}", exc_info=True)
                else: logging.warning(" -> Google provider needed, but GOOGLE_API_KEY not found.")
            else: logging.warning(" -> Google provider needed, but 'google-generativeai' library not imported.")
        elif provider == "groq":
            if HAS_LLM_APIS and GroqClient:
                if api_keys_loaded.get('groq'):
                    try:
                        initialized_components["groq_client"] = GroqClient(api_key=api_keys_loaded['groq'], timeout=60.0, max_retries=0)
                        logging.info(" -> Groq client initialized successfully.")
                    except Exception as e: logging.error(f" -> Failed to initialize Groq client: {e}", exc_info=True)
                else: logging.warning(" -> Groq provider needed, but GROQ_API_KEY not found.")
            else: logging.warning(" -> Groq provider needed, but 'groq' library not imported.")
        elif provider == "ollama":
            if HAS_LLM_APIS and ollama:
                ollama_base_url = next((d.get('base_url') for k, d in models_config.items() if d.get('provider') == 'ollama' and d.get('base_url')), None)
                if ollama_base_url:
                    try:
                        client = ollama.Client(host=ollama_base_url)
                        initialized_components["ollama_client"] = client
                        logging.info(f" -> Ollama client initialized for host: {ollama_base_url}")
                        try:
                            client.list() # Test connection
                            logging.info(f" -> Ollama connection test successful.")
                        except Exception as conn_test_e: logging.warning(f" -> Ollama connection test failed: {conn_test_e}. Calls might fail.")
                    except Exception as e: logging.error(f" -> Failed to initialize Ollama client: {e}", exc_info=True)
                else: logging.warning(" -> Ollama provider needed, but no 'base_url' found in config.")
            else: logging.warning(" -> Ollama provider needed, but 'ollama' library not imported.")
        else:
            logging.warning(f" -> Unknown provider '{provider}' found in config. Skipping initialization.")

    # --- Initialize Tokenizer ---
    tokenizer_name_or_path = dense_embedder_name if dense_embedder_name else "bert-base-uncased"
    logging.info(f"Attempting to initialize Tokenizer: {tokenizer_name_or_path}")
    if HAS_SPLITTER_LIBS and AutoTokenizer:
        try:
            initialized_components["tokenizer"] = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
            logging.info(" -> Tokenizer initialized successfully.")
        except Exception as e:
            logging.error(f" -> Failed to initialize Tokenizer '{tokenizer_name_or_path}': {e}. Chunking/Truncation may use character count.", exc_info=True)
    else:
        logging.warning(" -> Transformers library not available. Cannot initialize tokenizer.")


    if HAS_SBERT and SentenceTransformer:
        if dense_embedder_name:
            logging.info(f"Initializing Dense Embedding model: {dense_embedder_name}")
            try:
                initialized_components["dense_embedder"] = SentenceTransformer(dense_embedder_name)
                logging.info(" -> Dense Embedder initialized.")
            except Exception as e: logging.error(f" -> Failed to initialize Dense Embedding model: {e}", exc_info=True)
        else: logging.warning(" -> No dense_retriever_embedder specified in config.")

        if semantic_embedder_name:
            logging.info(f"Initializing Semantic Similarity model: {semantic_embedder_name}")
            try:
                 initialized_components["semantic_embedder"] = SentenceTransformer(semantic_embedder_name)
                 logging.info(" -> Semantic Embedder initialized.")
            except Exception as e: logging.error(f" -> Failed to initialize Semantic Similarity model: {e}", exc_info=True)
        else: logging.warning(" -> No semantic_similarity_embedder specified in config.")
    else:
        logging.warning(" -> Sentence Transformers library not available. Cannot initialize embedding models.")

    # --- Load System Prompt ---
    logging.info(f"Loading system prompt from: {SYSTEM_PROMPT_PATH}")
    try:
        if SYSTEM_PROMPT_PATH.is_file():
            initialized_components["system_prompt"] = SYSTEM_PROMPT_PATH.read_text(encoding='utf-8')
            logging.info(f" -> System prompt loaded ({len(initialized_components['system_prompt'])} chars).")
        else:
            logging.error(f"FATAL: System prompt file not found: {SYSTEM_PROMPT_PATH}"); sys.exit(1)
    except Exception as e:
        logging.error(f"FATAL: Failed to load system prompt: {e}", exc_info=True); sys.exit(1)

    logging.info("--- Initialization Complete ---")
    return initialized_components

# =============================================================================
# --- Data Loading Functions ---
# =============================================================================

def load_gold_qa(qa_file_path: Path) -> pd.DataFrame:
    """Loads and validates the Phase 2 Gold Standard QA CSV file."""
    logging.info(f"--- Loading Gold QA Data from: {qa_file_path} ---")
    if not qa_file_path.is_file():
        logging.error(f"Gold QA file not found: {qa_file_path}"); sys.exit(1)
    expected_cols = [
        "question_id", "persona", "doc_id", "language", "question_text",
        "category", "gt_answer_snippet", "gt_page_number", "_self_grounded",
        "parsed_steps", "passed_strict_check", "corrected_steps", "procedural_comments"
    ]
    try:
         df = pd.read_csv(qa_file_path, keep_default_na=False, low_memory=False)
         logging.info(f" -> Loaded {len(df)} rows from Gold QA file.")
         if df.empty: logging.error(f"Gold QA file is empty: {qa_file_path}"); sys.exit(1)

         missing = [col for col in expected_cols if col not in df.columns]
         if missing: logging.error(f"Gold QA file missing expected columns: {missing}"); sys.exit(1)
         else: logging.info(" -> All expected columns found.")

         df['gt_page_number'] = df['gt_page_number'].replace(['None', ''], np.nan)
         df['gt_page_number'] = pd.to_numeric(df['gt_page_number'], errors='coerce').astype(pd.Int64Dtype())

         logging.info(" -> Parsing 'parsed_steps' column...")
         df['gold_steps_list'] = df['parsed_steps'].apply(lambda x: safe_parse_list_string(x, default_value=[]))
         parsed_count = df['gold_steps_list'].apply(lambda x: isinstance(x, list)).sum()
         logging.info(f" -> Successfully parsed 'parsed_steps' for {parsed_count}/{len(df)} rows.")

         # Basic type conversions and validations
         for col in df.columns:
             if col in ['_self_grounded', 'passed_strict_check']:
                 if df[col].dtype == 'object': df[col] = df[col].str.lower().map({'true': True, 'false': False, '': None}).astype(pd.BooleanDtype())
                 elif pd.api.types.is_bool_dtype(df[col]): df[col] = df[col].astype(pd.BooleanDtype())
             elif col not in ['gt_page_number', 'gold_steps_list']: df[col] = df[col].astype(str)

         invalid_categories = df[~df['category'].isin(VALID_CATEGORIES)]['category'].unique()
         if len(invalid_categories) > 0: logging.warning(f" -> Found unexpected category values: {list(invalid_categories)}.")
         if 'persona' in df.columns:
             invalid_personas = df[~df['persona'].isin(VALID_PERSONAS)]['persona'].unique()
             if len(invalid_personas) > 0: logging.warning(f" -> Found unexpected persona values: {list(invalid_personas)}.")

         logging.info("--- Gold QA Data Loading Complete ---")
         return df
    except Exception as e:
         logging.error(f"Failed to load or process Gold QA file {qa_file_path}: {e}", exc_info=True); sys.exit(1)


def load_pages(page_file_path: Path) -> List[Dict[str, Any]]:
    """Loads pages from the manual's JSONL file."""
    logging.info(f"--- Loading Manual Pages from: {page_file_path} ---")
    if not page_file_path.is_file():
        logging.error(f"Pages file not found: {page_file_path}"); sys.exit(1)
    pages = []
    line_num = 0
    try:
        with open(page_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line_num += 1
                line = line.strip()
                if not line: continue
                try:
                    page_data = json.loads(line)
                    page_num = page_data.get("page_num")
                    content = page_data.get("markdown_content")
                    doc_id = page_data.get("doc_id")

                    if page_num is None or not isinstance(page_num, int):
                        logging.warning(f"Skipping line {line_num}: 'page_num' missing/invalid.")
                        continue
                    if content is None or not isinstance(content, str):
                        logging.warning(f"Page {page_num} line {line_num}: 'markdown_content' missing/invalid. Treating as empty.")
                        page_data["markdown_content"] = ""
                    if doc_id is None or not isinstance(doc_id, str):
                         logging.warning(f"Page {page_num} line {line_num}: 'doc_id' missing/invalid. Using fallback.")
                         page_data["doc_id"] = page_file_path.stem
                    pages.append(page_data)
                except json.JSONDecodeError:
                    logging.warning(f"Skipping invalid JSON line {line_num}")
                    continue
        if not pages: logging.error(f"No valid pages loaded from {page_file_path}."); sys.exit(1)

        logging.info(f" -> Loaded {len(pages)} pages successfully.")
        pages.sort(key=lambda p: p.get('page_num', float('inf')))

        page_nums = [p['page_num'] for p in pages]
        if len(page_nums) != len(set(page_nums)): logging.warning(" -> Duplicate page numbers found.")

        logging.info("--- Manual Pages Loading Complete ---")
        return pages
    except Exception as e:
         logging.error(f"Failed to load pages from {page_file_path}: {e}", exc_info=True); sys.exit(1)

# =============================================================================
# --- Retriever Setup Functions ---
# =============================================================================

def setup_retrievers(pages: List[Dict], embedder_name: str, embedder_model: Any, tokenizer: Any, index_dir: Path) -> Dict[str, Any]:
    """
    Sets up the different retrievers (BM25, Dense/FAISS).
    Requires initialized embedder_model and tokenizer.
    """
    logging.info("--- Starting Retriever Setup ---")
    retrievers = {"bm25": None, "dense": None, "chunks": None, "chunk_embeddings": None} # Store chunks and embeddings
    bm25_searcher = None # Initialize searcher variable
    faiss_index = None # Initialize index variable
    all_chunks = [] # Initialize chunk list
    chunk_embeddings = None # Initialize embeddings array

    if not HAS_SPLITTER_LIBS:
        logging.error("Cannot proceed with retriever setup: Text splitter libraries not found.")
        return retrievers # Return empty retrievers

    # --- 1. Text Chunking ---
    logging.info(" -> Step 3a: Chunking document pages...")
    try:
        # Define length function based on tokenizer
        if tokenizer:
            # Use tokenizer to count tokens
            def len_func(text: str) -> int:
                # Handle potential errors during encoding just in case
                try:
                    # Use list() to ensure we get a list of token IDs for len()
                    return len(list(tokenizer.encode(text, truncation=False, add_special_tokens=False)))
                except Exception as e:
                    logging.warning(f"Tokenizer error on text: '{text[:50]}...'. Error: {e}. Falling back to len().")
                    return len(text)
            logging.info(f" -> Using tokenizer length function (Chunk: {CHUNK_SIZE_TOKENS}, Overlap: {CHUNK_OVERLAP_TOKENS}).")
        else:
            # Fallback to character count if tokenizer failed
            len_func = len
            logging.warning(f" -> Tokenizer unavailable. Using character count (Chunk: {CHUNK_SIZE_TOKENS}, Overlap: {CHUNK_OVERLAP_TOKENS}).")

        # Initialize the splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE_TOKENS, chunk_overlap=CHUNK_OVERLAP_TOKENS,
            length_function=len_func, separators=["\n\n", "\n", ". ", " ", ""], add_start_index=False
        )
        chunk_id_counter = 0
        for page in tqdm(pages, desc="Chunking Pages", disable=log_level > logging.INFO):
            page_num, doc_id, content = page.get('page_num'), page.get('doc_id'), page.get('markdown_content', '')
            if not content.strip(): continue
            page_chunks_texts = text_splitter.split_text(content)
            for i, chunk_text in enumerate(page_chunks_texts):
                chunk_id = f"{doc_id}_page{page_num}_chunk{i}"
                all_chunks.append({"id": chunk_id, "chunk_id": chunk_id, "doc_id": doc_id,
                                   "page_num": page_num, "contents": chunk_text, "text": chunk_text,
                                   "chunk_index_in_page": i})
        logging.info(f" -> Successfully created {len(all_chunks)} chunks.")
        if not all_chunks: logging.warning(" -> No chunks were created.")
        retrievers["chunks"] = all_chunks
    except Exception as e:
        logging.error(f" -> Error during text chunking: {e}", exc_info=True); return retrievers

    # --- 2. BM25 Setup ---
    if "bm25" in RETRIEVERS_TO_TEST or "hybrid" in RETRIEVERS_TO_TEST:
        logging.info(" -> Step 3b: Setting up BM25 index...")
        if HAS_PYSERINI and all_chunks:
            collection_jsonl_path = index_dir / "bm25_collection.jsonl"
            bm25_index_path = index_dir / "bm25_index"
            try:
                build_index = True
                if bm25_index_path.exists() and not FORCE_REINDEX: build_index = False; logging.info(f" -> BM25 index found at {bm25_index_path}. Skipping build.")
                elif FORCE_REINDEX and bm25_index_path.exists(): logging.info(f" -> FORCE_REINDEX=true. Removing existing BM25 index."); shutil.rmtree(bm25_index_path, ignore_errors=True)
                elif build_index and bm25_index_path.exists(): logging.info(f" -> Index needs rebuild. Removing existing BM25 index."); shutil.rmtree(bm25_index_path, ignore_errors=True)

                if build_index:
                    logging.info(f" -> Writing {len(all_chunks)} chunks to {collection_jsonl_path}...")
                    with open(collection_jsonl_path, 'w', encoding='utf-8') as f:
                        for chunk in all_chunks: f.write(json.dumps({"id": chunk.get("id"), "contents": chunk.get("contents")}, ensure_ascii=False) + '\n')
                    cmd = ["python", "-m", "pyserini.index.lucene", "--collection", "JsonCollection", "--input", str(collection_jsonl_path.parent),
                           "--index", str(bm25_index_path), "--generator", "DefaultLuceneDocumentGenerator", "--threads", str(PYSERINI_THREADS),
                           "--storePositions", "--storeDocvectors", "--storeRaw", "-language", "en"]
                    logging.info(f" -> Running Pyserini indexing...")
                    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
                    if result.returncode != 0: logging.error(f" -> Pyserini indexing failed: {result.stderr}")
                    else: logging.info(" -> Pyserini indexing completed successfully.")

                if bm25_index_path.exists():
                    try:
                        bm25_searcher = LuceneSearcher(str(bm25_index_path)); bm25_searcher.set_bm25()
                        retrievers["bm25"] = bm25_searcher
                        logging.info(" -> BM25 LuceneSearcher initialized.")
                    except Exception as e: logging.error(f" -> Failed to initialize LuceneSearcher: {e}", exc_info=True)
                else: logging.error(" -> BM25 index directory not found after build attempt.")
            except Exception as e: logging.error(f" -> An error occurred during BM25 setup: {e}", exc_info=True)
        elif not all_chunks: logging.warning(" -> Skipping BM25 setup: no chunks.")
        else: logging.warning(" -> Skipping BM25 setup: Pyserini library/Java issue.")
    else: logging.info(" -> BM25 retriever not requested.")

    # --- 3. Dense Setup ---
    if "hybrid" in RETRIEVERS_TO_TEST or "dense" in RETRIEVERS_TO_TEST:
        logging.info(" -> Step 3c: Setting up Dense (FAISS) index...")
        if HAS_FAISS and HAS_SBERT and embedder_model and all_chunks:
            faiss_index_path = index_dir / "faiss_index.idx"
            embeddings_path = index_dir / "chunk_embeddings.npy"
            try:
                faiss_index, chunk_embeddings = None, None
                if faiss_index_path.exists() and embeddings_path.exists() and not FORCE_REINDEX:
                    logging.info(f" -> Loading existing FAISS index and embeddings...")
                    faiss_index = faiss.read_index(str(faiss_index_path))
                    chunk_embeddings = np.load(str(embeddings_path))
                    if faiss_index.ntotal != len(chunk_embeddings) or faiss_index.ntotal != len(all_chunks):
                         logging.error(" -> Mismatch between loaded index/embeddings/chunks. Re-indexing."); faiss_index, chunk_embeddings = None, None
                    else: logging.info(f" -> FAISS index/embeddings loaded ({faiss_index.ntotal} vectors).")
                elif FORCE_REINDEX and (faiss_index_path.exists() or embeddings_path.exists()):
                     logging.info(" -> FORCE_REINDEX=true. Removing existing FAISS index/embeddings."); faiss_index_path.unlink(missing_ok=True); embeddings_path.unlink(missing_ok=True)

                if faiss_index is None or chunk_embeddings is None:
                    logging.info(f" -> Creating new FAISS index and embeddings...")
                    chunk_texts = [chunk['text'] for chunk in all_chunks]
                    logging.info(f" -> Embedding {len(chunk_texts)} chunks using {embedder_name}...")
                    chunk_embeddings = embedder_model.encode(chunk_texts, batch_size=EMBEDDING_BATCH_SIZE, show_progress_bar=True, normalize_embeddings=True)
                    if chunk_embeddings.dtype != np.float32: chunk_embeddings = chunk_embeddings.astype(np.float32) # Cast BEFORE saving
                    logging.info(f" -> Saving embeddings ({chunk_embeddings.shape}) to {embeddings_path}...")
                    np.save(str(embeddings_path), chunk_embeddings)
                    embedding_dim = chunk_embeddings.shape[1]
                    faiss_index = faiss.IndexFlatIP(embedding_dim)
                    logging.info(f" -> Created FAISS IndexFlatIP index (dim {embedding_dim}).")
                    faiss_index.add(chunk_embeddings)
                    logging.info(f" -> Added {faiss_index.ntotal} vectors to FAISS index.")
                    logging.info(f" -> Saving FAISS index to {faiss_index_path}...")
                    faiss.write_index(faiss_index, str(faiss_index_path))

                retrievers["dense"] = faiss_index
                retrievers["chunk_embeddings"] = chunk_embeddings
            except Exception as e: logging.error(f" -> An error occurred during Dense setup: {e}", exc_info=True)
        elif not all_chunks: logging.warning(" -> Skipping Dense setup: no chunks.")
        else:
            missing = [lib for lib, present in [("FAISS", HAS_FAISS), ("SBERT", HAS_SBERT), ("Embedder", bool(embedder_model))] if not present]
            logging.warning(f" -> Skipping Dense setup: Missing components ({', '.join(missing)}).")
    else: logging.info(" -> Dense retriever not requested.")

    logging.info("--- Retriever Setup Finished ---")
    return retrievers


# =============================================================================
# --- Core Execution Functions ---
# =============================================================================

def _cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculates cosine similarity between two normalized vectors."""
    if np.all(v1 == 0) or np.all(v2 == 0): return 0.0
    v1, v2 = np.asarray(v1).flatten(), np.asarray(v2).flatten()
    return np.dot(v1, v2)

def _mmr_rerank(query_embedding: np.ndarray, candidate_embeddings: np.ndarray,
                candidate_ids: List[int], lambda_param: float, k: int) -> List[Tuple[int, float]]:
    """Performs Maximal Marginal Relevance (MMR) re-ranking."""
    if candidate_embeddings is None or len(candidate_embeddings) == 0 or query_embedding is None:
        logging.warning("MMR input invalid."); return []
    lambda_param = max(0.0, min(1.0, lambda_param))
    n_candidates = candidate_embeddings.shape[0]
    k = min(k, n_candidates)
    if query_embedding.ndim > 1: query_embedding = query_embedding.flatten()

    relevance_scores = candidate_embeddings.dot(query_embedding)
    selected_indices_in_candidates = []
    remaining_indices_in_candidates = list(range(n_candidates))

    while len(selected_indices_in_candidates) < k:
        best_mmr_score, best_idx_in_remaining = -np.inf, -1
        for current_remaining_idx, candidate_idx in enumerate(remaining_indices_in_candidates):
            relevance = relevance_scores[candidate_idx]
            diversity = 0.0
            if selected_indices_in_candidates:
                selected_embeddings = candidate_embeddings[selected_indices_in_candidates]
                similarities_to_selected = selected_embeddings.dot(candidate_embeddings[candidate_idx])
                max_similarity = np.max(similarities_to_selected) if len(similarities_to_selected) > 0 else 0.0
                diversity = max_similarity
            mmr_score = lambda_param * relevance - (1 - lambda_param) * diversity
            if mmr_score > best_mmr_score:
                best_mmr_score, best_idx_in_remaining = mmr_score, current_remaining_idx

        if best_idx_in_remaining != -1:
            selected_candidate_idx = remaining_indices_in_candidates.pop(best_idx_in_remaining)
            selected_indices_in_candidates.append(selected_candidate_idx)
        else: break

    final_results = [(candidate_ids[idx], relevance_scores[idx]) for idx in selected_indices_in_candidates]
    return final_results


def get_context(
    query: str, retriever_type: str, retrievers: Dict[str, Any],
    full_manual_text: str, dense_embedder: SentenceTransformer,
    retrieval_params: Dict, tokenizer: Any, max_context_tokens: int # Use max_context_tokens directly
) -> Tuple[str, List[Dict]]:
    """
    Retrieves context based on the specified retriever type.
    Performs a SINGLE final truncation based on max_context_tokens.
    """
    logging.debug(f"        -> Getting context (Retriever: {retriever_type}, Max Tokens: {max_context_tokens})")

    all_chunks_list = retrievers.get("chunks", [])
    if not all_chunks_list and retriever_type != "none":
         logging.warning(f"        -> No chunks for retriever '{retriever_type}'. Returning error.")
         return "Error: No chunks available for retrieval.", []
    chunk_lookup = {chunk['id']: chunk for chunk in all_chunks_list} if all_chunks_list else {}

    context_str, retrieved_docs = "", []
    was_truncated, original_token_count = False, -1

    # --- Retrieve based on type ---
    if retriever_type == "none":
        context_str = full_manual_text
        retrieved_docs = [{"doc_id": "full_manual", "content": "N/A", "score": 1.0, "page_num": None, "chunk_id": None}]
    elif retriever_type == "bm25":
        bm25_searcher = retrievers.get("bm25")
        if bm25_searcher and chunk_lookup:
            k = retrieval_params.get('bm25_k', 10)
            try:
                hits = bm25_searcher.search(query, k=k)
                context_parts = []
                for hit in hits:
                    hit_id_str = str(hit.docid)
                    original_chunk = chunk_lookup.get(hit_id_str)
                    if original_chunk:
                        context_parts.append(original_chunk.get('text', ''))
                        retrieved_docs.append({"chunk_id": hit_id_str, "doc_id": original_chunk.get('doc_id'),
                                               "page_num": original_chunk.get('page_num'), "score": hit.score,
                                               "retriever": "bm25", "content": original_chunk.get('text', '')})
                    else: logging.warning(f"        -> BM25 hit ID '{hit_id_str}' not found.")
                context_str = "\n\n".join(context_parts)
            except Exception as e: logging.error(f"        -> Error during BM25 search: {e}", exc_info=True); context_str, retrieved_docs = f"Error: BM25 search failed: {e}", []
        else: context_str, retrieved_docs = "Error: BM25 retriever unavailable/no chunks.", []
    elif retriever_type == "dense":
        faiss_index = retrievers.get("dense")
        if faiss_index and dense_embedder and all_chunks_list:
            k = retrieval_params.get('dense_k', 10)
            try:
                query_embedding = dense_embedder.encode([query], normalize_embeddings=True).astype(np.float32)
                scores, indices = faiss_index.search(query_embedding, k=k)
                context_parts = []
                if len(indices) > 0:
                    for i, idx in enumerate(indices[0]):
                        if 0 <= idx < len(all_chunks_list):
                            original_chunk = all_chunks_list[idx]
                            context_parts.append(original_chunk.get('text', ''))
                            retrieved_docs.append({"chunk_id": original_chunk.get('id'), "doc_id": original_chunk.get('doc_id'),
                                                   "page_num": original_chunk.get('page_num'), "score": float(scores[0][i]),
                                                   "retriever": "dense", "content": original_chunk.get('text', '')})
                context_str = "\n\n".join(context_parts)
            except Exception as e: logging.error(f"        -> Error during Dense search: {e}", exc_info=True); context_str, retrieved_docs = f"Error: Dense search failed: {e}", []
        else: context_str, retrieved_docs = "Error: Dense retriever components unavailable.", []
    elif retriever_type == "hybrid":
        bm25_searcher, faiss_index, chunk_embeddings = retrievers.get("bm25"), retrievers.get("dense"), retrievers.get("chunk_embeddings")
        if bm25_searcher and faiss_index and dense_embedder and all_chunks_list and chunk_embeddings is not None:
            bm25_k, dense_k, hybrid_k = retrieval_params.get('bm25_k', 10), retrieval_params.get('dense_k', 10), retrieval_params.get('hybrid_k', 10)
            lambda_param = retrieval_params.get('mmr_lambda', 0.5)
            bm25_hit_indices, dense_hit_indices, query_embedding = set(), [], None
            try: # BM25
                bm25_hits_raw = bm25_searcher.search(query, k=bm25_k)
                bm25_hit_ids = {str(hit.docid) for hit in bm25_hits_raw}
                if bm25_hit_ids: bm25_hit_indices = {i for i, chunk in enumerate(all_chunks_list) if chunk['id'] in bm25_hit_ids}
            except Exception as e: logging.error(f"        -> Error during BM25 search (hybrid): {e}", exc_info=True)
            try: # Dense
                query_embedding = dense_embedder.encode([query], normalize_embeddings=True).astype(np.float32)
                scores, indices = faiss_index.search(query_embedding, k=dense_k)
                if len(indices) > 0: dense_hit_indices = [idx for idx in indices[0] if 0 <= idx < len(all_chunks_list)]
            except Exception as e: logging.error(f"        -> Error during Dense search (hybrid): {e}", exc_info=True)

            combined_indices = list(bm25_hit_indices.union(set(dense_hit_indices)))
            if not combined_indices: context_str, retrieved_docs = "Error: Hybrid search yielded no candidates.", []
            elif query_embedding is None: context_str, retrieved_docs = "Error: Query embedding failed for MMR.", []
            else:
                try: # MMR
                    valid_combined_indices = [idx for idx in combined_indices if 0 <= idx < len(chunk_embeddings)]
                    if not valid_combined_indices: context_str, retrieved_docs = "Error: No valid combined indices for MMR.", []
                    else:
                        candidate_embeddings_np = chunk_embeddings[valid_combined_indices].astype(np.float32)
                        if candidate_embeddings_np.ndim == 1: candidate_embeddings_np = candidate_embeddings_np.reshape(1, -1)
                        mmr_results = _mmr_rerank(query_embedding[0], candidate_embeddings_np, valid_combined_indices, lambda_param, hybrid_k)
                        context_parts = []
                        for original_index, score in mmr_results:
                            if 0 <= original_index < len(all_chunks_list):
                                original_chunk = all_chunks_list[original_index]
                                context_parts.append(original_chunk.get('text', ''))
                                retrieved_docs.append({"chunk_id": original_chunk.get('id'), "doc_id": original_chunk.get('doc_id'),
                                                       "page_num": original_chunk.get('page_num'), "score": float(score),
                                                       "retriever": "hybrid", "content": original_chunk.get('text', '')})
                        context_str = "\n\n".join(context_parts)
                        logging.info(f"        -> Hybrid retrieved {len(retrieved_docs)} docs after MMR (k={hybrid_k}).")
                except Exception as e: logging.error(f"        -> Error during Hybrid MMR/Formatting: {e}", exc_info=True); context_str, retrieved_docs = f"Error: Hybrid MMR failed: {e}", []
        else: context_str, retrieved_docs = "Error: Hybrid retriever components unavailable.", []
    else: context_str, retrieved_docs = f"Error: Unknown retriever type '{retriever_type}'.", []

    # --- Apply SINGLE Truncation Pass ---
    if tokenizer:
        try:
            tokens = list(tokenizer.encode(context_str, truncation=False, add_special_tokens=False))
            original_token_count = len(tokens)
            if original_token_count > max_context_tokens:
                logging.warning(f"        -> Context ({original_token_count} tokens) exceeds max ({max_context_tokens}). Truncating...")
                truncated_tokens = tokens[:max_context_tokens]
                context_str = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
                was_truncated = True
                logging.info(f"        -> Truncated context to ~{len(truncated_tokens)} tokens.")
            # Add truncation info even if not truncated for consistency
            if retrieved_docs and isinstance(retrieved_docs, list) and len(retrieved_docs) > 0 and isinstance(retrieved_docs[0], dict):
                 retrieved_docs[0]['context_truncated'] = was_truncated
                 retrieved_docs[0]['original_context_tokens'] = original_token_count
        except Exception as e: logging.error(f"        -> Error during context tokenization/truncation: {e}. Using untruncated.", exc_info=True); original_token_count = -1
    elif context_str: # Fallback character truncation
         max_chars = max_context_tokens * 4 # Rough estimate
         original_char_count = len(context_str)
         if original_char_count > max_chars:
              logging.warning(f"        -> Tokenizer unavailable. Truncating context chars ({original_char_count} > {max_chars}).")
              context_str = context_str[:max_chars]; was_truncated = True; original_token_count = -2
              if retrieved_docs and isinstance(retrieved_docs, list) and len(retrieved_docs) > 0 and isinstance(retrieved_docs[0], dict):
                    retrieved_docs[0]['context_truncated'] = True; retrieved_docs[0]['original_context_tokens'] = original_token_count

    logging.info(f"        -> Final context length: {len(context_str)} chars.")
    return context_str, retrieved_docs


def build_prompt_messages(
    question: str, context: str, prompt_type: str, system_prompt: str,
    gold_qa_df: pd.DataFrame, current_index: int, tokenizer: Any,
    max_prompt_tokens: int # Use max_prompt_tokens directly
) -> Tuple[List[Dict[str, str]], int]:
    """
    Builds the list of messages for the LLM API call based on prompt type.
    Includes stratified few-shot example selection and prompt truncation.
    Returns the final messages list and the calculated total token count.
    """
    logging.info(f"        -> Building prompt messages (type: {prompt_type})...")
    user_content_final_instruction = "IMPORTANT: Answer the question based *only* on the provided context. If the context lacks the answer, output the specific JSON for unanswerable questions."
    user_content = f"Context:\n```\n{context}\n```\n\nQuestion:\n```\n{question}\n```\n\n{user_content_final_instruction}\n\nOutput JSON:"

    if system_prompt is None or "ERROR" in system_prompt:
         logging.error("System prompt is missing or invalid. Using a basic fallback.")
         system_prompt = "You are a helpful assistant. Answer the question based on the context and return JSON."

    base_messages = [{"role": "system", "content": system_prompt},
                     {"role": "user", "content": user_content}] # Placeholder for final user message
    example_messages = []

    # --- Few-Shot Example Selection & Formatting ---
    if prompt_type in ["few_shot", "few_shot_cot"]:
        logging.info(f"        -> Selecting few-shot examples (strategy: {FEW_SHOT_STRATEGY}, num: {NUM_FEW_SHOT_EXAMPLES})...")
        if gold_qa_df is not None and not gold_qa_df.empty and current_index >= 0:
            possible_indices = gold_qa_df.index[gold_qa_df.index != current_index].tolist()
            num_available = len(possible_indices)
            selected_indices = []

            if num_available >= min(NUM_FEW_SHOT_EXAMPLES, 1):
                if FEW_SHOT_STRATEGY == 'stratified':
                    # (Stratified sampling logic - simplified for brevity)
                    df_possible = gold_qa_df.loc[possible_indices]
                    unans_indices = df_possible[df_possible['category'] == UNANSWERABLE_CATEGORY_NAME].index.tolist()
                    proc_indices = df_possible[df_possible['category'] == PROCEDURAL_CATEGORY_NAME].index.tolist()
                    other_indices = df_possible[~df_possible['category'].isin([UNANSWERABLE_CATEGORY_NAME, PROCEDURAL_CATEGORY_NAME])].index.tolist()
                    if unans_indices: selected_indices.append(random.choice(unans_indices))
                    if proc_indices and len(selected_indices) < NUM_FEW_SHOT_EXAMPLES: selected_indices.append(random.choice(proc_indices))
                    num_needed = NUM_FEW_SHOT_EXAMPLES - len(selected_indices)
                    if num_needed > 0 and other_indices: selected_indices.extend(random.sample(other_indices, min(num_needed, len(other_indices))))
                    num_needed = NUM_FEW_SHOT_EXAMPLES - len(selected_indices)
                    if num_needed > 0:
                        remaining_pool = [idx for idx in possible_indices if idx not in selected_indices]
                        if remaining_pool: selected_indices.extend(random.sample(remaining_pool, min(num_needed, len(remaining_pool))))
                else: # Random
                    if num_available > 0: selected_indices = random.sample(possible_indices, min(NUM_FEW_SHOT_EXAMPLES, num_available))
                logging.info(f"        -> Selected {len(selected_indices)} few-shot example indices: {selected_indices}")

                # --- Format Examples ---
                for i, example_idx in enumerate(selected_indices):
                    example_row = gold_qa_df.loc[example_idx]
                    ex_question, ex_category = example_row['question_text'], example_row['category']
                    ex_persona = example_row.get('persona', None)
                    ex_context = example_row['corrected_steps'] if ex_category == PROCEDURAL_CATEGORY_NAME else example_row['gt_answer_snippet']
                    example_user_content = f"Context:\n```\n{ex_context}\n```\n\nQuestion:\n```\n{ex_question}\n```\n\n"
                    if prompt_type == "few_shot_cot": example_user_content += f"Reasoning: [Provide step-by-step explanation...]\n\n"
                    example_user_content += f"{user_content_final_instruction}\n\nOutput JSON:"
                    example_messages.append({"role": "user", "content": example_user_content})
                    ex_answer, ex_page = None, None
                    if ex_category == UNANSWERABLE_CATEGORY_NAME: ex_persona = None
                    elif ex_category == PROCEDURAL_CATEGORY_NAME:
                        ex_answer = example_row.get('gold_steps_list', [])
                        ex_page = example_row.get('gt_page_number'); ex_page = int(ex_page) if pd.notna(ex_page) else None
                    else:
                        ex_answer = example_row['gt_answer_snippet']
                        ex_page = example_row.get('gt_page_number'); ex_page = int(ex_page) if pd.notna(ex_page) else None
                    ex_json_output = {"answer": ex_answer, "page": ex_page, "predicted_category": ex_category, "predicted_persona": ex_persona}
                    example_messages.append({"role": "assistant", "content": json.dumps(ex_json_output, ensure_ascii=False, separators=(',', ':'))})
            else: logging.warning(f"        -> Not enough examples available ({num_available}) to select {NUM_FEW_SHOT_EXAMPLES}.")
        else: logging.warning("        -> Cannot select few-shot examples: gold_qa_df missing or invalid index.")
    else:
         logging.info("        -> Using zero-shot prompt (no examples).")

    # --- Combine System, Examples, and Final User Prompt ---
    final_messages = [base_messages[0]] + example_messages + [base_messages[1]] # System + Examples + Final User

    # --- Prompt Truncation (Refined) ---
    if tokenizer:
        current_tokens = 0
        try:
            for msg in final_messages: current_tokens += len(list(tokenizer.encode(msg['content'])))
        except Exception as e: logging.error(f"        -> Error calculating initial prompt tokens: {e}. Skipping truncation."); return final_messages, -1

        logging.info(f"        -> Estimated prompt tokens before final check: {current_tokens}")

        if current_tokens > max_prompt_tokens:
            logging.warning(f"        -> Combined prompt ({current_tokens} tokens) exceeds limit ({max_prompt_tokens}). Truncating...")
            tokens_to_remove = current_tokens - max_prompt_tokens
            removed_examples = 0
            # Rebuild message list safely, removing oldest examples first
            temp_messages = [final_messages[0]] # Start with system prompt
            example_pairs = [(example_messages[i], example_messages[i+1]) for i in range(0, len(example_messages), 2)]

            while tokens_to_remove > 0 and example_pairs:
                oldest_user, oldest_asst = example_pairs.pop(0)
                try:
                    pair_tokens = len(list(tokenizer.encode(oldest_user['content']))) + len(list(tokenizer.encode(oldest_asst['content'])))
                    logging.warning(f"        -> Removing oldest example pair to save ~{pair_tokens} tokens.")
                    tokens_to_remove -= pair_tokens; removed_examples += 1
                except Exception as e: logging.error(f"        -> Error tokenizing example during truncation: {e}. Stopping."); example_pairs.insert(0, (oldest_user, oldest_asst)); break

            for user_msg, asst_msg in example_pairs: temp_messages.extend([user_msg, asst_msg])
            temp_messages.append(final_messages[-1]) # Add final user prompt back

            # Recalculate token count after removing examples
            final_token_count = sum(len(list(tokenizer.encode(msg['content']))) for msg in temp_messages)

            # If still too long, truncate the *context* within the final user message
            if final_token_count > max_prompt_tokens:
                 logging.warning(f"        -> Still too long ({final_token_count} > {max_prompt_tokens}). Truncating final user context.")
                 tokens_to_remove_from_final = final_token_count - max_prompt_tokens
                 final_user_msg_content = temp_messages[-1]['content']
                 context_start_marker, context_end_marker = "Context:\n```\n", "\n```\n\nQuestion:"
                 start_idx, end_idx = final_user_content.find(context_start_marker), final_user_content.find(context_end_marker)
                 if start_idx != -1 and end_idx != -1:
                      prefix, suffix = final_user_content[:start_idx + len(context_start_marker)], final_user_content[end_idx:]
                      original_context = final_user_content[start_idx + len(context_start_marker):end_idx]
                      try:
                           original_context_tokens = list(tokenizer.encode(original_context))
                           other_tokens = sum(len(list(tokenizer.encode(msg['content']))) for msg in temp_messages[:-1]) + len(list(tokenizer.encode(prefix + suffix)))
                           max_context_only_tokens = max(0, max_prompt_tokens - other_tokens - 50) # Extra buffer
                           keep_tokens = max(0, len(original_context_tokens) - tokens_to_remove_from_final - 50) # Alternative calc based on overflow
                           keep_tokens = min(keep_tokens, max_context_only_tokens) # Ensure we don't exceed overall limit

                           if len(original_context_tokens) > keep_tokens:
                                truncated_context_tokens = original_context_tokens[:keep_tokens]
                                truncated_context = tokenizer.decode(truncated_context_tokens, skip_special_tokens=True)
                                temp_messages[-1]['content'] = prefix + truncated_context + suffix
                                final_token_count = sum(len(list(tokenizer.encode(msg['content']))) for msg in temp_messages) # Recalculate
                                logging.warning(f"        -> Truncated final user context. New total: {final_token_count} tokens.")
                           else: logging.debug("        -> Context already small enough after example removal.")
                      except Exception as e: logging.error(f"        -> Error during final context truncation: {e}.")
                 else: logging.error("        -> Could not find context markers for final truncation.")

            final_messages = temp_messages
            logging.info(f"        -> Final prompt token count after truncation: {final_token_count} (Removed {removed_examples} examples)")
        else: # Prompt fits
             final_token_count = current_tokens
             logging.info(f"        -> Prompt fits within limit ({current_tokens} <= {max_prompt_tokens}).")

    else: # No tokenizer
        logging.warning("        -> Tokenizer unavailable. Cannot perform prompt truncation.")
        final_messages = [base_messages[0]] + example_messages + [base_messages[1]]
        final_token_count = -1 # Indicate unknown token count

    logging.debug(f"        -> Final messages structure: {[m['role'] for m in final_messages]}")
    return final_messages, final_token_count


def call_llm(model_key: str, model_details: Dict, prompt_messages: List[Dict], clients: Dict) -> str:
    """Calls the appropriate LLM API using initialized clients/config, with retries."""
    provider, api_id = model_details['provider'], model_details['api_id']
    logging.info(f"        -> Calling LLM: {model_key} (Provider: {provider}, API ID: {api_id})")
    raw_output, last_exception = None, None

    # Define retryable errors per provider
    retryable_errors = {
        "openai": (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError),
        "google": (google.api_core.exceptions.ResourceExhausted, google.api_core.exceptions.InternalServerError, google.api_core.exceptions.DeadlineExceeded),
        "groq": (GroqRateLimitError, GroqTimeoutError, GroqConnectionError),
        "ollama": (ollama.ResponseError,) # Add specific Ollama transient errors if known
    }
    # Add GroqAPIError with status >= 500 as retryable
    retryable_groq_api_error = lambda e: isinstance(e, GroqAPIError) and e.status_code >= 500

    for attempt in range(API_MAX_RETRIES + 1):
        logging.info(f"        -> LLM Call Attempt {attempt+1}/{API_MAX_RETRIES+1}")
        try:
            # --- OpenAI ---
            if provider == "openai":
                client = clients.get("openai_client")
                if client:
                    response = client.chat.completions.create(model=api_id, messages=prompt_messages, temperature=0.0, response_format={"type": "json_object"})
                    raw_output = response.choices[0].message.content; last_exception = None; break
                else: raw_output = json.dumps({"error": "OpenAI client not initialized"}); break
            # --- Google ---
            elif provider == "google":
                genai_module = clients.get("google_client")
                if genai_module and GenerationConfig:
                    system_instructions = prompt_messages[0]['content'] if prompt_messages[0]['role'] == 'system' else ""
                    start_idx = 1 if system_instructions else 0
                    gemini_contents = []
                    current_role = "user"
                    for i in range(start_idx, len(prompt_messages)):
                        msg = prompt_messages[i]; role = "model" if msg["role"] == "assistant" else "user"
                        if not gemini_contents and role == "user": gemini_contents.append({"role": role, "parts": [msg["content"]]}); current_role = "model"
                        elif role == "user" and current_role == "model": gemini_contents.append({"role": role, "parts": [msg["content"]]}); current_role = "model"
                        elif role == "model" and current_role == "user": gemini_contents.append({"role": role, "parts": [msg["content"]]}); current_role = "user"
                        else: logging.warning(f"Gemini role sequence issue: Expected {current_role}, got {role}. Appending."); gemini_contents.append({"role": role, "parts": [msg["content"]]}) # Try appending anyway
                    model = genai_module.GenerativeModel(api_id, system_instruction=system_instructions, generation_config=GenerationConfig(response_mime_type="application/json", temperature=0.0))
                    response = model.generate_content(gemini_contents)
                    try: raw_output = response.text; last_exception = None; break
                    except ValueError as e:
                        block_reason = response.prompt_feedback.block_reason if hasattr(response, 'prompt_feedback') else 'Unknown'; finish_reason = response.candidates[0].finish_reason if (response.candidates and len(response.candidates) > 0) else 'Unknown'
                        error_msg = f"Blocked/empty response ({block_reason}/{finish_reason})"; raw_output = json.dumps({"error": error_msg, "details": f"{response.prompt_feedback if hasattr(response, 'prompt_feedback') else 'N/A'}"}); last_exception = ValueError(error_msg)
                        if block_reason != 'BLOCK_REASON_UNSPECIFIED' and block_reason is not None: break
                        elif attempt < API_MAX_RETRIES: time.sleep(API_RETRY_DELAY); continue
                        else: break
                    except AttributeError as e: error_msg = f"Unexpected Gemini response structure: {e}"; raw_output = json.dumps({"error": error_msg, "details": str(response)}); last_exception = AttributeError(error_msg); break
                else: raw_output = json.dumps({"error": "Google client/config unavailable"}); break
            # --- Groq ---
            elif provider == "groq":
                client = clients.get("groq_client")
                if client:
                     response = client.chat.completions.create(model=api_id, messages=prompt_messages, temperature=0.0, response_format={"type": "json_object"})
                     raw_output = response.choices[0].message.content; last_exception = None; break
                else: raw_output = json.dumps({"error": "Groq client not initialized"}); break
            # --- Ollama ---
            elif provider == "ollama":
                client = clients.get("ollama_client")
                if client and ollama:
                    try:
                        response = client.chat(model=api_id, messages=prompt_messages, stream=False, options={"temperature": 0.0}, format="json")
                        if isinstance(response, dict) and 'message' in response and 'content' in response['message']:
                            raw_output = response['message']['content']; last_exception = None; break
                        else: error_msg = "Ollama response structure invalid"; raw_output = json.dumps({"error": error_msg, "details": str(response)}); last_exception = ValueError(error_msg); break
                    except ollama.ResponseError as e:
                        last_exception = e; logging.error(f"        -> Ollama API error: Status {e.status_code}, Error: {e.error}")
                        is_retryable = isinstance(e, retryable_errors_ollama) or (hasattr(e, 'status_code') and e.status_code >= 500 and e.status_code != 501)
                        if is_retryable and attempt < API_MAX_RETRIES: logging.warning(f"        -> Retrying Ollama error..."); time.sleep(API_RETRY_DELAY); continue
                        else: raw_output = json.dumps({"error": f"Ollama API error {e.status_code}", "details": e.error}); break
                    except Exception as e: # Catch other potential errors like connection errors
                        last_exception = e; logging.error(f"        -> Unexpected Ollama error: {e}", exc_info=True)
                        is_retryable = isinstance(e, retryable_errors_ollama) # Check if it's a known retryable type
                        if is_retryable and attempt < API_MAX_RETRIES: logging.warning(f"        -> Retrying Ollama error..."); time.sleep(API_RETRY_DELAY); continue
                        else: raw_output = json.dumps({"error": f"Unexpected Ollama error: {e}"}); break
                else: raw_output = json.dumps({"error": "Ollama client unavailable/library missing"}); break
            # --- Unknown Provider ---
            else: raw_output = json.dumps({"error": f"Unknown provider: {provider}"}); break

        # --- Specific API Error Handling with Retries ---
        except retryable_errors.get(provider, ()) as e: # Use provider specific retryable errors
            last_exception = e; error_type = type(e).__name__
            logging.warning(f"        -> {error_type} (Attempt {attempt+1}): {e}. Waiting {API_RETRY_DELAY}s...")
            if attempt < API_MAX_RETRIES: time.sleep(API_RETRY_DELAY); continue
            else: logging.error(f"        -> {error_type} persisted after retries."); raw_output = json.dumps({"error": f"{error_type} after retries: {e}"}); break
        # FIX: Specific check for Groq 5xx errors wrapped in GroqAPIError
        except GroqAPIError as e:
             last_exception = e; error_type = type(e).__name__
             if hasattr(e, 'status_code') and e.status_code >= 500 and attempt < API_MAX_RETRIES:
                 logging.warning(f"        -> {error_type} (Status {e.status_code}) (Attempt {attempt+1}): {e}. Waiting {API_RETRY_DELAY}s...")
                 time.sleep(API_RETRY_DELAY); continue
             else:
                 logging.error(f"        -> {error_type} (Status {getattr(e, 'status_code', 'N/A')}) (Attempt {attempt+1}): {e}. Not retrying or retries exhausted.")
                 raw_output = json.dumps({"error": f"{error_type}: {e}"}); break
        # --- Handle other API Errors (Non-retryable by default) ---
        except (openai.APIError, google.api_core.exceptions.GoogleAPIError) as e: # Removed GroqAPIError here
            last_exception = e; error_type = type(e).__name__
            logging.error(f"        -> {error_type} (Attempt {attempt+1}): {e}. Not retrying.")
            raw_output = json.dumps({"error": f"{error_type}: {e}"}); break
        # --- Generic Error Handling ---
        except Exception as e:
             last_exception = e
             logging.error(f"        -> Unexpected Error during LLM call (Attempt {attempt+1}): {e}", exc_info=True)
             raw_output = json.dumps({"error": f"Unexpected error: {e}"}); break

    # --- Final Output Check ---
    if raw_output is None:
        logging.error(f"        -> LLM call failed after all retries. Last exception: {last_exception}")
        raw_output = json.dumps({"error": f"LLM call failed after retries: {last_exception}"})
    elif last_exception is None:
         logging.info(f"        -> LLM call successful (Attempt {attempt+1}).")

    # --- API Delay ---
    if last_exception is None and provider != "ollama": time.sleep(API_DELAY_SECONDS)

    logging.debug(f"        -> LLM Raw Output (first 100 chars): {raw_output[:100]}...")
    return raw_output


def parse_llm_json_output(llm_output_str: str, model_key: str) -> Tuple[bool, Dict]: # Added model_key parameter
    """Parses the LLM's raw string output, expecting JSON. Logs model key on failure."""
    # Add model key to the initial debug log if desired
    logging.debug(f"        -> Attempting to parse LLM output for model '{model_key}': {llm_output_str[:200]}...")
    if not isinstance(llm_output_str, str) or not llm_output_str.strip():
        # Add model key to warning log
        logging.warning(f"        -> [{model_key}] LLM output is empty or not a string.")
        return False, {"error": "Empty or non-string output", "raw_output": llm_output_str}
    try:
        # Clean potential markdown fences (more robustly)
        match = re.search(r'```(json)?\s*(\{.*?\})\s*```', llm_output_str, re.DOTALL | re.IGNORECASE)
        if match:
            json_str = match.group(2)
            logging.debug("        -> Extracted JSON from markdown fences.")
        else:
            # Assume the whole string is JSON if no fences found
            json_str = llm_output_str.strip()

        parsed_json = json.loads(json_str)
        logging.info(f"        -> [{model_key}] Successfully parsed LLM output as JSON.") # Add model key on success too
        return True, parsed_json
    except json.JSONDecodeError as e:
        # Add model key to warning log
        logging.warning(f"        -> [{model_key}] Failed to parse LLM output as JSON: {e}. Raw: {llm_output_str[:500]}...")
        return False, {"error": f"JSONDecodeError: {e}", "raw_output": llm_output_str}
    except Exception as e:
        # Add model key to error log
        logging.error(f"        -> [{model_key}] Unexpected error parsing LLM output: {e}", exc_info=True)
        return False, {"error": f"Unexpected parsing error: {e}", "raw_output": llm_output_str}


# =============================================================================
# --- Main Execution Block ---
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG/LLM benchmark grid for one manual.")
    parser.add_argument("-i", "--input_dir", type=pathlib.Path, required=True, help="Path to manual's processed files dir")
    parser.add_argument("-o", "--outdir", type=pathlib.Path, required=True, help="Base output directory for results")
    # Add verbose flag
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG level logging")
    args = parser.parse_args()

    # Adjust log level if verbose flag is set
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.info("Verbose logging enabled (DEBUG level).")

    logging.info("==============================================================")
    logging.info(f"--- Starting Benchmark Run for Manual in: {args.input_dir} ---")
    logging.info("==============================================================")

    # --- Set Random Seed ---
    random.seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)
    logging.info(f"Set random seed to: {RANDOM_SEED}")

    # --- Validate Input & Setup Output ---
    if not args.input_dir.is_dir(): logging.error(f"Input dir not found: {args.input_dir}"); sys.exit(1)
    manual_name = args.input_dir.name
    logging.info(f"Manual Name: {manual_name}")
    pages_file_path = args.input_dir / f"{manual_name}_pages.jsonl" # Adjusted filename
    qa_file_path = args.input_dir / f"{manual_name}_gold.csv"
    if not pages_file_path.is_file(): logging.error(f"Pages file not found: {pages_file_path}"); sys.exit(1)
    if not qa_file_path.is_file(): logging.error(f"Gold QA file not found: {qa_file_path}"); sys.exit(1)

    run_output_dir = args.outdir / manual_name
    index_dir = run_output_dir / "indices"
    results_dir = run_output_dir / "results"
    output_pickle_file = results_dir / f"{manual_name}_benchmark_results.pkl"
    try:
        run_output_dir.mkdir(parents=True, exist_ok=True)
        index_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory for this run: {run_output_dir}")
        print(f"Detailed results will be saved to: {output_pickle_file}")
    except OSError as e: logging.error(f"Could not create output dirs: {e}"); sys.exit(1)

    # --- Initialize Components ---
    print("\n--- Step 2: Initializing Components ---")
    initialized_components = initialize_clients_and_models(MODELS_TO_TEST, DENSE_RETRIEVAL_EMBEDDER, SEMANTIC_SIMILARITY_EMBEDDER)
    clients = {
        "openai_client": initialized_components["openai_client"],
        "google_client": initialized_components["google_client"],
        "groq_client": initialized_components["groq_client"],
        "ollama_client": initialized_components["ollama_client"] # Add Ollama client
    }
    dense_embedder = initialized_components["dense_embedder"]
    tokenizer = initialized_components["tokenizer"]
    system_prompt = initialized_components["system_prompt"]
    print("--- Component Initialization Complete ---")

    # --- Load Data ---
    print("\n--- Step 1: Loading Data ---")
    gold_qa_df = load_gold_qa(qa_file_path)
    pages_list = load_pages(pages_file_path)
    full_manual_text = "\n\n".join([f"--- Page {p['page_num']} ---\n{p.get('markdown_content', '')}" for p in pages_list])
    print(f"Created full manual text string ({len(full_manual_text)} chars).")
    print("--- Data Loading Complete ---")

    # --- Setup Retrievers ---
    print("\n--- Step 3: Setting up Retrievers ---")
    retrievers = setup_retrievers(pages_list, DENSE_RETRIEVAL_EMBEDDER, dense_embedder, tokenizer, index_dir)
    all_chunks = retrievers.get("chunks", [])
    bm25_ready = retrievers.get("bm25") is not None
    dense_ready = retrievers.get("dense") is not None
    print(f"--- Retriever Setup Complete (Chunks: {len(all_chunks)}, BM25: {bm25_ready}, Dense: {dense_ready}) ---")

    # --- Main Benchmark Loop ---
    total_conditions = len(MODELS_TO_TEST) * len(RETRIEVERS_TO_TEST) * len(PROMPTS_TO_TEST)
    total_questions = len(gold_qa_df)
    print(f"\n--- Step 4: Starting Benchmark Run ({total_questions} questions x {total_conditions} conditions) ---")
    all_results = []

    for index, gold_row in tqdm(gold_qa_df.iterrows(), total=total_questions, desc="Benchmarking Questions", disable=log_level > logging.INFO):
        question_id, question_text, gold_category = gold_row['question_id'], gold_row['question_text'], gold_row['category']
        logging.info(f"\n--- Processing QID: {question_id} (Index: {index}, Category: {gold_category}) ---")
        logging.info(f" -> Question: {question_text}")

        for retriever_type in RETRIEVERS_TO_TEST:
            logging.info(f" --> Retriever: {retriever_type}")
            if (retriever_type == "bm25" and not bm25_ready) or \
               (retriever_type == "dense" and not dense_ready) or \
               (retriever_type == "hybrid" and (not bm25_ready or not dense_ready)):
                 logging.warning(f"    Retriever '{retriever_type}' unavailable. Skipping."); continue

            for model_key, model_details in MODELS_TO_TEST.items():
                logging.info(f" ----> Model: {model_key}")
                provider = model_details['provider']
                client_key = f"{provider}_client" if provider != 'ollama' else 'ollama_client'
                if not clients.get(client_key):
                     logging.warning(f"      Skipping model '{model_key}' (provider '{provider}' client unavailable)."); continue
                model_context_window = model_details.get('context_window', 8192)

                # Calculate Max Context Tokens for get_context
                # This depends on the prompt structure which varies slightly
                # Estimate non-context tokens roughly here, refine in build_prompt
                estimated_non_context = len(list(tokenizer.encode(system_prompt))) + 500 # Rough estimate for question, examples, instructions
                max_context_tokens_for_retrieval = model_context_window - estimated_non_context - LLM_RESPONSE_BUFFER_TOKENS
                if max_context_tokens_for_retrieval <= 0:
                    logging.error(f"Model context window {model_context_window} too small for estimated prompt overhead + response buffer. Skipping context retrieval for {model_key}.")
                    continue

                logging.info(f"        -> Max tokens for retrieved context: {max_context_tokens_for_retrieval}")
                context_str, retrieved_sources = get_context(
                    question_text, retriever_type, retrievers, full_manual_text,
                    dense_embedder, RETRIEVAL_PARAMS, tokenizer, max_context_tokens_for_retrieval
                )

                for prompt_type in PROMPTS_TO_TEST:
                    logging.info(f" ------> Prompt: {prompt_type}")
                    trial_info = {
                        "manual_id": manual_name, "question_id": question_id, "gold_question_text": question_text,
                        "gold_category": gold_category, "gold_answer_snippet": gold_row.get('gt_answer_snippet'),
                        "gold_page_number": gold_row.get('gt_page_number'), "gold_steps_list": gold_row.get('gold_steps_list'),
                        "gold_persona": gold_row.get('persona'), "condition_model": model_key,
                        "condition_retriever": retriever_type, "condition_prompt": prompt_type,
                        "retrieved_context": context_str, "retrieved_sources": retrieved_sources,
                        "llm_prompt_messages": None, "llm_prompt_total_tokens": -1,
                        "llm_raw_output": None, "llm_parsed_output": None, "json_parsable": None,
                        "error_message": None, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    try:
                        # Build final prompt messages (includes truncation)
                        prompt_messages, final_token_count = build_prompt_messages(
                            question_text, context_str, prompt_type, system_prompt,
                            gold_qa_df, index, tokenizer, model_context_window
                        )
                        trial_info["llm_prompt_messages"] = prompt_messages # Store potentially truncated messages
                        trial_info["llm_prompt_total_tokens"] = final_token_count

                        # Call LLM
                        llm_output_raw = call_llm(model_key, model_details, prompt_messages, clients)
                        trial_info["llm_raw_output"] = llm_output_raw

                        # Parse LLM Output
                        parsable, parsed_output = parse_llm_json_output(llm_output_raw, model_key)
                        trial_info["json_parsable"], trial_info["llm_parsed_output"] = parsable, parsed_output

                    except Exception as e:
                         logging.error(f"      ****** ERROR during trial QID {question_id}, {model_key}, {retriever_type}, {prompt_type}: {e} ******", exc_info=log_level <= logging.DEBUG)
                         trial_info["error_message"] = f"{type(e).__name__}: {e}"

                    all_results.append(trial_info)
                    logging.info(f"        -> Trial Complete. Parsable: {trial_info['json_parsable']}. Error: {trial_info['error_message']}")

            logging.info(f" ----> Finished all prompts for Model: {model_key}")
        logging.info(f" --> Finished all models/prompts for Retriever: {retriever_type}")
    logging.info(f"--- Finished processing all questions for QID: {question_id} ---")


    logging.info(f"\n--- Benchmark Run Complete. Total trials executed: {len(all_results)} ---")

    # --- Save Results ---
    logging.info(f"\n--- Step 5: Saving Detailed Results ---")
    try:
        results_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Saving {len(all_results)} results to: {output_pickle_file}")
        with open(output_pickle_file, 'wb') as f:
            pickle.dump(all_results, f, protocol=pickle.HIGHEST_PROTOCOL)
        logging.info(f"Successfully saved results.")
        if output_pickle_file.exists() and output_pickle_file.stat().st_size > 100 * 1024 * 1024:
             logging.warning(f"Output file {output_pickle_file} is large. Consider JSONL or compression.")
    except Exception as e:
        logging.error(f"Failed to save results: {e}", exc_info=True)

    logging.info(f"\n--- run_benchmark_manual_v2.py finished for manual '{manual_name}' ---")
    logging.info("==============================================================")

