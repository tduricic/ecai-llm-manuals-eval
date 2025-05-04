#!/usr/bin/env python3
# ==============================================================
# run_benchmark_manual.py
# ==============================================================
# Runs the RAG/LLM benchmark grid (Models x Retrievers x Prompts)
# for a single technical manual using a pre-processed gold QA dataset.
# Saves detailed results per condition (input, context, raw output)
# to allow for separate scoring later.
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
    from groq import Groq as GroqClient # Renamed to avoid conflict with module name
    from dotenv import load_dotenv
    HAS_LLM_APIS = True
except ImportError:
     logging.warning("LLM API clients (openai, google-generativeai, groq) or python-dotenv not found. LLM calls will fail.")
     HAS_LLM_APIS = False; openai = None; genai = None; GroqClient = None; load_dotenv = None

# Metrics (Optional components - Imports removed as scoring is deferred)

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
    # Retrievers and Prompts to Test (Defined in config)
    RETRIEVERS_TO_TEST = config_bench['retrievers_to_test']
    PROMPTS_TO_TEST = config_bench['prompts_to_test']
    # Embedding Models (Defined in config)
    DENSE_RETRIEVAL_EMBEDDER = config_bench['dense_retriever_embedder']
    SEMANTIC_SIMILARITY_EMBEDDER = config_bench['semantic_similarity_embedder'] # Used in scoring script
    # Retrieval & Scoring Params (Defined in config)
    RETRIEVAL_PARAMS = config_bench['retrieval_params']
    # Chunking parameters
    CHUNK_SIZE_TOKENS = RETRIEVAL_PARAMS.get('chunk_size_tokens', 512) # Default 512
    CHUNK_OVERLAP_TOKENS = RETRIEVAL_PARAMS.get('chunk_overlap_tokens', 64) # Default 64
    logging.info(f"Using Chunk Size: {CHUNK_SIZE_TOKENS} tokens, Overlap: {CHUNK_OVERLAP_TOKENS} tokens")
    # BM25 params from config
    BM25_K = RETRIEVAL_PARAMS.get('bm25_k', 10)
    # Dense params from config
    DENSE_K = RETRIEVAL_PARAMS.get('dense_k', 10)
    # Hybrid params from config
    HYBRID_K = RETRIEVAL_PARAMS.get('hybrid_k', 10)
    MMR_LAMBDA = RETRIEVAL_PARAMS.get('mmr_lambda', 0.5)
    # Pyserini indexing threads
    PYSERINI_THREADS = RETRIEVAL_PARAMS.get('pyserini_threads', 4) # Add to config if needed
    # Embedding batch size
    EMBEDDING_BATCH_SIZE = RETRIEVAL_PARAMS.get('embedding_batch_size', 32) # Add to config if needed

    SCORING_PARAMS = config_bench['scoring_params'] # Some params might still be useful (e.g., page tolerance)
    # API Params (Defined in config)
    API_DELAY_SECONDS = config_bench['api_params']['api_delay_seconds']
    # Files (Get system prompt path)
    SYSTEM_PROMPT_PATH_STR = config_bench['files']['system_prompt']
    SYSTEM_PROMPT_PATH = SCRIPT_DIR.parent / SYSTEM_PROMPT_PATH_STR # Construct full path

    # Constants defined directly in script (could also be moved to config)
    VALID_CATEGORIES = [ # Should match Phase 1 CATEGORY_TARGETS keys and system_instructions.md
        "Specification Lookup", "Tool/Material Identification",
        "Procedural Step Inquiry", "Location/Definition",
        "Conditional Logic/Causal Reasoning", "Safety Information Lookup",
        "Unanswerable"
    ]
    VALID_PERSONAS = ["Novice User", "Technician", "SafetyOfficer"]
    # Phase 1 category name for procedural
    PROCEDURAL_CATEGORY_NAME = "Procedural Step Inquiry"

except KeyError as e:
     logging.error(f"Missing key in benchmark configuration file {CONFIG_PATH}: {e}"); sys.exit(1)
# -----------------------------------------------------------------------------

# --- Setup basic logging ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(lineno)d: %(message)s')

# =============================================================================
# --- Helper Function for Parsing List Strings ---
# =============================================================================

def safe_parse_list_string(list_str: str, default_value=None):
    """Safely parses a string representation of a list (e.g., "['a', 'b']") into a Python list."""
    if pd.isna(list_str) or not isinstance(list_str, str) or not list_str.strip():
        return default_value
    try:
        # Ensure it looks like a list before trying to parse
        if list_str.startswith('[') and list_str.endswith(']'):
            parsed = ast.literal_eval(list_str)
            if isinstance(parsed, list):
                 # For steps, we expect strings inside
                 if all(isinstance(item, str) for item in parsed):
                     return parsed
                 else:
                     logging.warning(f"Parsed list contains non-string elements: {list_str[:100]}... Returning default.")
                     return default_value
            else:
                 logging.warning(f"Parsed literal is not a list: {list_str[:100]}... Returning default.")
                 return default_value
        else:
            # Handle cases where it might be an empty string representing an empty list by convention
            if list_str == '[]': return []
            logging.warning(f"String does not look like a list literal: {list_str[:100]}... Returning default.")
            return default_value
    except (ValueError, SyntaxError, TypeError, MemoryError) as e: # Added MemoryError just in case
        logging.warning(f"Could not parse list string using ast.literal_eval: '{list_str[:100]}...'. Error: {e}. Returning default.")
        return default_value

# =============================================================================
# --- Initialization Functions ---
# =============================================================================

def initialize_clients_and_models(models_config: Dict, dense_embedder_name: str, semantic_embedder_name: str) -> Dict:
    """
    Initializes API clients (OpenAI, Google, Groq) and Sentence Transformer models.
    Loads API keys from environment variables.
    """
    initialized_components = {
        "openai_client": None,
        "google_client": None,
        "groq_client": None,
        "dense_embedder": None,
        "semantic_embedder": None, # Keep for potential future use or simple checks
        "system_prompt": None,
        "tokenizer": None # Add tokenizer here
    }
    api_keys_loaded = {}

    # --- Load API Keys ---
    if HAS_LLM_APIS and load_dotenv:
        logging.info("Attempting to load API keys from .env file...")
        if load_dotenv():
            logging.info(" -> .env file loaded.")
        else:
            logging.info(" -> .env file not found. Relying on environment variables.")

        api_keys_loaded['openai'] = os.getenv("OPENAI_API_KEY")
        api_keys_loaded['google'] = os.getenv("GOOGLE_API_KEY")
        api_keys_loaded['groq'] = os.getenv("GROQ_API_KEY")
    else:
        logging.warning("LLM API libraries or python-dotenv not available. Cannot load keys.")

    # --- Determine Needed Providers ---
    providers_needed = set(details['provider'] for details in models_config.values())
    logging.info(f"Providers needed based on config: {providers_needed}")

    # --- Initialize OpenAI Client ---
    if 'openai' in providers_needed:
        if HAS_LLM_APIS and openai:
            if api_keys_loaded.get('openai'):
                try:
                    initialized_components["openai_client"] = openai.OpenAI(api_key=api_keys_loaded['openai'])
                    logging.info("OpenAI client initialized successfully.")
                except Exception as e:
                    logging.error(f"Failed to initialize OpenAI client: {e}", exc_info=True)
            else:
                logging.warning("OpenAI provider needed, but OPENAI_API_KEY not found in environment.")
        else:
            logging.warning("OpenAI provider needed, but 'openai' library not imported.")

    # --- Initialize Google Client ---
    if 'google' in providers_needed:
        if HAS_LLM_APIS and genai:
            if api_keys_loaded.get('google'):
                try:
                    genai.configure(api_key=api_keys_loaded['google'])
                    # We typically initialize the model later with genai.GenerativeModel(model_id)
                    # Store the configured module itself or a flag indicating configuration success
                    initialized_components["google_client"] = genai # Store the configured module
                    logging.info("Google GenAI configured successfully.")
                except Exception as e:
                    logging.error(f"Failed to configure Google GenAI: {e}", exc_info=True)
            else:
                logging.warning("Google provider needed, but GOOGLE_API_KEY not found in environment.")
        else:
            logging.warning("Google provider needed, but 'google-generativeai' library not imported.")

    # --- Initialize Groq Client ---
    if 'groq' in providers_needed:
        if HAS_LLM_APIS and GroqClient:
            if api_keys_loaded.get('groq'):
                try:
                    initialized_components["groq_client"] = GroqClient(api_key=api_keys_loaded['groq'])
                    logging.info("Groq client initialized successfully.")
                except Exception as e:
                    logging.error(f"Failed to initialize Groq client: {e}", exc_info=True)
            else:
                logging.warning("Groq provider needed, but GROQ_API_KEY not found in environment.")
        else:
            logging.warning("Groq provider needed, but 'groq' library not imported.")

    # --- Initialize Sentence Transformer Models ---
    if HAS_SBERT and SentenceTransformer:
        # Dense Embedder (for retrieval)
        if dense_embedder_name:
            try:
                logging.info(f"Initializing Dense Embedding model: {dense_embedder_name}")
                initialized_components["dense_embedder"] = SentenceTransformer(dense_embedder_name)
                logging.info(" -> Dense Embedder initialized.")
                # --- Initialize Tokenizer using the name of the dense embedder ---
                if HAS_SPLITTER_LIBS and AutoTokenizer:
                    try:
                        logging.info(f"Initializing Tokenizer for: {dense_embedder_name}")
                        # Use the same name/path as the embedder model
                        initialized_components["tokenizer"] = AutoTokenizer.from_pretrained(dense_embedder_name)
                        logging.info(" -> Tokenizer initialized.")
                    except Exception as e:
                        logging.error(f"Failed to initialize Tokenizer '{dense_embedder_name}': {e}. Chunking may use character count.", exc_info=True)
                else:
                    logging.warning("Transformers library not available. Cannot initialize tokenizer.")

            except Exception as e:
                logging.error(f"Failed to initialize Dense Embedding model '{dense_embedder_name}': {e}", exc_info=True)
        else:
            logging.warning("No dense_retriever_embedder specified in config. Cannot initialize tokenizer for chunking.")

        # Semantic Embedder (usually for scoring, but initialize anyway if specified)
        if semantic_embedder_name:
             try:
                 logging.info(f"Initializing Semantic Similarity model: {semantic_embedder_name}")
                 initialized_components["semantic_embedder"] = SentenceTransformer(semantic_embedder_name)
                 logging.info(" -> Semantic Embedder initialized.")
             except Exception as e:
                 logging.error(f"Failed to initialize Semantic Similarity model '{semantic_embedder_name}': {e}", exc_info=True)
        else:
             logging.warning("No semantic_similarity_embedder specified in config.")

    else:
        logging.warning("Sentence Transformers library not available. Cannot initialize embedding models or tokenizer.")

    # --- Load System Prompt ---
    try:
        logging.info(f"Loading system prompt from: {SYSTEM_PROMPT_PATH}")
        if SYSTEM_PROMPT_PATH.is_file():
            initialized_components["system_prompt"] = SYSTEM_PROMPT_PATH.read_text(encoding='utf-8')
            logging.info(f" -> System prompt loaded ({len(initialized_components['system_prompt'])} chars).")
        else:
            # Treat missing system prompt as fatal error
            logging.error(f"FATAL: System prompt file not found at specified path: {SYSTEM_PROMPT_PATH}")
            logging.error("Please ensure the file exists and the path in 'config/settings_benchmark.yaml' is correct.")
            sys.exit(1) # Exit the script
    except Exception as e:
        # Treat any error during loading as fatal
        logging.error(f"FATAL: Failed to load system prompt from {SYSTEM_PROMPT_PATH}: {e}", exc_info=True)
        sys.exit(1) # Exit the script

    return initialized_components

# =============================================================================
# --- Data Loading Functions ---
# =============================================================================

def load_gold_qa(qa_file_path: Path) -> pd.DataFrame:
    """Loads and validates the Phase 2 Gold Standard QA CSV file."""
    logging.info(f"Loading Gold QA data from: {qa_file_path}")
    if not qa_file_path.is_file():
        logging.error(f"Gold QA file not found: {qa_file_path}")
        sys.exit(1)

    # Define ALL columns expected in the final _gold.csv after Phase 1 finalize
    # This ensures we load everything potentially needed for context or later analysis
    expected_cols = [
        "question_id", "persona", "doc_id", "language", "question_text",
        "category", "gt_answer_snippet", "gt_page_number", "_self_grounded",
        "parsed_steps", "passed_strict_check", "corrected_steps", "procedural_comments"
    ]

    try:
         # keep_default_na=False prevents pandas interpreting "NA" or "" as NaN automatically
         # Use low_memory=False for potentially mixed types or large files
         df = pd.read_csv(qa_file_path, keep_default_na=False, low_memory=False)
         logging.info(f"Loaded {len(df)} rows from Gold QA file: {qa_file_path.name}")

         if df.empty:
              logging.error(f"Gold QA file is empty: {qa_file_path}")
              sys.exit(1)

         # --- Column Validation ---
         missing = [col for col in expected_cols if col not in df.columns]
         if missing:
              # Provide a more informative error message
              logging.error(f"Gold QA file {qa_file_path.name} is missing expected columns generated by Phase 1 '--finalize' step: {missing}.")
              logging.error(f"Ensure the Phase 1 script was run with '--finalize' and completed successfully.")
              logging.error(f"Available columns are: {list(df.columns)}")
              sys.exit(1)
         else:
              logging.info(f"All expected columns found in {qa_file_path.name}.")

         # --- Data Type Conversion and Parsing ---
         # Convert gt_page_number: handle 'None' string, empty strings, then to nullable Int
         df['gt_page_number'] = df['gt_page_number'].replace(['None', ''], np.nan)
         df['gt_page_number'] = pd.to_numeric(df['gt_page_number'], errors='coerce').astype(pd.Int64Dtype())

         # Parse 'parsed_steps' column (should contain the corrected steps as string list from finalize)
         logging.info("Parsing 'parsed_steps' column (containing final gold steps) into Python lists...")
         # Apply the safe parsing function. Default to empty list [] if parsing fails or original is empty/NaN/invalid
         df['gold_steps_list'] = df['parsed_steps'].apply(lambda x: safe_parse_list_string(x, default_value=[]))
         parsed_count = df['gold_steps_list'].apply(lambda x: isinstance(x, list)).sum()
         none_count = df['gold_steps_list'].isna().sum() # Should be 0 if default_value=[]
         if none_count > 0:
             logging.warning(f"{none_count} rows resulted in None for 'gold_steps_list'. Check 'safe_parse_list_string' default.")
         logging.info(f"Successfully parsed 'parsed_steps' for {parsed_count}/{len(df)} rows into 'gold_steps_list'. Others set to default (likely []).")

         # Diagnostic: Check if procedural questions actually got non-empty lists
         proc_mask = df['category'] == PROCEDURAL_CATEGORY_NAME
         num_procedural = proc_mask.sum()
         if num_procedural > 0:
            proc_with_steps = df.loc[proc_mask, 'gold_steps_list'].apply(lambda x: isinstance(x, list) and len(x) > 0).sum()
            logging.info(f"  -> Found {proc_with_steps} / {num_procedural} '{PROCEDURAL_CATEGORY_NAME}' rows with non-empty gold step lists.")
         else:
             logging.info(f"  -> No rows found with category '{PROCEDURAL_CATEGORY_NAME}'.")

         # Ensure other key text columns are explicitly strings, handle boolean cols
         for col in df.columns:
             if col in ['_self_grounded', 'passed_strict_check']:
                 # Convert boolean-like strings ('True', 'False') to actual booleans
                 # Handle potential mixed types or actual booleans already present
                 if df[col].dtype == 'object': # Only convert if it's string/object type
                     df[col] = df[col].str.lower().map({'true': True, 'false': False, '': None}).astype(pd.BooleanDtype())
                 elif pd.api.types.is_bool_dtype(df[col]): # If already boolean, ensure it's nullable Boolean
                      df[col] = df[col].astype(pd.BooleanDtype())
             elif col not in ['gt_page_number', 'gold_steps_list']: # Avoid re-casting special types
                 # Ensure all other potentially relevant columns are string
                 df[col] = df[col].astype(str)

         # --- Final Validation ---
         # Check if category values are valid
         invalid_categories = df[~df['category'].isin(VALID_CATEGORIES)]['category'].unique()
         if len(invalid_categories) > 0:
             logging.warning(f"Found rows with unexpected category values: {list(invalid_categories)}. Ensure VALID_CATEGORIES constant is up-to-date.")
         # Check if persona values are valid (if column exists)
         if 'persona' in df.columns:
             invalid_personas = df[~df['persona'].isin(VALID_PERSONAS)]['persona'].unique()
             if len(invalid_personas) > 0:
                 logging.warning(f"Found rows with unexpected persona values: {list(invalid_personas)}. Ensure VALID_PERSONAS constant is up-to-date.")

         return df

    except Exception as e:
         logging.error(f"Failed to load or process Gold QA file {qa_file_path}: {e}", exc_info=True)
         sys.exit(1)


def load_pages(page_file_path: Path) -> List[Dict[str, Any]]:
    """Loads pages from the manual's JSONL file."""
    logging.info(f"Loading Pages data from: {page_file_path}")
    if not page_file_path.is_file():
        logging.error(f"Pages file not found: {page_file_path}")
        sys.exit(1)

    pages = []
    line_num = 0
    try:
        with open(page_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line_num += 1
                line = line.strip()
                if not line: continue # Skip empty lines
                try:
                    page_data = json.loads(line)
                    # Basic validation of structure and types
                    page_num = page_data.get("page_num")
                    content = page_data.get("markdown_content")
                    doc_id = page_data.get("doc_id") # Keep doc_id for potential use

                    if page_num is None or not isinstance(page_num, int):
                        logging.warning(f"Skipping line {line_num} in {page_file_path}: 'page_num' missing, not integer, or null.")
                        continue
                    if content is None or not isinstance(content, str):
                        # Allow empty content, but log if it's not a string type
                        if content is not None:
                             logging.warning(f"Page {page_num} line {line_num} in {page_file_path}: 'markdown_content' is not string type ({type(content)}). Treating as empty.")
                             page_data["markdown_content"] = "" # Standardize to empty string
                        else: # If None, also standardize
                             page_data["markdown_content"] = ""

                    # Keep doc_id if present, default otherwise
                    if doc_id is None or not isinstance(doc_id, str):
                         logging.warning(f"Page {page_num} line {line_num} in {page_file_path}: 'doc_id' missing or not string. Using fallback.")
                         page_data["doc_id"] = page_file_path.stem # Use file stem as fallback doc_id

                    pages.append(page_data)
                except json.JSONDecodeError:
                    logging.warning(f"Skipping invalid JSON line {line_num} in {page_file_path}")
                    continue
        if not pages:
            logging.error(f"No valid pages loaded from {page_file_path}.")
            sys.exit(1)

        logging.info(f"Loaded {len(pages)} pages successfully from {page_file_path}.")
        # Sort by page number for predictable order
        pages.sort(key=lambda p: p.get('page_num', float('inf'))) # Put pages without num at end

        # Check for duplicate page numbers
        page_nums = [p['page_num'] for p in pages]
        if len(page_nums) != len(set(page_nums)):
             logging.warning(f"Duplicate page numbers found in {page_file_path}. Ensure source processing is correct.")

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
    logging.info("Starting retriever setup...")
    retrievers = {"bm25": None, "dense": None, "chunks": None, "chunk_embeddings": None} # Store chunks and embeddings
    bm25_searcher = None # Initialize searcher variable
    faiss_index = None # Initialize index variable
    all_chunks = [] # Initialize chunk list
    chunk_embeddings = None # Initialize embeddings array

    if not HAS_SPLITTER_LIBS:
        logging.error("Cannot proceed with retriever setup: Text splitter libraries not found.")
        return retrievers # Return empty retrievers

    # --- 1. Text Chunking ---
    logging.info("Step 3a: Chunking document pages...")
    try:
        # Define length function based on tokenizer
        if tokenizer:
            # Use tokenizer to count tokens
            def len_func(text: str) -> int:
                # Handle potential errors during encoding just in case
                try:
                    return len(tokenizer.encode(text))
                except Exception as e:
                    logging.warning(f"Tokenizer error on text: '{text[:50]}...'. Error: {e}. Falling back to len().")
                    return len(text)
            logging.info(f"Using tokenizer-based length function for chunk size {CHUNK_SIZE_TOKENS}, overlap {CHUNK_OVERLAP_TOKENS}.")
        else:
            # Fallback to character count if tokenizer failed
            len_func = len
            logging.warning(f"Tokenizer not available. Using character count for chunk size {CHUNK_SIZE_TOKENS}, overlap {CHUNK_OVERLAP_TOKENS}.")

        # Initialize the splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE_TOKENS,
            chunk_overlap=CHUNK_OVERLAP_TOKENS,
            length_function=len_func,
            separators=["\n\n", "\n", ". ", " ", ""], # Common separators, try paragraph, line, sentence, word
            add_start_index=False, # Pyserini doesn't need start index
        )

        chunk_id_counter = 0
        for page in tqdm(pages, desc="Chunking Pages"):
            page_num = page.get('page_num')
            doc_id = page.get('doc_id')
            content = page.get('markdown_content', '')

            if not content.strip(): # Skip empty pages
                continue

            # Split the content of the current page
            page_chunks_texts = text_splitter.split_text(content)

            # Create chunk dictionaries with metadata
            for i, chunk_text in enumerate(page_chunks_texts):
                chunk_id = f"{doc_id}_page{page_num}_chunk{i}" # Unique chunk ID
                chunk_dict = {
                    "id": chunk_id, # Use 'id' for Pyserini compatibility
                    "chunk_id": chunk_id, # Keep original name too
                    "doc_id": doc_id,
                    "page_num": page_num,
                    "contents": chunk_text, # Use 'contents' for Pyserini
                    "text": chunk_text, # Keep original name too
                    "chunk_index_in_page": i,
                }
                all_chunks.append(chunk_dict)
                chunk_id_counter += 1

        logging.info(f"Successfully created {len(all_chunks)} chunks from {len(pages)} pages.")
        if not all_chunks:
             logging.warning("No chunks were created. Check input pages and chunking parameters.")
             # Decide if this is fatal or allow script to continue with no RAG? For now, warn.
        retrievers["chunks"] = all_chunks # Store the chunks

    except Exception as e:
        logging.error(f"Error during text chunking: {e}", exc_info=True)
        # Return partially filled retrievers dict, subsequent steps might fail
        return retrievers

    # --- 2. BM25 Setup ---
    if "bm25" in RETRIEVERS_TO_TEST or "hybrid" in RETRIEVERS_TO_TEST:
        if HAS_PYSERINI and all_chunks:
            logging.info("Step 3b: Setting up BM25 index...")
            # Define paths
            collection_jsonl_path = index_dir / "bm25_collection.jsonl"
            bm25_index_path = index_dir / "bm25_index"

            try:
                # 1. Write chunks to Pyserini JSONL format
                logging.info(f"Writing {len(all_chunks)} chunks to {collection_jsonl_path} for indexing...")
                with open(collection_jsonl_path, 'w', encoding='utf-8') as f:
                    for chunk in all_chunks:
                        # Ensure we have 'id' and 'contents' keys
                        pyserini_doc = {"id": chunk.get("id"), "contents": chunk.get("contents")}
                        if pyserini_doc["id"] is None or pyserini_doc["contents"] is None:
                             logging.warning(f"Skipping chunk due to missing id/contents: {chunk.get('chunk_id')}")
                             continue
                        f.write(json.dumps(pyserini_doc, ensure_ascii=False) + '\n')
                logging.info("Chunk collection file written.")

                # 2. Construct and run Pyserini indexing command
                # Ensure index directory exists and is empty if we want a fresh index
                if bm25_index_path.exists():
                    logging.warning(f"BM25 index directory {bm25_index_path} already exists. Pyserini might skip or overwrite.")
                    # Optional: Add logic to remove existing index if needed: shutil.rmtree(bm25_index_path)

                # Command using Pyserini's indexing module
                # Using English analyzer with Porter stemmer by default (-language en)
                # Storing document vectors (-storeDocvectors) can be useful but increases index size
                # Storing raw docs (-storeRaw) might not be needed if we retrieve by ID and look up in all_chunks
                cmd = [
                    "python", "-m", "pyserini.index.lucene",
                    "--collection", "JsonCollection",
                    "--input", str(collection_jsonl_path.parent), # Input is the folder containing the jsonl
                    "--index", str(bm25_index_path),
                    "--generator", "DefaultLuceneDocumentGenerator",
                    "--threads", str(PYSERINI_THREADS),
                    "--storePositions", "--storeDocvectors", "--storeRaw", # Store positions, vectors, raw docs
                    "-language", "en" # Specify English analyzer
                ]
                logging.info(f"Running Pyserini indexing command: {' '.join(cmd)}")

                # Use subprocess.run for better control and error handling
                result = subprocess.run(cmd, capture_output=True, text=True, check=False) # check=False to handle errors manually

                if result.returncode == 0:
                    logging.info("Pyserini indexing completed successfully.")
                    logging.debug("Pyserini stdout:\n" + result.stdout[-500:]) # Log last part of stdout
                    # 3. Initialize LuceneSearcher
                    try:
                        bm25_searcher = LuceneSearcher(str(bm25_index_path))
                        # Set BM25 parameters (k1, b) - common defaults are k1=0.9, b=0.4
                        bm25_searcher.set_bm25(k1=0.9, b=0.4) # Example parameters, could be configurable
                        retrievers["bm25"] = bm25_searcher
                        logging.info(f"BM25 LuceneSearcher initialized from {bm25_index_path}.")
                    except Exception as e:
                        logging.error(f"Failed to initialize LuceneSearcher after indexing: {e}", exc_info=True)
                else:
                    logging.error(f"Pyserini indexing failed with return code {result.returncode}.")
                    logging.error("Pyserini stderr:\n" + result.stderr)
                    # Consider deleting potentially corrupted index dir?
                    # if bm25_index_path.exists(): shutil.rmtree(bm25_index_path)

                # Optional: Clean up the temporary collection file
                # collection_jsonl_path.unlink(missing_ok=True)

            except Exception as e:
                logging.error(f"An error occurred during BM25 setup: {e}", exc_info=True)

        elif not all_chunks:
             logging.warning("Skipping BM25 setup because no chunks were created.")
        else:
            logging.warning("Skipping BM25 setup: Pyserini library not available or Java 11+ issue.")

    # --- 3. Dense Setup ---
    if "hybrid" in RETRIEVERS_TO_TEST or "dense" in RETRIEVERS_TO_TEST: # Adjusted condition
        if HAS_FAISS and HAS_SBERT and embedder_model and all_chunks:
            logging.info("Step 3c: Setting up Dense (FAISS) index...")
            faiss_index_path = index_dir / "faiss_index.idx" # Define path for FAISS index file
            embeddings_path = index_dir / "chunk_embeddings.npy" # Path to save embeddings

            try:
                # Check if index and embeddings already exist
                if faiss_index_path.exists() and embeddings_path.exists():
                    logging.info(f"Loading existing FAISS index from {faiss_index_path}...")
                    faiss_index = faiss.read_index(str(faiss_index_path))
                    logging.info(f"Loading existing embeddings from {embeddings_path}...")
                    chunk_embeddings = np.load(str(embeddings_path))
                    # Sanity check
                    if faiss_index.ntotal == len(chunk_embeddings) == len(all_chunks):
                         logging.info(f"FAISS index and embeddings loaded successfully. Index contains {faiss_index.ntotal} vectors.")
                    else:
                         logging.error(f"Mismatch between loaded FAISS index ({faiss_index.ntotal}), embeddings ({len(chunk_embeddings)}), and chunks ({len(all_chunks)}). Re-indexing.")
                         faiss_index = None # Force re-indexing
                         chunk_embeddings = None
                else:
                    logging.info(f"Existing FAISS index or embeddings not found.")
                    faiss_index = None
                    chunk_embeddings = None

                # If index or embeddings were not loaded, create them
                if faiss_index is None or chunk_embeddings is None:
                    logging.info(f"Creating new FAISS index and embeddings...")
                    # 1. Extract text content from chunks
                    chunk_texts = [chunk['text'] for chunk in all_chunks]
                    logging.info(f"Embedding {len(chunk_texts)} chunks using {embedder_name}...")

                    # 2. Embed texts using embedder_model.encode() (batch for efficiency)
                    # show_progress_bar=True requires tqdm to be installed
                    chunk_embeddings = embedder_model.encode(
                        chunk_texts,
                        batch_size=EMBEDDING_BATCH_SIZE,
                        show_progress_bar=True,
                        normalize_embeddings=True # Normalize for cosine similarity (common practice)
                    )
                    logging.info(f"Embedding complete. Shape: {chunk_embeddings.shape}")

                    # Ensure embeddings are float32, required by FAISS
                    if chunk_embeddings.dtype != np.float32:
                        logging.info("Converting embeddings to float32 for FAISS.")
                        chunk_embeddings = chunk_embeddings.astype(np.float32)

                    # Save the embeddings
                    logging.info(f"Saving embeddings to {embeddings_path}...")
                    np.save(str(embeddings_path), chunk_embeddings)
                    logging.info("Embeddings saved.")

                    # 3. Create FAISS index
                    embedding_dim = chunk_embeddings.shape[1]
                    # Using IndexFlatL2 - simple brute-force L2 distance search.
                    # For larger datasets, consider IndexHNSWFlat for faster search (requires tuning).
                    # If using cosine similarity (normalize_embeddings=True), IndexFlatIP (Inner Product) is equivalent to cosine after normalization.
                    faiss_index = faiss.IndexFlatIP(embedding_dim) # Inner product is equivalent to cosine on normalized vectors
                    logging.info(f"Created FAISS IndexFlatIP index with dimension {embedding_dim}.")

                    # 4. Add embeddings to index
                    faiss_index.add(chunk_embeddings)
                    logging.info(f"Added {faiss_index.ntotal} vectors to FAISS index.")

                    # 5. Save index
                    logging.info(f"Saving FAISS index to {faiss_index_path}...")
                    faiss.write_index(faiss_index, str(faiss_index_path))
                    logging.info("FAISS index saved.")

                # Store the index and embeddings in the retrievers dictionary
                retrievers["dense"] = faiss_index
                retrievers["chunk_embeddings"] = chunk_embeddings # Store embeddings for MMR

            except Exception as e:
                logging.error(f"An error occurred during Dense (FAISS) setup: {e}", exc_info=True)

        elif not all_chunks:
             logging.warning("Skipping Dense setup because no chunks were created.")
        else:
            missing_libs = []
            if not HAS_FAISS: missing_libs.append("FAISS")
            if not HAS_SBERT: missing_libs.append("SentenceTransformers")
            if not embedder_model: missing_libs.append("Embedder Model")
            logging.warning(f"Skipping Dense setup: Required components missing ({', '.join(missing_libs)}).")

    logging.info("Retriever setup function finished.")
    return retrievers


# =============================================================================
# --- Core Execution Functions ---
# =============================================================================

def _cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculates cosine similarity between two vectors."""
    # Assumes vectors are already normalized (e.g., by SentenceTransformer or FAISS IndexFlatIP)
    # For normalized vectors, cosine similarity is just the dot product.
    # Add check for zero vector to avoid potential NaN
    if np.all(v1 == 0) or np.all(v2 == 0):
        return 0.0
    return np.dot(v1, v2)

def _mmr_rerank(query_embedding: np.ndarray,
                candidate_embeddings: np.ndarray,
                candidate_ids: List[int], # Indices into the original all_chunks list
                lambda_param: float,
                k: int) -> List[Tuple[int, float]]:
    """
    Performs Maximal Marginal Relevance (MMR) re-ranking.
    Args:
        query_embedding: The embedding of the query (normalized). Shape (dim,)
        candidate_embeddings: Embeddings of the candidate chunks (normalized). Shape: (n_candidates, dim)
        candidate_ids: List of original indices corresponding to candidate_embeddings.
        lambda_param: Balances relevance and diversity (0=max diversity, 1=max relevance).
        k: The number of results to return.
    Returns:
        List of tuples: (original_chunk_index, relevance_score) sorted by MMR selection order.
    """
    if candidate_embeddings is None or len(candidate_embeddings) == 0 or query_embedding is None:
        logging.warning("MMR input invalid (embeddings or query missing/empty).")
        return []
    if not (0 <= lambda_param <= 1):
        logging.warning(f"MMR lambda_param {lambda_param} out of range [0, 1]. Clamping.")
        lambda_param = max(0.0, min(1.0, lambda_param))

    n_candidates = candidate_embeddings.shape[0]
    k = min(k, n_candidates) # Cannot return more than available candidates

    # Ensure query_embedding is 1D array
    if query_embedding.ndim > 1:
        query_embedding = query_embedding.flatten()

    # Calculate relevance (similarity to query) - use dot product for normalized vectors
    relevance_scores = candidate_embeddings.dot(query_embedding)

    selected_indices_in_candidates = [] # Indices within the *candidate* list (0 to n_candidates-1)
    remaining_indices_in_candidates = list(range(n_candidates))

    while len(selected_indices_in_candidates) < k:
        best_mmr_score = -np.inf
        best_idx_in_remaining = -1 # Index within remaining_indices_in_candidates

        for current_remaining_idx, candidate_idx in enumerate(remaining_indices_in_candidates):
            relevance = relevance_scores[candidate_idx]

            if not selected_indices_in_candidates:
                # First item's diversity term is 0
                diversity = 0.0
            else:
                # Calculate max similarity to already selected items
                # Similarities = dot product between current candidate and all selected candidates
                selected_embeddings = candidate_embeddings[selected_indices_in_candidates]
                similarities_to_selected = selected_embeddings.dot(candidate_embeddings[candidate_idx])
                max_similarity = np.max(similarities_to_selected) if len(similarities_to_selected) > 0 else 0.0
                diversity = max_similarity

            # MMR Score = lambda * Relevance - (1 - lambda) * Max_Similarity_To_Selected
            mmr_score = lambda_param * relevance - (1 - lambda_param) * diversity

            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_idx_in_remaining = current_remaining_idx # Store index within the *remaining* list

        if best_idx_in_remaining != -1:
            # Get the actual index in the original candidate list
            selected_candidate_idx = remaining_indices_in_candidates.pop(best_idx_in_remaining)
            selected_indices_in_candidates.append(selected_candidate_idx)
        else:
            # Should not happen if k <= n_candidates, but break just in case
            logging.warning("MMR loop finished unexpectedly.")
            break

    # Map selected candidate indices back to original chunk indices and return with relevance scores
    final_results = []
    for idx_in_candidates in selected_indices_in_candidates:
        original_chunk_index = candidate_ids[idx_in_candidates]
        relevance_score = relevance_scores[idx_in_candidates]
        final_results.append((original_chunk_index, relevance_score))

    # Return in the order they were selected by MMR
    return final_results


def get_context(query: str, retriever_type: str, retrievers: Dict[str, Any], pages_dict: Dict[int, str], full_manual_text: str, dense_embedder: SentenceTransformer, retrieval_params: Dict) -> Tuple[str, List[Dict]]:
    """Retrieves context based on the specified retriever type. Needs dense_embedder."""
    logging.debug(f"Getting context for query '{query[:50]}...' using retriever: {retriever_type}")

    all_chunks_list = retrievers.get("chunks", []) # Get the list of chunk dicts
    if not all_chunks_list and retriever_type != "none":
         logging.warning(f"No chunks available for retriever type '{retriever_type}'. Returning empty context.")
         return "Error: No chunks available for retrieval.", []
    # Create lookup only if chunks exist
    chunk_lookup = {chunk['id']: chunk for chunk in all_chunks_list} if all_chunks_list else {}


    if retriever_type == "none":
        # For 'none', return the full manual text and a single source dict representing it
        # TODO: Implement truncation logic here based on model context window
        logging.debug("Using 'none' retriever (full manual text - truncation not implemented).")
        return full_manual_text, [{"doc_id": "full_manual", "content": full_manual_text, "score": 1.0, "page_num": None, "chunk_id": None}]

    elif retriever_type == "bm25":
        bm25_searcher = retrievers.get("bm25")
        if bm25_searcher and chunk_lookup:
            k = retrieval_params.get('bm25_k', 10) # Get k from params
            logging.debug(f"Performing BM25 search for top {k}...")
            try:
                hits = bm25_searcher.search(query, k=k)
                retrieved_docs = []
                context_parts = []

                for i in range(len(hits)):
                    hit_id = hits[i].docid
                    hit_score = hits[i].score
                    # Look up the original chunk data using the ID
                    original_chunk = chunk_lookup.get(hit_id)
                    if original_chunk:
                        chunk_text = original_chunk.get('text', '') # Use 'text' for context
                        context_parts.append(chunk_text)
                        retrieved_docs.append({
                            "chunk_id": hit_id,
                            "doc_id": original_chunk.get('doc_id'),
                            "page_num": original_chunk.get('page_num'),
                            "score": hit_score,
                            "retriever": "bm25", # Add retriever type
                            "content": chunk_text # Include content for reference
                        })
                    else:
                         logging.warning(f"BM25 hit ID '{hit_id}' not found in original chunk list.")

                context_str = "\n\n".join(context_parts)
                logging.debug(f"BM25 retrieved {len(retrieved_docs)} docs. Context length: {len(context_str)}")
                return context_str, retrieved_docs
            except Exception as e:
                 logging.error(f"Error during BM25 search: {e}", exc_info=True)
                 return f"Error during BM25 search: {e}", []
        else:
            logging.error("BM25 retriever requested but not available or no chunks.")
            return "Error: BM25 retriever not available or no chunks.", []

    elif retriever_type == "dense": # Added Dense only retrieval
        faiss_index = retrievers.get("dense")
        if faiss_index and dense_embedder and all_chunks_list:
            k = retrieval_params.get('dense_k', 10)
            logging.debug(f"Performing Dense search for top {k}...")
            try:
                # 1. Embed the query
                query_embedding = dense_embedder.encode([query], normalize_embeddings=True)
                if query_embedding.dtype != np.float32:
                     query_embedding = query_embedding.astype(np.float32)

                # 2. Search the FAISS index
                # D = distances (inner product scores), I = indices of chunks in original list
                scores, indices = faiss_index.search(query_embedding, k=k)

                retrieved_docs = []
                context_parts = []
                if len(indices) > 0: # Check if search returned results
                    for i, idx in enumerate(indices[0]): # Iterate through the indices for the first query
                        if idx < 0 or idx >= len(all_chunks_list): # FAISS can return -1 if k > index size
                            logging.warning(f"FAISS returned invalid index: {idx}")
                            continue
                        original_chunk = all_chunks_list[idx] # Get chunk by index
                        hit_score = scores[0][i] # Corresponding score
                        chunk_text = original_chunk.get('text', '')
                        context_parts.append(chunk_text)
                        retrieved_docs.append({
                            "chunk_id": original_chunk.get('id'),
                            "doc_id": original_chunk.get('doc_id'),
                            "page_num": original_chunk.get('page_num'),
                            "score": float(hit_score), # Convert numpy float to standard float
                            "retriever": "dense", # Add retriever type
                            "content": chunk_text
                        })

                context_str = "\n\n".join(context_parts)
                logging.debug(f"Dense retrieved {len(retrieved_docs)} docs. Context length: {len(context_str)}")
                return context_str, retrieved_docs

            except Exception as e:
                logging.error(f"Error during Dense search: {e}", exc_info=True)
                return f"Error during Dense search: {e}", []
        else:
            logging.error("Dense retriever requested but FAISS index, embedder, or chunks not available.")
            return "Error: Dense retriever components not available.", []


    elif retriever_type == "hybrid":
        bm25_searcher = retrievers.get("bm25")
        faiss_index = retrievers.get("dense")
        chunk_embeddings = retrievers.get("chunk_embeddings")

        if bm25_searcher and faiss_index and dense_embedder and all_chunks_list is not None and chunk_embeddings is not None:
            bm25_k = retrieval_params.get('bm25_k', 10)
            dense_k = retrieval_params.get('dense_k', 10)
            hybrid_k = retrieval_params.get('hybrid_k', 10) # Final k after MMR
            lambda_param = retrieval_params.get('mmr_lambda', 0.5)
            logging.debug(f"Performing Hybrid search (BM25 k={bm25_k}, Dense k={dense_k}, Final k={hybrid_k}, MMR λ={lambda_param})...")

            # --- Get BM25 Results ---
            bm25_hit_ids = set()
            try:
                bm25_hits_raw = bm25_searcher.search(query, k=bm25_k)
                bm25_hit_ids = {hit.docid for hit in bm25_hits_raw}
                logging.debug(f"BM25 search returned {len(bm25_hit_ids)} unique IDs.")
            except Exception as e:
                logging.error(f"Error during BM25 search for hybrid: {e}", exc_info=True)
                # Continue without BM25 results if it fails? Or return error? Let's continue for now.

            # --- Get Dense Results ---
            dense_hit_indices = [] # Store original indices from all_chunks_list
            try:
                query_embedding = dense_embedder.encode([query], normalize_embeddings=True)
                if query_embedding.dtype != np.float32:
                     query_embedding = query_embedding.astype(np.float32)
                scores, indices = faiss_index.search(query_embedding, k=dense_k)
                if len(indices) > 0:
                    dense_hit_indices = [idx for idx in indices[0] if 0 <= idx < len(all_chunks_list)]
                logging.debug(f"Dense search returned {len(dense_hit_indices)} valid indices.")
            except Exception as e:
                logging.error(f"Error during Dense search for hybrid: {e}", exc_info=True)
                # Continue without Dense results if it fails?

            # --- Combine Candidates ---
            # Map BM25 hit IDs to original indices
            bm25_hit_indices = set()
            if bm25_hit_ids:
                 for i, chunk in enumerate(all_chunks_list):
                     if chunk['id'] in bm25_hit_ids:
                         bm25_hit_indices.add(i)

            # Combine unique original indices
            combined_indices = list(bm25_hit_indices.union(set(dense_hit_indices)))
            logging.debug(f"Combined unique candidate indices: {len(combined_indices)}")

            if not combined_indices:
                 logging.warning("Hybrid search yielded no candidates from BM25 or Dense.")
                 return "No relevant chunks found by hybrid search.", []

            # --- Prepare for MMR ---
            try:
                candidate_embeddings_np = chunk_embeddings[combined_indices].astype(np.float32) # Get embeddings for combined candidates
                if candidate_embeddings_np.ndim == 1: # Handle case with only one candidate
                    candidate_embeddings_np = candidate_embeddings_np.reshape(1, -1)

                # --- Apply MMR ---
                logging.debug(f"Applying MMR to {len(combined_indices)} candidates to select top {hybrid_k}...")
                mmr_results = _mmr_rerank(
                    query_embedding=query_embedding[0], # Pass the single query embedding
                    candidate_embeddings=candidate_embeddings_np,
                    candidate_ids=combined_indices, # Pass original indices
                    lambda_param=lambda_param,
                    k=hybrid_k
                )
                logging.debug(f"MMR selected {len(mmr_results)} results.")

                # --- Format context and sources ---
                retrieved_docs = []
                context_parts = []
                for original_index, score in mmr_results:
                    original_chunk = all_chunks_list[original_index]
                    chunk_text = original_chunk.get('text', '')
                    context_parts.append(chunk_text)
                    retrieved_docs.append({
                        "chunk_id": original_chunk.get('id'),
                        "doc_id": original_chunk.get('doc_id'),
                        "page_num": original_chunk.get('page_num'),
                        "score": float(score), # Using relevance score from MMR for now
                        "retriever": "hybrid", # Add retriever type
                        "content": chunk_text
                    })

                context_str = "\n\n".join(context_parts)
                logging.debug(f"Hybrid retrieved {len(retrieved_docs)} docs after MMR. Context length: {len(context_str)}")
                return context_str, retrieved_docs

            except Exception as e:
                logging.error(f"Error during Hybrid search (MMR/Formatting): {e}", exc_info=True)
                return f"Error during Hybrid search: {e}", []
        else:
            logging.error("Hybrid retriever requested but BM25, FAISS index, embedder, or embeddings not available.")
            return "Error: Hybrid retriever components not available.", []

    else:
        logging.error(f"Unknown retriever type requested: {retriever_type}")
        return f"Error: Unknown retriever type '{retriever_type}'.", []


def build_prompt_messages(question: str, context: str, prompt_type: str, system_prompt: str, gold_category: str) -> List[Dict[str, str]]:
    """Builds the list of messages for the LLM API call based on prompt type."""
    logging.debug(f"Building prompt type: {prompt_type}")
    # Placeholder: Return basic zero-shot structure for now
    # Requires: Implementing logic for few-shot (finding examples) and CoT (adding reasoning steps to examples)
    user_content = f"Context:\n```\n{context}\n```\n\nQuestion:\n```\n{question}\n```\n\nOutput JSON:"
    if system_prompt is None or "ERROR" in system_prompt:
         # This case should not be reached if initialize_clients_and_models exits on error
         logging.error("System prompt is missing or invalid. Using a basic fallback.")
         system_prompt = "You are a helpful assistant. Answer the question based on the context and return JSON."

    messages = [{"role": "system", "content": system_prompt}]

    if prompt_type == "zero_shot":
        messages.append({"role": "user", "content": user_content})
    elif prompt_type == "few_shot":
        # TODO: Implement few-shot example selection and formatting
        # examples_str = "EXAMPLE 1: \n..."
        # messages.append({"role": "user", "content": examples_str + "\n\n" + user_content})
        logging.warning("Few-shot prompt building not implemented.")
        messages.append({"role": "user", "content": "[FEW-SHOT EXAMPLES NOT IMPLEMENTED]\n\n" + user_content}) # Placeholder
    elif prompt_type == "few_shot_cot":
        # TODO: Implement few-shot CoT example selection and formatting
        # examples_str = "EXAMPLE 1 (with reasoning): \n..."
        # messages.append({"role": "user", "content": examples_str + "\n\n" + user_content})
        logging.warning("Few-shot-CoT prompt building not implemented.")
        messages.append({"role": "user", "content": "[FEW-SHOT-COT EXAMPLES NOT IMPLEMENTED]\n\n" + user_content}) # Placeholder
    else:
        logging.error(f"Unknown prompt type: {prompt_type}")
        messages.append({"role": "user", "content": "[UNKNOWN PROMPT TYPE]\n\n" + user_content}) # Placeholder

    return messages

def call_llm(model_key: str, provider: str, api_id: str, context_window: int, prompt_messages: List[Dict], clients: Dict) -> str:
    """Calls the appropriate LLM API based on the provider using initialized clients."""
    logging.info(f"Calling LLM: {model_key} (Provider: {provider}, API ID: {api_id})")
    # Placeholder: Return dummy JSON string indicating unimplemented call
    # Requires: Implementing API call logic for openai, google, groq using clients dict
    raw_output = json.dumps({
      "answer": f"LLM Call Not Implemented for {provider} ({model_key})",
      "page": None,
      "predicted_category": "Unanswerable",
      "predicted_persona": None,
      "_debug_provider": provider,
      "_debug_model_key": model_key
    })

    if provider == "openai":
        if clients.get("openai_client"):
            # TODO: Implement OpenAI API call
            logging.warning(f"OpenAI call logic for {model_key} not implemented.")
            pass # Replace with actual call
        else:
            logging.error("OpenAI client requested but not initialized.")
            raw_output = json.dumps({"error": "OpenAI client not initialized"})
    elif provider == "google":
        if clients.get("google_client"):
             # TODO: Implement Google Gemini API call
             logging.warning(f"Google Gemini call logic for {model_key} not implemented.")
             pass # Replace with actual call
        else:
             logging.error("Google client requested but not initialized/configured.")
             raw_output = json.dumps({"error": "Google client not initialized"})
    elif provider == "groq":
        if clients.get("groq_client"):
             # TODO: Implement Groq API call
             logging.warning(f"Groq call logic for {model_key} not implemented.")
             pass # Replace with actual call
        else:
             logging.error("Groq client requested but not initialized.")
             raw_output = json.dumps({"error": "Groq client not initialized"})
    else:
        logging.error(f"Unknown provider specified: {provider}")
        raw_output = json.dumps({"error": f"Unknown provider: {provider}"})

    # Simulate API delay
    time.sleep(API_DELAY_SECONDS)

    return raw_output # Return the raw string output from the LLM (or error JSON)


def parse_llm_json_output(llm_output_str: str) -> Tuple[bool, Dict]:
    """Parses the LLM's raw string output, expecting JSON."""
    # This function remains simple as complex scoring is deferred
    logging.debug(f"Attempting to parse LLM output: {llm_output_str[:200]}...")
    if not isinstance(llm_output_str, str) or not llm_output_str.strip():
        logging.warning("LLM output is empty or not a string.")
        return False, {"error": "Empty or non-string output", "raw_output": llm_output_str}
    try:
        # Clean potential markdown fences (more robustly)
        match = re.search(r'```(json)?\s*(\{.*?\})\s*```', llm_output_str, re.DOTALL | re.IGNORECASE)
        if match:
            json_str = match.group(2)
            logging.debug("Extracted JSON from markdown fences.")
        else:
            # Assume the whole string is JSON if no fences found
            json_str = llm_output_str.strip()

        parsed_json = json.loads(json_str)
        logging.debug("Successfully parsed LLM output as JSON.")
        return True, parsed_json
    except json.JSONDecodeError as e:
        logging.warning(f"Failed to parse LLM output as JSON: {e}. Raw output: {llm_output_str[:500]}...")
        return False, {"error": f"JSONDecodeError: {e}", "raw_output": llm_output_str}
    except Exception as e:
        logging.error(f"Unexpected error parsing LLM output: {e}", exc_info=True)
        return False, {"error": f"Unexpected parsing error: {e}", "raw_output": llm_output_str}


# =============================================================================
# --- Main Execution Block ---
# =============================================================================

if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run one technical manual through the RAG/LLM benchmark grid.")
    parser.add_argument("-i", "--input_dir", type=pathlib.Path, required=True,
                        help="Path to the directory containing the manual's processed files (e.g., data/processed/heat_pump_dryer)")
    parser.add_argument("-o", "--outdir", type=pathlib.Path, required=True,
                        help="Base output directory where the results subdirectory for this manual will be created (e.g., results/).")
    args = parser.parse_args()

    # --- Validate Input Directory ---
    if not args.input_dir.is_dir():
        logging.error(f"Input directory not found or is not a directory: {args.input_dir}"); sys.exit(1)

    # --- Derive Manual Name and File Paths ---
    manual_name = args.input_dir.name
    pages_file_path = args.input_dir / f"{manual_name}.jsonl"
    qa_file_path = args.input_dir / f"{manual_name}_gold.csv" # Assumes Phase 1 finalize output

    # --- Validate Derived File Paths ---
    if not pages_file_path.is_file(): logging.error(f"Pages file not found at expected location: {pages_file_path}"); sys.exit(1)
    if not qa_file_path.is_file(): logging.error(f"Gold QA file not found at expected location: {qa_file_path}"); sys.exit(1)

    # --- Setup Output Directory ---
    run_output_dir = args.outdir / manual_name
    index_dir = run_output_dir / "indices"     # For storing retriever indices
    results_dir = run_output_dir / "results"   # For storing raw LLM outputs/scores per condition
    # Define the single output file path for this manual's run
    output_pickle_file = results_dir / f"{manual_name}_benchmark_results.pkl"
    try:
        run_output_dir.mkdir(parents=True, exist_ok=True)
        index_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory for this run: {run_output_dir}")
        print(f"Detailed results will be saved to: {output_pickle_file}")
    except OSError as e:
         logging.error(f"Could not create output directories under {args.outdir}: {e}"); sys.exit(1)

    # --- Initialize API Clients, Embedders, and Load System Prompt ---
    print("\n--- Step 2: Initializing API Clients, Embedders & Loading System Prompt ---")
    initialized_components = initialize_clients_and_models(
        MODELS_TO_TEST,
        DENSE_RETRIEVAL_EMBEDDER,
        SEMANTIC_SIMILARITY_EMBEDDER
    )
    # Extract components for easier access
    clients = {
        "openai": initialized_components["openai_client"],
        "google": initialized_components["google_client"],
        "groq": initialized_components["groq_client"]
    }
    dense_embedder = initialized_components["dense_embedder"]
    tokenizer = initialized_components["tokenizer"] # Get the tokenizer
    # semantic_embedder = initialized_components["semantic_embedder"] # Not used in this script
    system_prompt = initialized_components["system_prompt"]
    # System prompt loading failure is now handled inside initialize_clients_and_models
    print("--- Initialization Complete ---")


    # --- Load Gold QA Data ---
    print("\n--- Step 1a: Loading Gold QA Data ---")
    gold_qa_df = load_gold_qa(qa_file_path)
    print(f"Successfully loaded {len(gold_qa_df)} Gold QA pairs.")
    # Display some info about the loaded QA data
    print("Gold QA DataFrame Info:")
    gold_qa_df.info(verbose=False, show_counts=True) # Less verbose info
    # print("\nFirst 2 Gold QA pairs (showing relevant columns):")
    # cols_to_show = ['question_id', 'question_text', 'category', 'gt_page_number', 'parsed_steps', 'gold_steps_list']
    # print(gold_qa_df[cols_to_show].head(2).to_string())
    print("\nGold QA Category Distribution:")
    print(gold_qa_df['category'].value_counts())

    # --- Load Manual Pages ---
    print("\n--- Step 1b: Loading Manual Pages ---")
    pages_list = load_pages(pages_file_path)
    print(f"Successfully loaded {len(pages_list)} pages.")
    # Create a dictionary for quick page lookup by page_num
    pages_dict = {p['page_num']: p.get('markdown_content', '') for p in pages_list}
    # Create full manual text for 'none' retriever
    full_manual_text = "\n\n".join([f"--- Page {p['page_num']} ---\n{p.get('markdown_content', '')}" for p in pages_list])
    print(f"Created full manual text string (length: {len(full_manual_text)} chars).")
    # if pages_list:
    #     print(f"First page number in list: {pages_list[0].get('page_num')}, Content length: {len(pages_list[0].get('markdown_content', ''))} chars")
    #     print(f"Last page number in list: {pages_list[-1].get('page_num')}, Content length: {len(pages_list[-1].get('markdown_content', ''))} chars")
    print("--- Data Loading Complete ---")


    # --- Setup Retrievers ---
    print("\n--- Step 3: Setting up Retrievers ---")
    # Pass the initialized dense_embedder and tokenizer
    retrievers = setup_retrievers(
        pages=pages_list,
        embedder_name=DENSE_RETRIEVAL_EMBEDDER,
        embedder_model=dense_embedder,
        tokenizer=tokenizer, # Pass the tokenizer
        index_dir=index_dir
    )
    # Extract chunks for potential later use if needed (though get_context should handle retrieval)
    all_chunks = retrievers.get("chunks", [])
    print(f"--- Retriever Setup Complete ({len(all_chunks)} chunks created, BM25 Indexing Implemented, Dense Indexing Implemented) ---") # Updated log


    # --- Main Benchmark Loop ---
    # Use the lists loaded from config
    total_conditions = len(MODELS_TO_TEST) * len(RETRIEVERS_TO_TEST) * len(PROMPTS_TO_TEST)
    print(f"\n--- Step 4: Starting Benchmark Run ({len(gold_qa_df)} questions x {total_conditions} conditions) ---")
    all_results = [] # List to store results for each trial

    # Outer loop: Iterate through each question in the gold dataset
    # Use tqdm for progress bar over questions
    for index, gold_row in tqdm(gold_qa_df.iterrows(), total=len(gold_qa_df), desc="Benchmarking Questions"):
        question_id = gold_row['question_id']
        question_text = gold_row['question_text']
        gold_category = gold_row['category'] # Needed for prompt building potentially

        # Middle loop: Iterate through each retriever type (from config)
        for retriever_type in RETRIEVERS_TO_TEST:
            # Check if retriever is available/supported
            # Adjusted check slightly: BM25 needs retrievers['bm25'], Hybrid needs both bm25 and dense
            # Dense only needs retrievers['dense']
            if retriever_type == "bm25" and retrievers.get("bm25") is None:
                 logging.warning(f"Retriever 'bm25' not available/setup. Skipping for QID {question_id}.")
                 continue
            if retriever_type == "dense" and retrievers.get("dense") is None:
                 logging.warning(f"Retriever 'dense' not available/setup. Skipping for QID {question_id}.")
                 continue
            if retriever_type == "hybrid" and (retrievers.get("bm25") is None or retrievers.get("dense") is None):
                 logging.warning(f"Retriever 'hybrid' needs BM25 and Dense, one or both missing. Skipping for QID {question_id}.")
                 continue
            # 'none' retriever doesn't need a check here

            # 1. Get Context
            # Pass necessary components to get_context
            context_str, retrieved_sources = get_context(
                query=question_text,
                retriever_type=retriever_type,
                retrievers=retrievers,
                pages_dict=pages_dict,
                full_manual_text=full_manual_text,
                dense_embedder=dense_embedder, # Pass embedder
                retrieval_params=RETRIEVAL_PARAMS # Pass params
            )

            # Inner loops: Iterate through models and prompts (from config)
            for model_key, model_details in MODELS_TO_TEST.items():
                for prompt_type in PROMPTS_TO_TEST:

                    # --- Prepare for Trial ---
                    # Store the entire gold row for easy access in the results file
                    trial_info = {
                        "manual_id": manual_name,
                        "gold_data": gold_row.to_dict(), # Store all gold data
                        "condition_model": model_key,
                        "condition_retriever": retriever_type,
                        "condition_prompt": prompt_type,
                        "retrieved_context": context_str,
                        "retrieved_sources": retrieved_sources, # Store info about sources (doc_id, page_num, score)
                        "llm_raw_output": None,
                        "llm_parsed_output": None,
                        "json_parsable": None,
                        "error_message": None
                    }

                    try:
                        # 2. Build Prompt
                        prompt_messages = build_prompt_messages(
                            question=question_text,
                            context=context_str,
                            prompt_type=prompt_type,
                            system_prompt=system_prompt,
                            gold_category=gold_category # Pass category for potential prompt logic
                        )

                        # 3. Call LLM
                        llm_output_raw = call_llm(
                            model_key=model_key,
                            provider=model_details['provider'],
                            api_id=model_details['api_id'],
                            context_window=model_details['context_window'],
                            prompt_messages=prompt_messages,
                            clients=clients # Pass the dictionary of initialized clients
                        )
                        trial_info["llm_raw_output"] = llm_output_raw

                        # 4. Parse LLM Output (Basic Check)
                        parsable, parsed_output = parse_llm_json_output(llm_output_raw)
                        trial_info["json_parsable"] = parsable
                        trial_info["llm_parsed_output"] = parsed_output # Store parsed dict or error dict

                    except Exception as e:
                         logging.error(f"Error during trial for QID {question_id}, Model {model_key}, Retriever {retriever_type}, Prompt {prompt_type}: {e}", exc_info=True)
                         trial_info["error_message"] = str(e)

                    # Append result of this trial to the list
                    all_results.append(trial_info)

    print(f"--- Benchmark Run Complete. Total trials executed (attempted): {len(all_results)} ---")

    # --- Save Results ---
    print(f"\n--- Step 5: Saving Detailed Results ---")
    try:
        with open(output_pickle_file, 'wb') as f:
            pickle.dump(all_results, f)
        print(f"Successfully saved {len(all_results)} trial results to: {output_pickle_file}")
    except Exception as e:
        logging.error(f"Failed to save results to {output_pickle_file}: {e}", exc_info=True)

    print("\n--- run_benchmark_manual.py finished ---")