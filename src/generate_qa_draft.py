import os
import csv
import json
import sys
import random
import argparse
import time
import re
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Any
import ast # For parsing LLM step output

# Attempt to import necessary libraries
try:
    import pandas as pd
    from sentence_transformers import SentenceTransformer, util
    import google.generativeai as genai
    import google.api_core.exceptions # For rate limit handling
    import openai
    from dotenv import load_dotenv
    from datasets import Dataset, Features, Value, Sequence
    from ragas import evaluate as ragas_eval
    from ragas.metrics import faithfulness, answer_correctness
    from sklearn.metrics import cohen_kappa_score
except ImportError as e:
    # --- Indented block for exception handling ---
    print(f"Error importing libraries: {e}")
    print("Please ensure prerequisites are installed:")
    print("pip install PyYAML google-generativeai google-api-core openai sentence-transformers pandas scikit-learn datasets ragas python-dotenv") # Added PyYAML
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(lineno)d: %(message)s')

# ------------------ CONFIGURATION LOADING ------------------------------------
# (Configuration loading unchanged)
try:
    import yaml
    SCRIPT_DIR = Path(__file__).resolve().parent
    CONFIG_PATH = SCRIPT_DIR.parent / "config" / "settings_generate.yaml"
    logging.info(f"Loading configuration from: {CONFIG_PATH}")
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    logging.info("Configuration loaded successfully.")
except FileNotFoundError:
    logging.error(f"Configuration file not found at calculated path: {CONFIG_PATH}")
    logging.error("Ensure 'config/settings_generate.yaml' exists relative to the script's parent directory.")
    sys.exit(1)
except ImportError:
     logging.error("PyYAML not found. Please install it (`pip install pyyaml`) to load config/settings_generate.yaml.")
     sys.exit(1)
except yaml.YAMLError as e:
     logging.error(f"Error parsing configuration file {CONFIG_PATH}: {e}")
     sys.exit(1)
except Exception as e:
     logging.error(f"An unexpected error occurred during configuration loading: {e}", exc_info=True)
     sys.exit(1)

# --- Extract config values into global constants (or pass config dict around) ---
try:
    # Pipeline Params
    OVERGEN_FACTOR = config['pipeline']['overgen_factor']
    DUP_THRESHOLD = config['pipeline']['dup_threshold']
    RAGAS_THRESHOLD = config['pipeline']['ragas_threshold']
    JUDGE_THRESHOLD = config['pipeline']['judge_threshold']
    FINAL_DATASET_SIZE = config['pipeline']['final_dataset_size']
    TARGET_RAW_ROWS = FINAL_DATASET_SIZE * OVERGEN_FACTOR

    # Models
    EMBED_MODEL = config['models']['embed']
    GENERATION_MODEL = config['models']['generation']
    STEP_PARSING_MODEL = config['models']['step_parsing']
    JUDGE_MODEL = config['models']['judge'] # User must verify ID

    # Files
    MASTER_PROMPT_PATH = SCRIPT_DIR.parent / config['files']['master_prompt'] # Relative to project root
    RAW_OUTPUT_SUFFIX = config['files']['raw_output_suffix']
    CANDIDATE_SUFFIX = config['files']['candidate_suffix']
    STATS_SUFFIX = config['files']['stats_suffix']
    AUDIT_SUFFIX_A = config['files']['audit_suffix_a']
    AUDIT_SUFFIX_B = config['files']['audit_suffix_b']
    GOLD_DATASET_SUFFIX = config['files']['gold_dataset_suffix']

    # Quotas
    CATEGORY_TARGETS = config['quotas']['category_targets']
    # Verify final size matches sum of targets
    if sum(CATEGORY_TARGETS.values()) != FINAL_DATASET_SIZE:
         logging.warning(f"Sum of CATEGORY_TARGETS ({sum(CATEGORY_TARGETS.values())}) != FINAL_DATASET_SIZE ({FINAL_DATASET_SIZE}). Using sum value.")
         FINAL_DATASET_SIZE = sum(CATEGORY_TARGETS.values()) # Recalculate

    # Audit
    AUDIT_FRACTION = config['audit']['audit_fraction']
    KAPPA_MIN = config['audit']['kappa_min']
    SENTINEL_ACCURACY_THRESHOLD = config['audit']['sentinel_accuracy_threshold']

    # API Params
    JUDGE_DELAY_SECONDS = config['api_params']['judge_delay_seconds']
    MAX_PARSE_RETRIES = config['api_params']['max_parse_retries']
    RETRY_DELAY_SECONDS = config['api_params']['retry_delay_seconds']

except KeyError as e:
     logging.error(f"Missing key in configuration file {CONFIG_PATH}: {e}")
     sys.exit(1)

# --- Constants defined in script ---
PROCEDURAL_CATEGORY_NAME = "Procedural Step Inquiry"
LOCATION_DEF_CATEGORY_NAME = "Location/Definition"
UNANSWERABLE_CATEGORY_NAME = "Unanswerable"
SCHEMA = ["question_id", "persona", "doc_id", "language",
          "question_text", "category", "gt_answer_snippet",
          "gt_page_number", "_self_grounded"]
FINAL_SCHEMA = SCHEMA + ['parsed_steps', 'passed_strict_check', 'corrected_steps', 'procedural_comments'] # Added corrected_steps and procedural_comments
SENTINEL_ROWS_INFO = [ # Using 1 BAD Sentinel
    ({"question_id": "SENTINEL_BAD_01", "persona": "Technician", "doc_id": "N/A", "language": "fr",
      "question_text": "Explain about safety features thing?", "category": "Specification Lookup",
      "gt_answer_snippet": "Safety important.", "gt_page_number": "-1",
      "_self_grounded": "False", "parsed_steps": None, "passed_strict_check": False},
     {"answer_correct?": "no", "grounded?": "no", "question_clear?": "no", "category_correct?": "no", "persona_tone_ok?": "no"})
]
NUM_SENTINELS = len(SENTINEL_ROWS_INFO)

# --- NEW: Valid Personas and Categories (derived from master prompt) ---
VALID_PERSONAS = ["Novice User", "Technician", "SafetyOfficer"]
VALID_CATEGORIES = [
    "Specification Lookup",
    "Tool/Material Identification",
    PROCEDURAL_CATEGORY_NAME,  # Use constant
    LOCATION_DEF_CATEGORY_NAME, # Use constant
    "Conditional Logic/Causal Reasoning",
    "Safety Information Lookup",
    UNANSWERABLE_CATEGORY_NAME  # Use constant
]
# -----------------------------------------------------------------------------

# --- Setup basic logging ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
# -----------------------------------------------------------------------------


# ------------------ INITIALISE MODELS ----------------------------------------
# (Model initialization unchanged)
print("Loading environment variables from .env file if present...")
load_dotenv()
try:
    google_api_key = os.getenv("GOOGLE_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not google_api_key: raise ValueError("GOOGLE_API_KEY not found.")
    if not openai_api_key: raise ValueError("OPENAI_API_KEY not found.")
    print("API keys loaded successfully.")

    genai.configure(api_key=google_api_key)
    # Configure OpenAI client only if needed by selected models
    if "gpt" in JUDGE_MODEL.lower() or "gpt" in STEP_PARSING_MODEL.lower():
        if not openai: raise ImportError("OpenAI library needed but not imported.")
        openai.api_key = openai_api_key; print("OpenAI API key configured.")
    else: print("OpenAI models not selected, skipping key config for it.")

    print(f"Initializing generation model: {GENERATION_MODEL}")
    gemini_model = genai.GenerativeModel(GENERATION_MODEL)
    print(f"Initializing judge model: {JUDGE_MODEL}")
    print(f"Initializing step parsing model: {STEP_PARSING_MODEL}")
    print(f"Initializing embedding model: {EMBED_MODEL}")
    embedding_model = SentenceTransformer(EMBED_MODEL)
    print("Models initialized successfully.")

except ValueError as ve: logging.error(f"Configuration Error: {ve}"); sys.exit(1)
except ImportError as ie: logging.error(f"Import Error during conditional OpenAI config: {ie}"); sys.exit(1)
except Exception as e: logging.error(f"Error initializing models: {e}", exc_info=True); sys.exit(1)
# -----------------------------------------------------------------------------


# =============================================================================
# --- FUNCTION DEFINITIONS (Steps 1-10) ---
# =============================================================================

# (Functions 1-2a remain unchanged)
# ------------------ STEP 1: I/O UTILS --------------------------------------
def load_manual(jsonl_path_str: str) -> Tuple[List[Dict], str, str]:
    """Loads manual data from a JSONL file."""
    jsonl_path = Path(jsonl_path_str)
    if not jsonl_path.is_file(): raise FileNotFoundError(f"Input file not found at {jsonl_path_str}")
    pages = []
    print(f"Attempting to load manual from: {jsonl_path}")
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line: continue
                try: page_data = json.loads(line)
                except json.JSONDecodeError: logging.warning(f"Skipping line {i+1} invalid JSON."); continue
                if "page_num" not in page_data or "markdown_content" not in page_data: logging.warning(f"Line {i+1} missing keys. Skipping."); continue
                if not isinstance(page_data.get("markdown_content"), str): page_data["markdown_content"] = str(page_data["markdown_content"]); logging.warning(f"Line {i+1} content not string. Converted.")
                pages.append(page_data)
    except Exception as e: logging.error(f"Unexpected error loading {jsonl_path_str}: {e}", exc_info=True); sys.exit(1)
    if not pages: logging.error(f"No valid pages loaded from {jsonl_path_str}."); sys.exit(1)
    print(f"Successfully loaded {len(pages)} pages from {jsonl_path_str}")
    first_page = pages[0]; doc_id = first_page.get("doc_id", "unknown_doc"); language = first_page.get("language", "unknown_lang")
    print(f"Detected doc_id: {doc_id}, language: {language} (from first page)")
    return pages, doc_id, language

# ------------------ STEP 2a Helper: Prompt Building -------------------------
def build_prompt(pages: List[Dict]) -> str:
    """Loads the master prompt and inserts the formatted page data."""
    try: master_prompt_text = MASTER_PROMPT_PATH.read_text(encoding='utf-8')
    except Exception as e: logging.error(f"Error reading master prompt file {MASTER_PROMPT_PATH}: {e}", exc_info=True); sys.exit(1)
    pages_jsonl_string = "\n".join([json.dumps(p, ensure_ascii=False) for p in pages])
    placeholder = "<PASTE ALL JSONL LINES FOR THIS MANUAL HERE>"
    if placeholder not in master_prompt_text: logging.error(f"Placeholder '{placeholder}' not found in {MASTER_PROMPT_PATH}"); sys.exit(1)
    full_prompt = master_prompt_text.replace(placeholder, pages_jsonl_string); return full_prompt

# ------------------ STEP 2a Helper: Generation ------------------------------
def over_generate(prompt: str, k: int, generation_model, generation_temperature: float) -> List[str]:
    """Calls the generation model k times to over-generate raw CSV rows."""
    all_raw_rows = []
    print(f"\n--- Starting Generation (x{k} calls, Temp={generation_temperature}, Target Lines ~{TARGET_RAW_ROWS}) ---")
    for i in range(k):
        print(f"Generation call {i+1} of {k}...")
        try:
            response = generation_model.generate_content(prompt, generation_config={"temperature": generation_temperature})
            raw_text = None
            if response.candidates:
                 if response.candidates[0].content and response.candidates[0].content.parts: raw_text = response.candidates[0].content.parts[0].text
            elif hasattr(response, 'text'): raw_text = response.text
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason: logging.warning(f"Call {i+1} potentially blocked: {response.prompt_feedback.block_reason}"); continue
            if raw_text:
                 generated_lines = [line for line in raw_text.splitlines() if line.strip()]
                 print(f"  -> Received {len(generated_lines)} non-empty lines.")
                 all_raw_rows.extend(generated_lines)
            else: logging.warning(f"Call {i+1} did not produce expected text output. Parts: {response.parts if hasattr(response, 'parts') else 'N/A'}")
        except Exception as e: logging.error(f"Error during generation call {i+1}: {e}", exc_info=True); print("  -> Attempting to continue...")
    print(f"--- Generation Complete: Collected {len(all_raw_rows)} raw rows total ---"); return all_raw_rows


# ------------------ STEP 3: PARSING -----------------------------------------
# (parse_rows function remains unchanged)
def parse_rows(raw_rows: List[str], schema: List[str], doc_id: str, language: str) -> pd.DataFrame:
    """Parses raw CSV strings into DataFrame, initializes parsed_steps."""
    parsed_data = [];
    expected_columns = len(schema);
    row_num_counter = 0
    doc_prefix = doc_id.split('.')[0] if '.' in doc_id else doc_id
    print(f"\n--- Starting Parsing of {len(raw_rows)} raw rows ---")
    for i, raw_row in enumerate(raw_rows):
        try:
            reader = csv.reader([raw_row], quotechar='"', doublequote=True, skipinitialspace=True)
            cells = next(reader)
            if len(cells) != expected_columns: logging.warning(
                f"Row {i + 1} bad columns ({len(cells)}/{expected_columns}). Skipping."); continue
            row_dict = dict(zip(schema, cells));
            # Ensure values are stripped strings AFTER confirming dict creation
            row_dict = {k: v.strip() if isinstance(v, str) else v for k, v in row_dict.items()}
            row_dict['doc_id'] = doc_id;
            row_dict['language'] = language;
            row_num_counter += 1;
            row_dict['question_id'] = f"{doc_prefix}_Q{row_num_counter:03d}"
            # Check boolean string validity more explicitly
            if str(row_dict.get('_self_grounded')).strip().capitalize() not in ["True", "False"]:
                 row_dict['_self_grounded'] = "False" # Default to False if invalid
            else:
                 row_dict['_self_grounded'] = str(row_dict.get('_self_grounded')).strip().capitalize() # Standardize capitalization
            # Check page number validity
            if row_dict.get('gt_page_number') != "None":
                try:
                    # Try converting to int, but keep it as string in the dict for consistency
                    int(row_dict['gt_page_number'])
                except (ValueError, TypeError):
                    logging.warning(f"Row {i + 1} QID {row_dict['question_id']} invalid page number '{row_dict['gt_page_number']}'. Setting to None.")
                    row_dict['gt_page_number'] = "None"
            # Ensure persona and category exist for potential later validation
            if 'persona' not in row_dict: logging.warning(f"Row {i + 1} QID {row_dict['question_id']} missing 'persona'. Setting to empty string."); row_dict['persona'] = ""
            if 'category' not in row_dict: logging.warning(f"Row {i + 1} QID {row_dict['question_id']} missing 'category'. Setting to empty string."); row_dict['category'] = ""

            row_dict['parsed_steps'] = None  # Initialize column
            parsed_data.append(row_dict)
        except csv.Error as csv_e: # Be more specific about CSV errors
            logging.warning(f"CSV parsing error row {i + 1}: {csv_e}. Skipping.")
        except Exception as e:
            logging.warning(f"General parsing error row {i + 1}: {e}. Skipping.", exc_info=True); continue
    print(f"--- Parsing Complete: Successfully parsed {len(parsed_data)} rows ---")
    if not parsed_data: logging.error("No rows parsed."); return pd.DataFrame()
    # Convert to DataFrame
    df = pd.DataFrame(parsed_data)
    # Set column types explicitly if needed, e.g.,
    # df['gt_page_number'] = df['gt_page_number'].astype(str) # Ensure it remains string
    # df['_self_grounded'] = df['_self_grounded'].astype(str) # Ensure it remains string
    return df

# --- NEW Step 3a: Filter Invalid Persona/Category ---
def filter_invalid_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Filters out rows with invalid persona or category values."""
    if df.empty:
        logging.info("Filter Invalid Metadata: Skipping, input DataFrame is empty.")
        return df

    if 'persona' not in df.columns or 'category' not in df.columns:
        logging.error("Filter Invalid Metadata: Missing 'persona' or 'category' column. Skipping filter.")
        return df

    n_rows_before = len(df)
    print(f"\n--- Step 3a: Filtering Invalid Persona/Category (Valid Personas: {len(VALID_PERSONAS)}, Valid Categories: {len(VALID_CATEGORIES)}) ---")

    # Create masks for valid rows
    valid_persona_mask = df['persona'].isin(VALID_PERSONAS)
    valid_category_mask = df['category'].isin(VALID_CATEGORIES)

    # Combine masks: keep rows where BOTH are valid
    combined_mask = valid_persona_mask & valid_category_mask

    # Identify and log invalid rows before filtering
    invalid_rows = df[~combined_mask]
    if not invalid_rows.empty:
        invalid_personas_found = invalid_rows[~valid_persona_mask]['persona'].unique()
        invalid_categories_found = invalid_rows[~valid_category_mask]['category'].unique()
        logging.warning(f"Found {len(invalid_rows)} rows with invalid metadata:")
        if len(invalid_personas_found) > 0:
            logging.warning(f"  - Invalid Personas encountered: {list(invalid_personas_found)}")
        if len(invalid_categories_found) > 0:
            logging.warning(f"  - Invalid Categories encountered: {list(invalid_categories_found)}")
        # Can optionally log the QIDs of removed rows if needed for debugging:
        # logging.warning(f"  - QIDs removed: {invalid_rows['question_id'].tolist()}")

    # Apply the filter
    df_filtered = df[combined_mask].reset_index(drop=True)
    n_rows_after = len(df_filtered)
    n_removed = n_rows_before - n_rows_after

    print(f"Filter Invalid Metadata: Kept {n_rows_after} / {n_rows_before} rows (Removed {n_removed}).")
    print("\n--- Step 3a Complete (Invalid Metadata Filter) ---")
    return df_filtered

# --- Step 3b: Add Parsed Steps via LLM ---
# (llm_parse_snippet_to_steps helper unchanged)
def llm_parse_snippet_to_steps(snippet: str, parsing_model_name: str, max_retries=MAX_PARSE_RETRIES) -> List[str]:
    """
    Uses the specified LLM (Gemini or OpenAI) to parse a text snippet into a
    list of procedural steps, using few-shot examples in the prompt and retries on rate limits.
    Returns None if parsing fails or yields no steps.
    """
    if not snippet or not isinstance(snippet, str): return None

    prompt = f"""Parse the following text, which represents procedural steps from a technical manual, into a list of distinct steps. A distinct step typically represents a single complete action or instruction.

    **Instructions:**
    1. Preserve the meaning and approximate number of steps accurately.
    2. Maintain the original wording within each step as much as possible.
    3. Treat text separated by ellipses ('...') as part of the SAME step unless a new action clearly begins after the ellipses. Merge the related text.
    4. Try to logically rejoin words that might be hyphenated across line breaks in the input text.
    5. Ignore non-standard bullet points or markers (like '', '–') at the beginning of lines when determining step content, focus on the actions described.
    6. Output ONLY a single, valid, Python-style list of strings (e.g., ["Step 1...", "Step 2..."]). Do not include any other text, explanations, or markdown formatting.

    --- Examples ---

    Example 1:
    Text to Parse:
    ```
     Pull the upper filter out.
     Remove fluff . . .
    . . . from surfaces and de-
    flector.
     Push filter back.
    ```
    Parsed List:
    ["Pull the upper filter out.", "Remove fluff from surfaces and deflector.", "Push filter back."]

    Example 2:
    Text to Parse:
    ```
    1. Attach tool.
    2. Unscrew anti-clockwise.
    3. Insert new part. Note: Ensure
    correct orientation.
    4. Screw clockwise.
    ```
    Parsed List:
    ["Attach tool.", "Unscrew anti-clockwise.", "Insert new part. Note: Ensure correct orientation.", "Screw clockwise."]

    Example 3:
    Text to Parse:
    ```
    Remove any fluff after every drying\ncycle.\n\n Pull the upper fluff filter forwards to\nremove it.\n\n Remove the fluff (see arrows) . . .\n\n . . . from the surfaces of all the fluff\n\nfilters.\n\n . . . from the perforated laundry de‐\nflector.\n\n Push the upper fluff filter back into\nposition until it clicks.
    ```
    Parsed List:
    ["Remove any fluff after every drying cycle.", "Pull the upper fluff filter forwards to remove it.", "Remove the fluff (see arrows) from the surfaces of all the fluff filters and from the perforated laundry deflector.", "Push the upper fluff filter back into position until it clicks."]

    --- Task ---

    Text to Parse:
    ```
    {snippet}
    ```

    Parsed List:
    """

    for attempt in range(max_retries):
        try:
            logging.debug(f"Parsing attempt {attempt+1}/{max_retries} using {parsing_model_name}")
            raw_text = None
            # --- Select API based on model name ---
            if "gemini" in parsing_model_name.lower(): # Gemini Call
                # Assumes genai is imported and configured globally
                parser_model = genai.GenerativeModel(parsing_model_name)
                response = parser_model.generate_content(prompt, generation_config={"temperature": 0.0})
                if response.candidates and response.candidates[0].content and response.candidates[0].content.parts: raw_text = response.candidates[0].content.parts[0].text
                elif hasattr(response, 'text'): raw_text = response.text

            elif "gpt" in parsing_model_name.lower(): # OpenAI Call
                 # Assumes openai is imported and configured globally
                 response = openai.chat.completions.create(model=parsing_model_name, messages=[{"role": "user", "content": prompt}], temperature=0.0, max_tokens=500, n=1)
                 if response.choices and response.choices[0].message.content: raw_text = response.choices[0].message.content.strip()
            else: logging.error(f"Unsupported parser model: {parsing_model_name}"); return None

            logging.debug(f"LLM Parser Raw Response (Attempt {attempt+1}): '{raw_text}'")

            # --- Process response ---
            if raw_text:
                try:
                    cleaned_text = re.sub(r"^```python\s*|```$", "", raw_text.strip(), flags=re.MULTILINE).strip()
                    logging.debug(f"LLM Parser Cleaned Text: '{cleaned_text}'")
                    parsed_list = None
                    if cleaned_text.startswith('[') and cleaned_text.endswith(']'):
                        logging.debug("Attempting ast.literal_eval...")
                        parsed_list = ast.literal_eval(cleaned_text)
                    else: logging.warning(f"LLM Parser output not list format: '{cleaned_text[:100]}...' Trying line split."); parsed_list = [line.strip() for line in cleaned_text.splitlines() if line.strip()]

                    if isinstance(parsed_list, list) and all(isinstance(item, str) for item in parsed_list):
                        final_steps = [step.strip() for step in parsed_list if step.strip()]
                        if final_steps: logging.debug(f"LLM Parser SUCCESS -> Parsed List: {final_steps}"); return final_steps
                        else: logging.warning(f"LLM Parser parsed to EMPTY list: '{cleaned_text[:100]}...'")
                    else: logging.warning(f"LLM Parser output not list of strings: Type={type(parsed_list)}")
                except Exception as parse_error: logging.warning(f"LLM Parser output eval failed: {parse_error}. Cleaned Text: '{cleaned_text}'")
            else: logging.warning(f"LLM Parser call ({parsing_model_name}) returned no text.")

            if attempt < max_retries - 1: logging.warning(f"LLM Parsing failed attempt {attempt+1}. Retrying..."); time.sleep(1); continue
            else: logging.error(f"LLM Parsing failed after {max_retries} attempts."); return None

        except (openai.RateLimitError, google.api_core.exceptions.ResourceExhausted) as e: # Rate Limit
            logging.warning(f"Rate limit hit... Waiting {RETRY_DELAY_SECONDS}s...");
            if attempt < max_retries - 1: time.sleep(RETRY_DELAY_SECONDS); continue
            else: logging.error(f"Rate limit persists. Skipping snippet."); return None
        except Exception as e: # Other API errors
             if "gpt" in parsing_model_name.lower() and openai and "The model" in str(e) and "does not exist" in str(e): logging.error(f"Step Parsing Model Error: '{parsing_model_name}' invalid."); raise e
             logging.error(f"API Error LLM parsing ({parsing_model_name}): {e}", exc_info=True); return None

    logging.error(f"LLM Parsing failed all retries."); return None

# (add_parsed_steps_llm function unchanged)
def add_parsed_steps_llm(df: pd.DataFrame, parsing_model_name: str) -> pd.DataFrame:
    """Adds/Updates 'parsed_steps' column by parsing procedural snippets using an LLM."""
    if df.empty: logging.info("Add Parsed Steps: Skipping, input empty."); return df.assign(parsed_steps=None)
    if 'category' not in df.columns or 'gt_answer_snippet' not in df.columns: logging.error(
        "Missing columns for step parsing."); return df.assign(parsed_steps=None)
    print(f"\n--- Step 3b: Parsing Procedural Snippets using LLM ({parsing_model_name}) ---")
    parsed_steps_col_data = []
    rows_to_parse_indices = df[df['category'] == PROCEDURAL_CATEGORY_NAME].index
    num_to_parse = len(rows_to_parse_indices);
    print(f"Found {num_to_parse} rows in category '{PROCEDURAL_CATEGORY_NAME}' to parse.")
    parsed_count = 0
    for index, row in df.iterrows():
        if index in rows_to_parse_indices:
            snippet = row.get('gt_answer_snippet')
            if isinstance(snippet, str) and snippet:
                parsed_list = llm_parse_snippet_to_steps(snippet, parsing_model_name)
                parsed_steps_col_data.append(parsed_list)
                if parsed_list is not None: parsed_count += 1
                time.sleep(0.1)  # Small delay
            else:
                logging.warning(f"QID {row.get('question_id')} procedural bad snippet."); parsed_steps_col_data.append(
                    None)
        else:
            parsed_steps_col_data.append(None) # Append None for non-procedural rows
    # Ensure the length matches before assigning
    if len(parsed_steps_col_data) != len(df):
         logging.error(f"Parsed steps list length ({len(parsed_steps_col_data)}) != DataFrame length ({len(df)}). Cannot assign column.")
         return df # Return original df to prevent error
    df['parsed_steps'] = parsed_steps_col_data
    print(f"LLM Step Parsing complete. Successfully parsed steps for {parsed_count} / {num_to_parse} rows.")
    df['parsed_steps'] = df['parsed_steps'].astype(object); # Ensure dtype is object for lists
    print("\n--- Step 3b Complete (LLM Step Parsing) ---")
    return df


# ------------------ STEP 4: Deduplication Filter -----------------------------
# (deduplicate function unchanged)
def deduplicate(df: pd.DataFrame, embedder: SentenceTransformer, threshold: float, model_name_str: str) -> pd.DataFrame:
    """Removes rows with questions too similar to preceding questions."""
    if df.empty or 'question_text' not in df.columns: logging.info("Deduplication: Skipping."); return df
    questions = df["question_text"].tolist();
    n_rows_before = len(df);
    print(f"\n--- Step 4: Deduplicating Questions ---")
    print(f"Deduplication: Encoding {n_rows_before} questions using '{model_name_str}'...")
    embeddings = embedder.encode(questions, normalize_embeddings=True, show_progress_bar=True);
    indices_to_keep = [];
    print(f"Deduplication: Comparing questions with threshold {threshold:.2f}...")
    for i, emb_i in enumerate(embeddings):
        is_duplicate = False
        for j in indices_to_keep:
            emb_j = embeddings[j];
            similarity = util.cos_sim(emb_i, emb_j).item()
            if similarity >= threshold: is_duplicate = True; break
        if not is_duplicate: indices_to_keep.append(i)
    deduplicated_df = df.iloc[indices_to_keep].reset_index(drop=True);
    n_rows_after = len(deduplicated_df);
    n_removed = n_rows_before - n_rows_after;
    print(f"Deduplication: Kept {n_rows_after} out of {n_rows_before} rows (Removed {n_removed}).");
    print(f" Shape after Dedupe: {deduplicated_df.shape}"); print("\n--- Step 4 Complete ---")
    return deduplicated_df


# ------------------ STEP 5: Page Check Annotation ----------------------------
# (page_check_annotate function unchanged)
def page_check_annotate(df: pd.DataFrame, pages_data: List[Dict]) -> pd.DataFrame:
    """Annotates DataFrame with 'passed_strict_check' boolean column."""
    if df.empty: logging.info("Page Check Annotate: Skipping."); df['passed_strict_check'] = pd.Series(
        dtype='boolean'); return df
    if not pages_data: logging.warning("Page Check Annotate: Missing pages_data."); df[
        'passed_strict_check'] = False; return df
    print(f"\n--- Step 5: Annotating Snippet Grounding ---")
    print(f"Page Check Annotate: Verifying strict snippet presence for {len(df)} rows...")
    page_lookup = {page['page_num']: page.get('markdown_content', '') for page in pages_data if
                   isinstance(page.get('page_num'), int) and isinstance(page.get('markdown_content'), str)}
    if not page_lookup: logging.warning("Page Check Annotate: Failed lookup creation."); df[
        'passed_strict_check'] = False; return df
    passed_strict_list = []
    for index, row in df.iterrows():
        passed = False;
        category = row.get('category');
        snippet = row.get('gt_answer_snippet');
        page_num_str = row.get('gt_page_number')
        if category == UNANSWERABLE_CATEGORY_NAME or page_num_str == "None":
            passed = True # Unanswerable or None page pass automatically for quota selection purposes
        else:
            if isinstance(snippet, str) and snippet and page_num_str != "None":
                try:
                    page_num_int = int(page_num_str);
                    page_content = page_lookup.get(page_num_int)
                    if page_content is not None and isinstance(page_content,
                                                               str) and snippet in page_content: passed = True
                except (ValueError, TypeError):
                    # Log if page number format was bad?
                    # logging.debug(f"QID {row.get('question_id')} page num '{page_num_str}' invalid format for lookup.")
                    pass # Keep passed as False
        passed_strict_list.append(passed)

    # Ensure the list length matches before assigning
    if len(passed_strict_list) != len(df):
        logging.error(f"Strict check list length ({len(passed_strict_list)}) != DataFrame length ({len(df)}). Cannot assign column.")
        df['passed_strict_check'] = pd.Series(dtype='boolean'); # Assign empty/default
    else:
        df['passed_strict_check'] = pd.Series(passed_strict_list, index=df.index, dtype='boolean')  # Ensure index alignment

    n_passed = df['passed_strict_check'].sum();
    n_failed = len(df) - n_passed;
    print(f"Page Check Annotate: Marked {n_passed} rows pass strict check, {n_failed} fail.");
    print(f" Shape after Annotation: {df.shape}"); print("\n--- Step 5 Complete ---")
    return df


# ------------------ STEP 6: RAGAS Filter ------------------------------------
# (ragas_filter function unchanged)
def ragas_filter(df: pd.DataFrame, pages_data: List[Dict], threshold: float) -> pd.DataFrame:
    """Filters DataFrame based on Ragas faithfulness and answer_correctness scores."""
    if 'ragas_eval' not in globals() or 'faithfulness' not in globals() or 'answer_correctness' not in globals() or 'Dataset' not in globals():
        logging.warning(
            "Required Ragas/Datasets components not available (import likely failed). Skipping RAGAS filter.")
        return df

    if df.empty: logging.info("Ragas Filter: Skipping."); return df
    if not pages_data: logging.warning("Ragas Filter: Missing pages_data. Skipping."); return df
    n_rows_before = len(df);
    print(f"\n--- Step 6: Filtering with Ragas ---")
    print(f"Ragas Filter: Preparing data for {n_rows_before} rows...")
    page_lookup = {page['page_num']: page.get('markdown_content', '') for page in pages_data if
                   isinstance(page.get('page_num'), int) and isinstance(page.get('markdown_content'), str)}
    if not page_lookup: logging.warning("Ragas Filter: Failed page lookup creation. Skipping."); return df
    ragas_data_list = [];
    original_indices = [];
    skipped_count = 0
    for index, row in df.iterrows():
        category = row.get('category')
        if category == UNANSWERABLE_CATEGORY_NAME: skipped_count += 1; continue
        question = row.get('question_text');
        answer = row.get('gt_answer_snippet');
        page_num_str = row.get('gt_page_number')
        if not all([isinstance(question, str), question, isinstance(answer, str), answer,
                    page_num_str != "None"]): logging.warning(
            f"Ragas Filter: Skipping QID {row.get('question_id')} due to missing Q/A/Page."); skipped_count += 1; continue
        try:
            page_num_int = int(page_num_str);
            page_content = page_lookup.get(page_num_int)
            if page_content is None or not isinstance(page_content, str) or not page_content.strip(): logging.warning(
                f"Ragas Filter: Skipping QID {row.get('question_id')} - invalid page content."); skipped_count += 1; continue
            ragas_data_list.append(
                {"question": question, "answer": answer, "contexts": [page_content], "reference": answer});
            original_indices.append(index)
        except (ValueError, TypeError):
            logging.warning(
                f"Ragas Filter: Skipping QID {row.get('question_id')} - page num parse error."); skipped_count += 1; continue
    if not ragas_data_list: logging.warning(
        "Ragas Filter: No valid data to evaluate. Keeping only Unanswerable."); return df[
        df['category'] == UNANSWERABLE_CATEGORY_NAME].copy().reset_index(drop=True)
    print(f"Ragas Filter: Prepared {len(ragas_data_list)} rows for evaluation (skipped {skipped_count}).")
    ragas_features = Features(
        {'question': Value('string'), 'answer': Value('string'), 'contexts': Sequence(Value('string')),
         'reference': Value('string')})
    eval_dataset = Dataset.from_list(ragas_data_list, features=ragas_features)
    metrics = [faithfulness, answer_correctness];
    scores_df = None
    try:
        print(
            "Running Ragas evaluation (faithfulness, answer_correctness)... This may take time and requires API calls.")
        results = ragas_eval(dataset=eval_dataset, metrics=metrics, raise_exceptions=False);
        scores_df = results.to_pandas()
    except Exception as e:
        logging.error(f"Ragas evaluation failed: {e}", exc_info=True); logging.warning(
            "Ragas Filter: Skipping RAGAS filtering due to error."); return df
    if scores_df is None or scores_df.empty: logging.warning("Ragas Filter: No scores returned. Skipping."); return df
    if not {"faithfulness", "answer_correctness"}.issubset(scores_df.columns): logging.warning(
        f"Ragas Filter: Score columns missing {scores_df.columns}. Skipping."); return df
    scores_df = scores_df.fillna(0.0);
    ragas_mask = (scores_df["faithfulness"] >= threshold) & (scores_df["answer_correctness"] >= threshold)
    passed_indices_in_scores_df = scores_df[ragas_mask].index;
    original_indices_passed_ragas = {original_indices[i] for i in passed_indices_in_scores_df}
    unanswerable_indices = set(df[df['category'] == UNANSWERABLE_CATEGORY_NAME].index);
    final_indices_to_keep = list(original_indices_passed_ragas.union(unanswerable_indices))
    df_filtered = df.loc[list(final_indices_to_keep)].sort_index().reset_index(drop=True);
    n_rows_after = len(df_filtered);
    n_removed = n_rows_before - n_rows_after;
    print(f"Ragas Filter: Kept {n_rows_after} / {n_rows_before} rows (Removed {n_removed}).");
    print(f" Shape after Ragas: {df_filtered.shape}"); print("\n--- Step 6 Complete ---")
    return df_filtered


# ------------------ STEP 7: LLM Judge Filter --------------------------------
# (gpt_judge_score helper unchanged)
def gpt_judge_score(question: str, answer_snippet: str, context: str, judge_model_name: str) -> int:
    """Calls the specified OpenAI model to score the QA pair based on context."""
    default_score = 1;
    system_prompt = "You are a strict QA pair evaluator. You will be given a Question, an Answer Snippet supposedly extracted from a Context document, and the Context itself. Evaluate if the Answer Snippet is a good and correct answer to the Question, based *only* on the provided Context. Output ONLY a single integer score from 1 to 5 based on the following scale:"
    rubric = """
    5: Snippet fully, correctly, and clearly answers the Question based *only* on the Context.
    4: Snippet answers the Question correctly based on the Context, but might be slightly awkward, verbose, or contain very minor irrelevant info from the context.
    3: Snippet partially answers the Question based on the Context or contains minor inaccuracies according to the Context.
    2: Snippet is related to the Question but is significantly inaccurate or fails to answer the core of the Question based on the Context.
    1: Snippet is irrelevant to the Question or completely wrong according to the Context.
    """
    user_prompt = f"Context:\n```\n{context}\n```\n\nQuestion:\n```\n{question}\n```\n\nAnswer Snippet:\n```\n{answer_snippet}\n```\n\nScore (1-5):"
    try:
        if not openai: raise ImportError("OpenAI library not available.")
        response = openai.chat.completions.create(model=judge_model_name, messages=[
            {"role": "system", "content": system_prompt + "\n" + rubric}, {"role": "user", "content": user_prompt}],
                                                  temperature=0, max_tokens=5, n=1)
        content = response.choices[0].message.content.strip();
        match = re.search(r'\d', content)
        if match:
            score = int(match.group()); return score if 1 <= score <= 5 else default_score
        else:
            logging.warning(f"Judge no digit ('{content}')."); return default_score
    except openai.RateLimitError:
        logging.warning(f"Judge rate limit hit."); return default_score
    except Exception as e:  # (Error handling unchanged) ...
        if "The model" in str(e) and "does not exist" in str(e): logging.error(
            f"Judge Model Error: '{judge_model_name}' invalid."); raise e
        logging.error(f"Judge API error: {e}");
        return default_score

# (judge_filter function unchanged)
def judge_filter(df: pd.DataFrame, pages_data: List[Dict], judge_model_name: str, threshold: int) -> pd.DataFrame:
    """Filters DataFrame based on scores from an LLM judge."""
    global HAS_OPENAI  # Assumes openai was imported if needed
    openai_available = 'openai' in sys.modules and hasattr(sys.modules['openai'], 'chat')
    if not openai_available: logging.warning(
        "OpenAI client not available/configured. Skipping Judge filter."); return df
    if df.empty: logging.info("Judge Filter: Skipping."); return df
    if not pages_data: logging.warning("Judge Filter: Missing pages_data."); return df
    n_rows_before = len(df);
    print(f"\n--- Step 7: Filtering Answerable with LLM Judge ---")
    print(f"Judge Filter: Preparing {n_rows_before} rows (model '{judge_model_name}')...")
    page_lookup = {page['page_num']: page.get('markdown_content', '') for page in pages_data if
                   isinstance(page.get('page_num'), int) and isinstance(page.get('markdown_content'), str)}
    if not page_lookup: logging.warning("Judge Filter: Failed lookup creation."); return df
    scores = {};
    indices_to_judge = df[df['category'] != UNANSWERABLE_CATEGORY_NAME].index;
    num_to_judge = len(indices_to_judge)
    print(f"Judge Filter: Evaluating {num_to_judge} answerable rows...")
    for i, index in enumerate(indices_to_judge):  # ... (Loop and call to gpt_judge_score unchanged) ...
        if (i + 1) % 10 == 0: print(f"  Judging row {i + 1} of {num_to_judge}...")
        row = df.loc[index];
        question = row.get('question_text');
        answer_snippet = row.get('gt_answer_snippet');
        page_num_str = row.get('gt_page_number')
        if not all([isinstance(q, str) and q for q in [question, answer_snippet]]) or page_num_str == "None": scores[
            index] = 1; continue
        try:
            page_num_int = int(page_num_str); page_content = page_lookup.get(page_num_int)
        except:
            scores[index] = 1; continue
        if not isinstance(page_content, str) or not page_content: scores[index] = 1; continue
        try:
            score = gpt_judge_score(question, answer_snippet, page_content, judge_model_name); scores[
                index] = score; time.sleep(JUDGE_DELAY_SECONDS)
        except Exception as e:
            logging.error(f"Judge Filter error QID {row.get('question_id')}: {e}", exc_info=True); scores[index] = 1
    passed_judge_mask = [(row['category'] == UNANSWERABLE_CATEGORY_NAME) or (scores.get(index, 0) >= threshold) for
                         index, row in df.iterrows()]
    df_filtered = df[pd.Series(passed_judge_mask)].reset_index(drop=True)
    n_rows_after = len(df_filtered);
    n_removed = n_rows_before - n_rows_after;
    print(f"Judge Filter: Kept {n_rows_after} / {n_rows_before} rows (Removed {n_removed}).");
    print(f" Shape after Judge: {df_filtered.shape}"); print("\n--- Step 7 Complete ---")
    return df_filtered


# ------------------ STEP 7b: Unanswerable Verification ----------------------
# (get_full_manual_text helper unchanged)
def get_full_manual_text(pages_data: List[Dict]) -> str:
    """Combines markdown content from all pages into a single string."""
    return "\n\n".join(
        [f"--- Page {p.get('page_num', 'N/A')} ---\n{p.get('markdown_content', '')}" for p in pages_data])

# (verify_unanswerable function unchanged)
def verify_unanswerable(df_unanswerable: pd.DataFrame, full_manual_text: str, judge_model_name: str) -> pd.Index:
    """Uses the specified JUDGE model (OpenAI API) to verify unanswerable questions."""
    openai_available = 'openai' in sys.modules and hasattr(sys.modules['openai'], 'chat')
    if not openai_available: logging.warning(
        "OpenAI client not available. Skipping Unanswerable verification."); return df_unanswerable.index
    if df_unanswerable.empty: return pd.Index([])
    print(f"\n--- Step 7b: Verifying Unanswerable Questions ---")
    print(f"Verifying {len(df_unanswerable)} unanswerable questions using JUDGE model {judge_model_name}...")
    indices_confirmed_unanswerable = [];
    unanswerable_confirmation_string = "Unanswerable"
    for index, row in df_unanswerable.iterrows():
        question = row.get('question_text');
        qid = row.get('question_id', 'N/A');
        if not isinstance(question, str) or not question: continue
        system_prompt = f"You are an assistant evaluating questions against a document. Based ONLY on the following document content, please answer the question concisely. If the document does not contain the information to answer the question, reply with the single word '{unanswerable_confirmation_string}' and nothing else."
        user_prompt = f"DOCUMENT CONTENT:\n```\n{full_manual_text}\n```\n\nQUESTION:\n```\n{question}\n```\n\nANSWER:"
        try:
            response = openai.chat.completions.create(model=judge_model_name,
                                                      messages=[{"role": "system", "content": system_prompt},
                                                                {"role": "user", "content": user_prompt}],
                                                      temperature=0.1, max_tokens=100)
        except openai.RateLimitError:
            logging.warning(f"Verifier rate limit hit QID {qid}. Keeping."); indices_confirmed_unanswerable.append(
                index); time.sleep(RETRY_DELAY_SECONDS); continue  # Apply retry delay on rate limit
        except Exception as e:  # (Error handling unchanged - keeping row on error) ...
            if "The model" in str(e) and "does not exist" in str(e): logging.error(
                f"Verifier Model Error: '{judge_model_name}' invalid."); raise e
            logging.error(f"Error verifying QID {qid}: {e}");
            indices_confirmed_unanswerable.append(index);
            time.sleep(JUDGE_DELAY_SECONDS);
            continue  # Apply delay even on error

        if response.choices and response.choices[0].message.content:
            answer_text = response.choices[0].message.content.strip();
            if answer_text.lower() == unanswerable_confirmation_string.lower():
                indices_confirmed_unanswerable.append(index)
            else:
                logging.info(f"QID {qid} 'Unanswerable' potentially found answer: '{answer_text[:100]}...'")
        else:
            logging.warning(
                f"Verifier ({judge_model_name}) no text output QID {qid}. Keeping row."); indices_confirmed_unanswerable.append(
                index)  # Keep on empty response
        time.sleep(JUDGE_DELAY_SECONDS)  # Delay after successful call

    print(f"Verification complete: Confirmed/Kept {len(indices_confirmed_unanswerable)} unanswerable rows.");
    print("\n--- Step 7b Complete ---")
    return pd.Index(indices_confirmed_unanswerable)


# ------------------ STEP 8: Quota Selection (Prioritized) ------------------
# (quota_select function unchanged)
def quota_select(df: pd.DataFrame, category_targets: Dict[str, int], final_size: int, random_seed: int = 42) -> Tuple[
    pd.DataFrame, Dict[str, Dict[str, int]]]:
    """Selects rows prioritizing 'passed_strict_check' == True. Returns df and details."""
    if df.empty: logging.error("Quota Selection: Input empty."); raise ValueError("Quota selection on empty DataFrame.")
    print(f"\n--- Step 8: Selecting Final Rows per Category Quota ---")
    print(f"Quota Selection: Selecting up to {final_size} rows, prioritizing strictly checked rows...")
    final_rows_list = [];
    quota_details = {};
    all_quotas_met = True
    if 'category' not in df.columns: logging.error("Quota Selection: 'category' missing."); raise ValueError(
        "'category' column missing.")
    has_strict_check_col = 'passed_strict_check' in df.columns;
    if not has_strict_check_col: logging.warning("Quota Selection: 'passed_strict_check' column missing. Performing random selection only.")
    available_categories = set(df['category'].unique())
    for category_name in category_targets.keys():
        if category_name not in available_categories: logging.warning(
            f"Quota Selection: Target category '{category_name}' not found in available data.")
    for category_name, target_count in category_targets.items():  # ... (Loop and prioritized selection logic unchanged) ...
        print(f"  Processing Category: '{category_name}' (Target: {target_count})")
        df_pool = df[df['category'] == category_name].copy();
        available_total = len(df_pool);
        if available_total == 0: logging.warning(f"  -> Quota Warning: No rows available."); quota_details[
            category_name] = {'needed': target_count, 'available': 0, 'strict_taken': 0, 'nonstrict_taken': 0,
                              'selected_total': 0}; all_quotas_met = False; continue  # Log details even if 0 available
        selected_rows_for_cat_list = [];
        strict_taken_count = 0;
        nonstrict_taken_count = 0
        if has_strict_check_col:
            df_strict_passed = df_pool[df_pool['passed_strict_check'] == True];
            df_strict_failed = df_pool[df_pool['passed_strict_check'] == False];
            available_strict = len(df_strict_passed);
            available_failed = len(df_strict_failed);
            print(f"  -> Available: {available_strict} strict=True, {available_failed} strict=False.")
            num_needed = target_count;
            # Take strict first
            num_to_take_strict = min(num_needed, available_strict)
            if num_to_take_strict > 0:
                 selected_strict = df_strict_passed if available_strict <= num_to_take_strict else df_strict_passed.sample(n=num_to_take_strict, random_state=random_seed)
                 selected_rows_for_cat_list.extend(selected_strict.to_dict('records'))
                 num_needed -= num_to_take_strict
                 strict_taken_count = num_to_take_strict
                 print(f"  -> Selected {strict_taken_count} strict.")
            # Fill remaining quota with non-strict if needed
            if num_needed > 0 and available_failed > 0:
                 num_to_take_failed = min(num_needed, available_failed)
                 selected_failed = df_strict_failed if available_failed <= num_to_take_failed else df_strict_failed.sample(n=num_to_take_failed, random_state=random_seed)
                 selected_rows_for_cat_list.extend(selected_failed.to_dict('records'))
                 num_needed -= num_to_take_failed
                 nonstrict_taken_count = num_to_take_failed
                 print(f"  -> Selected {nonstrict_taken_count} non-strict.")
            if num_needed > 0: logging.warning(
                f"  -> Quota Warning: Category '{category_name}' not fully met. Short by {num_needed}."); all_quotas_met = False
        else:  # Fallback: No strict check column, select randomly
            num_to_select = min(target_count, available_total);
            print(f"  -> Selecting {num_to_select} randomly (no strict check data).");
            if available_total < target_count: logging.warning(
                f"  -> Quota Warning: Only {available_total} available."); all_quotas_met = False
            selected_rows_df = df_pool.sample(n=num_to_select, random_state=random_seed);
            selected_rows_for_cat_list.extend(selected_rows_df.to_dict('records'));
            nonstrict_taken_count = num_to_select # All taken are non-strict in this fallback
        quota_details[category_name] = {'needed': target_count, 'available': available_total,
                                        'strict_taken': strict_taken_count, 'nonstrict_taken': nonstrict_taken_count,
                                        'selected_total': strict_taken_count + nonstrict_taken_count}
        final_rows_list.extend(selected_rows_for_cat_list)
    final_df = pd.DataFrame(final_rows_list);
    if not final_df.empty: final_df = final_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    print(f"\nQuota Selection: Total rows selected = {len(final_df)}")
    quota_details['_summary'] = {'target_total': final_size, 'selected_total': len(final_df),
                                 'all_quotas_met': all_quotas_met}
    if len(final_df) != final_size:
        logging.error(f"Quota Failure: Final size ({len(final_df)}) != target ({final_size}). This usually means not enough valid rows survived the filtering steps.");
        logging.error(f"Quota Details:\n{json.dumps(quota_details, indent=2)}");
        # Allow script to continue to save candidates/stats even if quota not met
        # sys.exit(1)
        logging.error("Proceeding with the rows selected, but final dataset size requirement NOT met.")
    else:
        print(f"Quota Selection: Successfully selected {len(final_df)} rows matching the target size.")
    print(f" Shape after Quota Select: {final_df.shape}"); print("\n--- Step 8 Complete ---")
    return final_df, quota_details


# ------------------ STEP 9: Audit Preparation --------------------------------
# (export_audit_slice function unchanged)
def export_audit_slice(df: pd.DataFrame, output_base_path: Path, audit_fraction: float, random_seed: int = 42):
    """
    Exports audit files:
    1. A/B CSVs: Random slice of NON-PROCEDURAL rows + 1 sentinel for Kappa check.
    2. Review CSV (_procedural_review.csv): ALL PROCEDURAL rows for expert step correction.
       Contains parsed_steps and empty columns for corrected_steps/procedural_comments.
    """
    if df.empty: logging.warning("Audit Export: Input DataFrame empty."); return

    # Separate procedural and non-procedural rows
    df_procedural = df[df['category'] == PROCEDURAL_CATEGORY_NAME].copy()
    df_non_procedural = df[df['category'] != PROCEDURAL_CATEGORY_NAME].copy()

    print(f"\n--- Step 9: Saving Candidates & Exporting Audit Files ---")

    # --- Save Candidates FIRST ---
    candidate_file_path = Path(f"{output_base_path}{CANDIDATE_SUFFIX}") # Define candidate path locally
    try:
         # Ensure parsed_steps is converted to object type before saving pickle if it contains lists
         if 'parsed_steps' in df.columns:
             df['parsed_steps'] = df['parsed_steps'].astype(object)
         if 'corrected_steps' in df.columns: # Ensure this exists and is object before pickle
             df['corrected_steps'] = df['corrected_steps'].astype(object)
         else: # Add if missing before pickle (should be added earlier, but safety)
             df['corrected_steps'] = None; df['corrected_steps'] = df['corrected_steps'].astype(object)

         df.to_pickle(candidate_file_path);
         logging.info(f"Final candidates saved to: {candidate_file_path} ({len(df)} rows)")
    except Exception as e:
         logging.error(f"Failed to save candidate DataFrame: {e}", exc_info=True);
         logging.warning("Continuing with audit file export, but candidate save failed.")


    # --- Stage 1 Export: Non-Procedural + Sentinel for Kappa ---
    num_non_proc = len(df_non_procedural)
    # Calculate sample size k based on *non-procedural* count for Stage 1 audit size
    k = max(3, int(round(num_non_proc * audit_fraction + 0.5)));
    k = min(k, num_non_proc); # Cannot sample more non-procedural rows than available
    num_real_rows_to_sample = max(0, k - NUM_SENTINELS) # NUM_SENTINELS is 1

    stage1_audit_sample_df = pd.DataFrame()
    sentinel_data_for_export = [SENTINEL_ROWS_INFO[0][0]];
    sentinel_df = pd.DataFrame(sentinel_data_for_export)
    core_cols_audit = ["question_id", "question_text", "gt_answer_snippet", "gt_page_number", "category", "persona"] # Define core cols here


    if num_real_rows_to_sample <= 0 :
         logging.warning(f"Audit Export (Stage 1): Sample size {k} allows only sentinel(s).")
         # Ensure sentinel_df has the core columns
         stage1_audit_sample_df = sentinel_df.reindex(columns=core_cols_audit)
    else:
        real_sample_df = df_non_procedural.sample(n=num_real_rows_to_sample, random_state=random_seed)
        # Ensure both real_sample and sentinel have the core columns before concat
        real_sample_df_aligned = real_sample_df.reindex(columns=core_cols_audit)
        sentinel_df_aligned = sentinel_df.reindex(columns=core_cols_audit)
        stage1_audit_sample_df = pd.concat(
            [real_sample_df_aligned, sentinel_df_aligned],
            ignore_index=True
        )

    # Shuffle the combined set for Stage 1
    stage1_audit_sample_df = stage1_audit_sample_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    final_k_stage1 = len(stage1_audit_sample_df)

    print(f"\nAudit Export (Stage 1): Preparing {final_k_stage1} rows ({num_real_rows_to_sample} non-procedural + {NUM_SENTINELS if final_k_stage1 >= NUM_SENTINELS else 0} sentinel) for Kappa check.")
    base_audit_df = stage1_audit_sample_df[core_cols_audit] # Use the guaranteed core columns
    label_cols = ['answer_correct?', 'grounded?', 'question_clear?', 'category_correct?', 'persona_tone_ok?']

    for rater in ["A", "B"]:
        audit_rater_df = base_audit_df.copy()
        for label_col in label_cols: audit_rater_df[label_col] = ""
        suffix = AUDIT_SUFFIX_A if rater == 'A' else AUDIT_SUFFIX_B
        audit_file_path = Path(f"{output_base_path}{suffix}")
        try:
            audit_rater_df.to_csv(audit_file_path, index=False, quoting=csv.QUOTE_ALL, encoding='utf-8')
            print(f"  -> Stage 1 Audit file saved: {audit_file_path}")
        except Exception as e: logging.error(f"Failed to save audit file {audit_file_path}: {e}", exc_info=True)

    # --- Stage 2 Export: All Procedural Rows for Expert Review ---
    num_procedural = len(df_procedural)
    print(f"\nAudit Export (Stage 2): Preparing {num_procedural} procedural rows for expert review.")
    # Define the file path for the review file
    review_file_path = Path(f"{output_base_path}_procedural_review.csv") # Define path here

    if num_procedural > 0:
        # Define columns needed for procedural review
        # Ensure 'parsed_steps' is included if it exists in df_procedural
        review_cols_base = ["question_id", "question_text", "gt_answer_snippet", "gt_page_number"]
        # Ensure parsed_steps is added ONLY IF the column exists in the procedural df
        review_cols = review_cols_base + (["parsed_steps"] if 'parsed_steps' in df_procedural.columns else [])

        cols_to_select_review = [col for col in review_cols if col in df_procedural.columns]
        if 'parsed_steps' not in cols_to_select_review and 'parsed_steps' in df_procedural.columns:
             logging.warning("Procedural Review Export: 'parsed_steps' column exists but wasn't selected?") # Should not happen

        procedural_review_df = df_procedural[cols_to_select_review].reset_index(drop=True)

        # Add empty columns for expert input
        procedural_review_df['corrected_steps'] = "" # Expert fills this
        procedural_review_df['procedural_comments'] = "" # Expert fills this

        try:
            # Represent list as string for CSV compatibility easily
            if 'parsed_steps' in procedural_review_df.columns:
                 # Ensure None becomes '[]' or similar safe string rep
                 procedural_review_df['parsed_steps'] = procedural_review_df['parsed_steps'].apply(lambda x: str(x) if x is not None else '[]')

            procedural_review_df.to_csv(review_file_path, index=False, quoting=csv.QUOTE_ALL, encoding='utf-8')
            print(f"  -> Stage 2 Procedural Review file saved: {review_file_path}")
            print(f"     -> Please edit 'corrected_steps' and 'procedural_comments' in this file.") # Added instruction hint
        except Exception as e:
            logging.error(f"Failed to save procedural review file {review_file_path}: {e}", exc_info=True)
    else:
        print("  -> No procedural rows found to export for Stage 2 review.")
        # Create an empty review file with headers if no procedural rows exist
        # This ensures the finalize step doesn't fail looking for the file if called later.
        try:
            review_cols_headers = ["question_id", "question_text", "gt_answer_snippet", "gt_page_number", "parsed_steps", "corrected_steps", "procedural_comments"]
            pd.DataFrame(columns=review_cols_headers).to_csv(review_file_path, index=False, quoting=csv.QUOTE_ALL, encoding='utf-8')
            print(f"  -> Created empty Stage 2 Procedural Review file: {review_file_path}")
        except Exception as e:
            logging.error(f"Failed to create empty procedural review file {review_file_path}: {e}", exc_info=True)

    print("\n--- Step 9 Complete (Audit/Review File Export) ---")


# --- MODIFIED Step 10 Function -> Now only checks Stage 1 Kappa ---
# (check_stage1_kappa function unchanged)
def check_stage1_kappa(output_base_path: Path, kappa_min_threshold: float) -> Dict[str, Any]:
    """
    Reads Stage 1 audit files (non-procedural + sentinel), checks sentinel accuracy,
    computes Cohen's Kappa on non-procedural/non-sentinel rows.
    Returns dict containing check status and scores.
    """
    audit_file_a_path = Path(f"{output_base_path}{AUDIT_SUFFIX_A}")
    audit_file_b_path = Path(f"{output_base_path}{AUDIT_SUFFIX_B}")
    label_cols = ['answer_correct?', 'grounded?', 'question_clear?', 'category_correct?', 'persona_tone_ok?']
    sentinel_id = SENTINEL_ROWS_INFO[0][0]['question_id']
    sentinel_expected_ratings = SENTINEL_ROWS_INFO[0][1]
    results = {"passed": False, "min_kappa": None, "kappa_scores": {}, "rater_A_acc": None, "rater_B_acc": None}

    print("\n--- Stage 1 Audit Check: Calculating Kappa & Checking Sentinel ---")
    try: # Load audit files
        df_a = pd.read_csv(audit_file_a_path); df_b = pd.read_csv(audit_file_b_path); logging.info(f"Loaded audit files.")
    except Exception as e: logging.error(f"Error reading audit files: {e}"); return results
    if len(df_a) != len(df_b) or len(df_a) == 0: logging.error("Audit files lengths differ/empty."); return results
    missing_cols = [col for col in label_cols + ['question_id'] if col not in df_a.columns or col not in df_b.columns]
    if missing_cols: logging.error(f"Missing columns: {missing_cols}"); return results

    # --- 1. Check Sentinel Accuracy ---
    print("\nChecking Rater Accuracy on Sentinel Question...")
    all_raters_passed = True; rater_acc = {}
    for rater_id, df_rater in [("A", df_a), ("B", df_b)]:
        # Ensure question_id is string for comparison
        df_rater['question_id'] = df_rater['question_id'].astype(str)
        rater_sentinel_row = df_rater[df_rater['question_id'] == sentinel_id].copy()
        accuracy = 0.0
        if len(rater_sentinel_row) == 1:
            correct_count = 0; total_ratings = 0; row = rater_sentinel_row.iloc[0]; expected = sentinel_expected_ratings;
            for col in label_cols:
                total_ratings += 1; rater_answer_raw = row.get(col); rater_answer = str(rater_answer_raw).strip().lower() if pd.notna(rater_answer_raw) else 'missing'; expected_answer = expected.get(col, 'undefined').lower()
                if rater_answer == expected_answer: correct_count += 1
                else: logging.warning(f" Rater {rater_id} Sentinel Mismatch: Col='{col}', Rater='{rater_answer}', Expected='{expected_answer}'")
            if total_ratings > 0: accuracy = correct_count / total_ratings
        else: logging.error(f"Rater {rater_id}: Sentinel '{sentinel_id}' issue ({len(rater_sentinel_row)} rows found)."); all_raters_passed = False; break # Added row count info
        results[f'rater_{rater_id}_acc'] = accuracy; print(f"  Rater {rater_id} Accuracy: {accuracy:.2%}")
        if accuracy < SENTINEL_ACCURACY_THRESHOLD: logging.error(f"Rater {rater_id} failed check."); all_raters_passed = False
    if not all_raters_passed: print("❌ Rater accuracy on sentinel too low."); return results # Return failure results
    else: print("✅ Rater accuracy on sentinels passed.")

    # --- 2. Filter out Sentinel for Kappa Calculation ---
    # Kappa calculated only on NON-SENTINEL rows from Stage 1 audit files
    df_a_real = df_a[df_a['question_id'] != sentinel_id]
    df_b_real = df_b[df_b['question_id'] != sentinel_id]
    num_real_rows = len(df_a_real)
    if num_real_rows == 0: # Handle case where only sentinel was in audit file
         logging.warning("No non-sentinel rows found in audit files to calculate Kappa on.")
         # If no real rows, should Kappa pass? Let's assume yes if sentinel check passed.
         results['passed'] = True; results['min_kappa'] = 1.0 # Assign perfect kappa if no data to disagree on
         return results

    # --- 3. Calculate Kappa on NON-SENTINEL Data ---
    print(f"\nCalculating Kappa on {num_real_rows} non-sentinel rows...")
    kappa_scores = {}; valid_kappa_found = False; possible_labels = ['yes', 'no']
    for col in label_cols:
        try: # (Kappa calculation loop is the same, just uses df_a_real, df_b_real)
            labels_a = df_a_real[col].fillna('missing').astype(str).str.lower(); labels_b = df_b_real[col].fillna('missing').astype(str).str.lower()
            valid_label_mask = labels_a.isin(possible_labels) & labels_b.isin(possible_labels); # ... (rest unchanged) ...
            if not valid_label_mask.all(): logging.warning(f"Invalid labels in '{col}'. Calculating Kappa on {valid_label_mask.sum()} valid pairs.")
            labels_a = labels_a[valid_label_mask]; labels_b = labels_b[valid_label_mask]
            if len(labels_a) < 2: logging.warning(f"Not enough valid ratings for '{col}'. Skipping Kappa."); kappa_scores[col] = float('nan'); continue
            if (labels_a == labels_b).all():
                logging.info(f"Perfect agreement found for '{col}'. Setting Kappa = 1.0.")
                k_score = 1.0
            else:
                k_score = cohen_kappa_score(labels_a, labels_b, labels=possible_labels)
            kappa_scores[col] = k_score; valid_kappa_found = True
        except Exception as e: logging.error(f"Error calculating Kappa for '{col}': {e}", exc_info=False); kappa_scores[col] = float('nan')

    results['kappa_scores'] = {k: (v if not pd.isna(v) else None) for k, v in kappa_scores.items()}
    if not valid_kappa_found: logging.error("Could not calculate Kappa for any column."); return results # passed=False

    # --- 4. Evaluate Minimum Kappa ---
    print(f"\nCohen's Kappa Scores (Non-Sentinel, Non-Procedural Only):"); [print(f"  - {col:<20}: {score:.3f}" if score is not None else f"  - {col:<20}: error/NaN") for col, score in results['kappa_scores'].items()]
    valid_scores = [s for s in results['kappa_scores'].values() if s is not None]
    if not valid_scores: logging.warning("No valid Kappa scores calculated."); min_k = -1.0 # Treat as failure if no scores
    else: min_k = min(valid_scores);
    results['min_kappa'] = min_k; print(f"\nMinimum valid Kappa score = {min_k:.3f}")

    # --- 5. Final Decision (Stage 1 ONLY) ---
    if min_k >= kappa_min_threshold:
        print(f"\n✅ Minimum Kappa meets threshold ({kappa_min_threshold:.2f}). Stage 1 Audit Passed.")
        results['passed'] = True
    else:
        print(f"\n❌ Minimum Kappa BELOW threshold ({kappa_min_threshold:.2f}). Agreement insufficient."); print("   Review guidelines/labels for Stage 1 audit. Cannot finalize dataset.")
        results['passed'] = False

    return results # Return results including pass/fail status

# -----------------------------------------------------------------------------

# --- Simulation Helpers for --finalize_test ---
# (_simulate_audit_files function unchanged)
def _simulate_audit_files(df_non_procedural: pd.DataFrame, output_base_path: Path, audit_fraction: float, random_seed: int = 42):
    """Creates dummy audit files (Stage 1) with high agreement for testing."""
    logging.info("Simulating Stage 1 audit file completion (High Agreement)...")
    # 1. Select sample (Non-procedural + Sentinel) - mirrors export_audit_slice logic
    df = df_non_procedural # Input df here is only non-procedural rows
    k = max(3, int(round(len(df) * audit_fraction + 0.5))); k = min(k, len(df))
    num_real_rows_to_sample = max(0, k - NUM_SENTINELS) # NUM_SENTINELS is 1
    sentinel_data = [SENTINEL_ROWS_INFO[0][0]]; sentinel_df = pd.DataFrame(sentinel_data)
    audit_sample_df = pd.DataFrame()
    core_cols_audit = ["question_id", "question_text", "gt_answer_snippet", "gt_page_number", "category", "persona"] # Define core cols here

    if num_real_rows_to_sample <= 0 :
        audit_sample_df = sentinel_df.reindex(columns=core_cols_audit) # Ensure sentinel has core cols
    else:
        real_sample_df = df.sample(n=num_real_rows_to_sample, random_state=random_seed)
        # Ensure both real_sample and sentinel have the core columns
        real_sample_df_aligned = real_sample_df.reindex(columns=core_cols_audit)
        sentinel_df_aligned = sentinel_df.reindex(columns=core_cols_audit)
        audit_sample_df = pd.concat([real_sample_df_aligned, sentinel_df_aligned], ignore_index=True)

    audit_sample_df = audit_sample_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    base_audit_df = audit_sample_df[core_cols_audit] # Use the guaranteed core columns

    # 2. Generate simulated ratings (mostly Yes, sentinel correct No's)
    label_cols = ['answer_correct?', 'grounded?', 'question_clear?', 'category_correct?', 'persona_tone_ok?']
    sentinel_id = SENTINEL_ROWS_INFO[0][0]['question_id']
    sentinel_expected = SENTINEL_ROWS_INFO[0][1]
    simulated_ratings = []
    for index, row in base_audit_df.iterrows():
        # Ensure comparison uses string type for question_id
        is_sentinel = str(row['question_id']) == str(sentinel_id)
        # Use .get() for category to handle potential missing column safely
        is_unanswerable = row.get('category') == UNANSWERABLE_CATEGORY_NAME
        ratings = {}
        for col in label_cols:
            if is_sentinel: ratings[col] = sentinel_expected.get(col, 'no')
            elif is_unanswerable and col in ['answer_correct?', 'grounded?']: ratings[col] = 'no'
            else: ratings[col] = 'yes' # Default to 'yes' for real rows in simulation
        simulated_ratings.append(ratings)
    ratings_df = pd.DataFrame(simulated_ratings, index=base_audit_df.index) # Align index

    # 3. Save simulated files for Rater A and Rater B (identical for high agreement)
    for rater in ["A", "B"]:
        audit_rater_df = pd.concat([base_audit_df, ratings_df], axis=1)
        suffix = AUDIT_SUFFIX_A if rater == 'A' else AUDIT_SUFFIX_B
        audit_file_path = Path(f"{output_base_path}{suffix}")
        try:
            audit_rater_df.to_csv(audit_file_path, index=False, quoting=csv.QUOTE_ALL, encoding='utf-8')
            logging.info(f"  -> Simulated audit file saved: {audit_file_path}")
        except Exception as e: logging.error(f"Failed to save simulated audit file {audit_file_path}: {e}", exc_info=True)

# (_simulate_review_file_edit function unchanged)
def _simulate_review_file_edit(review_file_path: Path):
    """
    Simulates editing the procedural review file for finalize_test mode.
    Reads the review file, copies 'parsed_steps' to 'corrected_steps',
    sets 'procedural_comments' to empty, and saves the file back.
    """
    logging.info(f"Simulating edit of procedural review file: {review_file_path.name}")
    if not review_file_path.is_file():
        logging.error(f"Cannot simulate review edit: File not found at {review_file_path}")
        # Depending on desired strictness, could raise error or just return False
        return False

    try:
        df_review = pd.read_csv(review_file_path, keep_default_na=False) # keep_default_na=False helps with empty strings

        # Check required columns
        required_cols = ['parsed_steps', 'corrected_steps']
        # Also check comments column exists or add it
        if 'procedural_comments' not in df_review.columns:
            df_review['procedural_comments'] = ""

        missing_cols = [col for col in required_cols if col not in df_review.columns]
        if missing_cols:
            logging.error(f"Review file {review_file_path.name} missing columns for simulation: {missing_cols}")
            return False

        # Perform the simulation: Copy parsed to corrected, clear comments
        logging.info("  -> Copying 'parsed_steps' to 'corrected_steps' and clearing 'procedural_comments'.")
        df_review['corrected_steps'] = df_review['parsed_steps']
        df_review['procedural_comments'] = "" # Ensure it's cleared

        # Write the modified DataFrame back to the review file
        df_review.to_csv(review_file_path, index=False, quoting=csv.QUOTE_ALL, encoding='utf-8')
        logging.info(f"  -> Successfully simulated edit and saved {review_file_path.name}")
        return True

    except Exception as e:
        logging.error(f"Error during simulation of review file edit ({review_file_path.name}): {e}", exc_info=True)
        return False

# --- Main execution block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate & process QA dataset drafts.")
    parser.add_argument("-i", "--input", dest="input_jsonl", required=True,
                        help="Path to input JSONL file.")
    parser.add_argument("--finalize", action="store_true",
                        help="Run finalization using HUMAN audit files.")
    parser.add_argument("--finalize_test", action="store_true",
                        help="Simulate successful audit and finalize for testing.")
    args = parser.parse_args()

    # --- Determine File Paths ---
    input_file_path = Path(args.input_jsonl)
    if not input_file_path.is_file(): logging.error(f"Input file not found: {input_file_path}"); sys.exit(1)
    output_base_path = input_file_path.parent / input_file_path.stem.replace("_pages", "")
    try: output_base_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e: logging.error(f"Could not create output dir: {e}"); sys.exit(1)

    candidate_file_path = Path(f"{output_base_path}{CANDIDATE_SUFFIX}")
    audit_file_a = Path(f"{output_base_path}{AUDIT_SUFFIX_A}")
    audit_file_b = Path(f"{output_base_path}{AUDIT_SUFFIX_B}")
    raw_output_file_path = Path(f"{output_base_path}{RAW_OUTPUT_SUFFIX}")
    gold_file_path = Path(f"{output_base_path}{GOLD_DATASET_SUFFIX}")
    stats_file_path = Path(f"{output_base_path}{STATS_SUFFIX}")
    # This is the ONLY file for procedural review/correction now
    procedural_review_file_path = Path(f"{output_base_path}_procedural_review.csv")


    # --- MODE SELECTION ---
    if args.finalize or args.finalize_test:
        # --- Finalize Mode OR Test Finalize Mode ---
        # (Finalize logic unchanged)
        is_test_mode = args.finalize_test
        mode = "TEST Finalization" if is_test_mode else "Finalization"
        print(f"--- Running in {mode.upper()} mode for base path: {output_base_path} ---")

        # --- Load Candidate Data ---
        if not candidate_file_path.is_file(): logging.error(f"Candidate file missing: {candidate_file_path}"); sys.exit(1)
        try:
            df = pd.read_pickle(candidate_file_path); logging.info(f"Loaded {len(df)} candidate rows.")
            # Ensure list-like columns are treated as objects after loading
            if 'parsed_steps' in df.columns: df['parsed_steps'] = df['parsed_steps'].astype(object)
            if 'corrected_steps' in df.columns: df['corrected_steps'] = df['corrected_steps'].astype(object)

        except Exception as e: logging.error(f"Error loading candidate data: {e}", exc_info=True); sys.exit(1)
        if df.empty: logging.error("Loaded candidate file is empty."); sys.exit(1)

        # --- Simulate audit files if in test mode ---
        if is_test_mode:
             print("\n--- Simulating Audit Files for Test Mode ---")
             # Simulate Stage 1 Audit Files
             df_non_proc_candidates = df[df['category'] != PROCEDURAL_CATEGORY_NAME]
             _simulate_audit_files(df_non_proc_candidates, output_base_path, AUDIT_FRACTION)

             # Simulate Stage 2: Edit the review file
             simulation_ok = _simulate_review_file_edit(procedural_review_file_path)
             if not simulation_ok:
                 logging.error("Failed to simulate procedural review file edit. Exiting.")
                 sys.exit(1) # Exit if simulation fails, as merge step would fail

             time.sleep(0.5) # Brief pause for file system

        # --- Check required Stage 1 audit files exist ---
        if not audit_file_a.is_file() or not audit_file_b.is_file():
             logging.error(f"Stage 1 Audit files ({audit_file_a.name}, {audit_file_b.name}) not found. Cannot finalize."); sys.exit(1)

        # --- Step 10a: Check Stage 1 Kappa ---
        print("\n--- Step 10a: Checking Stage 1 Audit (Non-Procedural Kappa & Sentinel) ---")
        stage1_results = check_stage1_kappa(output_base_path, KAPPA_MIN) # Pass KAPPA_MIN from config

        # --- Update Stats with Stage 1 Results ---
        # (Stats update logic unchanged)
        run_stats = {} # Initialize empty stats dict
        if stats_file_path.is_file(): # Load previous stats if they exist
             try:
                  with open(stats_file_path, 'r', encoding='utf-8') as f: run_stats = json.load(f)
             except Exception as e: logging.error(f"Could not load previous stats file {stats_file_path}: {e}", exc_info=True)
        run_stats['rater_A_sentinel_accuracy'] = stage1_results.get('rater_A_acc'); run_stats['rater_B_sentinel_accuracy'] = stage1_results.get('rater_B_acc')
        kappa_scores_serializable = {k: (float(v) if pd.notna(v) else None) for k,v in stage1_results.get('kappa_scores', {}).items()}
        run_stats['kappa_scores_stage1'] = kappa_scores_serializable
        run_stats['min_kappa_stage1'] = float(stage1_results.get('min_kappa')) if pd.notna(stage1_results.get('min_kappa')) else None
        run_stats['stage1_audit_passed'] = stage1_results.get('passed', False)
        run_stats['finalization_status'] = 'Stage 1 Failed' if not stage1_results.get('passed') else 'Stage 1 Passed' # Update status
        try:
             with open(stats_file_path, 'w', encoding='utf-8') as f: json.dump(run_stats, f, indent=2, default=str)
             logging.info(f"Updated statistics with Stage 1 results saved to: {stats_file_path}")
        except Exception as e: logging.error(f"Could not update stats file {stats_file_path}: {e}", exc_info=True)

        # Exit if Stage 1 failed
        if not stage1_results.get("passed"):
             print("\nFinalization failed due to low Kappa or rater accuracy in Stage 1.")
             sys.exit(1)

        # --- Stage 1 Passed - Proceed to Stage 2 Processing ---
        print("\n--- Step 10b: Merging Procedural Steps from Review File & Preparing Final Dataset ---")

        # Initialize df_merged with the candidate data
        df_merged = df.copy()
        procedural_merge_success = True # Flag to track success

        # ======================================================================
        # START: UNIFIED Logic for Procedural Step Handling (Reads Review File)
        # ======================================================================
        logging.info(f"Checking for procedural review file: {procedural_review_file_path.name}...")

        if not procedural_review_file_path.is_file():
             # This file should *always* exist after generate step (even if empty)
             logging.error(f"Procedural review file not found: {procedural_review_file_path}")
             logging.error("This file should have been created during the 'generate' step.")
             if not is_test_mode:
                 logging.error("Please ensure it exists and has been edited with corrections.")
             else:
                 logging.error("Generate step might have failed to create it, or it was deleted.")
             run_stats['finalization_status'] = 'Stage 2 Failed (Review File Missing)';
             procedural_merge_success = False # Mark as failed
        else:
            # --- Merge Corrected Procedural Steps from _procedural_review.csv ---
            logging.info(f"Attempting to merge corrected steps from {procedural_review_file_path.name}...")
            try:
                # Read the review file (edited by human in --finalize, simulated edit in --finalize_test)
                # Use keep_default_na=False to preserve empty strings
                df_review_edits = pd.read_csv(procedural_review_file_path, keep_default_na=False)

                # Check for required columns IN THE REVIEW FILE
                required_cols = ['question_id', 'corrected_steps', 'procedural_comments']
                missing_req_cols = [col for col in required_cols if col not in df_review_edits.columns]
                if missing_req_cols:
                     raise ValueError(f"Review file ('{procedural_review_file_path.name}') missing required columns: {missing_req_cols}.")

                # Function to safely parse list string from corrected_steps column
                def safe_literal_eval_steps(val):
                    # Handle empty strings or potential non-string types safely
                    if not isinstance(val, str) or not val.strip():
                        # If it's not a non-empty string, treat as empty list
                        if not pd.isna(val) and val != "": # Log only if it wasn't originally empty
                             logging.debug(f"Empty or non-string value ('{val}') found in corrected_steps, interpreting as empty list.")
                        return []
                    val_stripped = val.strip()
                    if val_stripped == '[]': return []
                    try:
                        # Check if it looks like a list representation
                        if val_stripped.startswith('[') and val_stripped.endswith(']'):
                            parsed = ast.literal_eval(val_stripped) # val is already confirmed str here
                            if isinstance(parsed, list) and all(isinstance(i, str) for i in parsed):
                                return [s.strip() for s in parsed if s.strip()] # Clean steps
                            else:
                                logging.warning(f"Invalid list structure in corrected_steps column value: {val[:100]}... Interpreting as empty list.")
                                return []
                        else:
                            # If it's not an empty string and doesn't look like a list, log a warning
                            logging.warning(f"Corrected_steps value is not a valid list representation: {val[:100]}... Interpreting as empty list.")
                            return []
                    except Exception as e:
                        logging.warning(f"Failed to parse corrected_steps column value: {val[:100]}... Error: {e}. Interpreting as empty list.")
                        return []

                # Apply parsing to the corrected_steps column from the review file
                df_review_edits['corrected_steps_list'] = df_review_edits['corrected_steps'].apply(safe_literal_eval_steps)
                # Ensure procedural_comments is string (already handled by keep_default_na=False + read_csv)
                df_review_edits['procedural_comments'] = df_review_edits['procedural_comments'].astype(str)

                # Drop rows from review file where question_id might be missing or empty
                df_review_edits_valid = df_review_edits.dropna(subset=['question_id'])
                df_review_edits_valid = df_review_edits_valid[df_review_edits_valid['question_id'].astype(str).str.strip() != '']

                # Ensure question_id is string for matching in both dataframes
                df_review_edits_valid['question_id'] = df_review_edits_valid['question_id'].astype(str)
                df_merged['question_id'] = df_merged['question_id'].astype(str)


                # Create maps from question_id to corrected data FROM THE REVIEW FILE
                correction_map = pd.Series(df_review_edits_valid.corrected_steps_list.values, index=df_review_edits_valid.question_id).to_dict()
                comments_map = pd.Series(df_review_edits_valid.procedural_comments.values, index=df_review_edits_valid.question_id).to_dict()

                # Apply the corrections to the candidate dataframe ('df_merged')
                # Create columns in df_merged if they don't exist and ensure type is object/str
                if 'corrected_steps' not in df_merged.columns: df_merged['corrected_steps'] = None
                df_merged['corrected_steps'] = df_merged['corrected_steps'].astype(object) # Ensure object type for lists
                if 'procedural_comments' not in df_merged.columns: df_merged['procedural_comments'] = ""
                df_merged['procedural_comments'] = df_merged['procedural_comments'].astype(str) # Ensure string type

                # Identify procedural rows in the main dataframe to apply updates
                update_mask = (df_merged['category'] == PROCEDURAL_CATEGORY_NAME) & (df_merged['question_id'].isin(correction_map))

                # Apply corrections using .map() - map requires the series index to be unique
                # If duplicate QIDs exist in the review file, mapping might be unpredictable. Check for this.
                if df_review_edits_valid.question_id.duplicated().any():
                     duplicates = df_review_edits_valid[df_review_edits_valid.question_id.duplicated()]['question_id'].unique()
                     logging.warning(f"Duplicate question_ids found in review file: {duplicates}. Mapping behavior for these might be unpredictable (last value usually wins).")
                # Create the series again just before mapping to be sure index is set correctly
                correction_series = pd.Series(df_review_edits_valid.corrected_steps_list.values, index=df_review_edits_valid.question_id)
                comments_series = pd.Series(df_review_edits_valid.procedural_comments.values, index=df_review_edits_valid.question_id)

                # Perform the map operation on the subset defined by update_mask
                mapped_corrections = df_merged.loc[update_mask, 'question_id'].map(correction_series)
                mapped_comments = df_merged.loc[update_mask, 'question_id'].map(comments_series).fillna("") # Fill NaN comments with ""

                # Assign the mapped values back to the original DataFrame using loc
                df_merged.loc[update_mask, 'corrected_steps'] = mapped_corrections
                df_merged.loc[update_mask, 'procedural_comments'] = mapped_comments

                # Final check on column types after merge
                df_merged['corrected_steps'] = df_merged['corrected_steps'].astype(object)
                df_merged['procedural_comments'] = df_merged['procedural_comments'].astype(str)

                updated_count = update_mask.sum()
                total_review_rows = len(df_review_edits_valid) # Use valid rows count
                logging.info(f"Successfully merged corrected steps and comments for {updated_count} procedural rows from '{procedural_review_file_path.name}'.")
                if updated_count < total_review_rows:
                    # Find QIDs in review but not in merge target
                    missing_qids = set(df_review_edits_valid['question_id']) - set(df_merged.loc[update_mask, 'question_id'])
                    logging.warning(f"{len(missing_qids)} rows from the review file did not match procedural QIDs in the candidate set. Examples: {list(missing_qids)[:5]}")

                procedural_merge_success = True

            except FileNotFoundError:
                 # This case should ideally be caught earlier, but handle again just in case
                 logging.error(f"Error merging: Procedural review file not found at {procedural_review_file_path}")
                 run_stats['finalization_status'] = 'Stage 2 Failed (Review File Missing)';
                 procedural_merge_success = False
            except Exception as e:
                logging.error(f"Error merging corrected procedural steps from '{procedural_review_file_path.name}': {e}", exc_info=True)
                run_stats['finalization_status'] = 'Stage 2 Failed (Review Merge Error)';
                procedural_merge_success = False
        # ======================================================================
        # END: UNIFIED Logic for Procedural Step Handling
        # ======================================================================

        # Exit if procedural merge/handling failed
        if not procedural_merge_success:
             try: # Save final status before exiting
                 with open(stats_file_path, 'w', encoding='utf-8') as f: json.dump(run_stats, f, indent=2, default=str)
             except Exception as dump_e: logging.error(f"Could not save final status: {dump_e}")
             sys.exit(1)

        # --- Save Final Gold Dataset ---
        print("\nSaving final gold dataset...")
        try:
            # Use FINAL_SCHEMA global constant - Select only columns that EXIST in df_merged
            final_export_cols = [col for col in FINAL_SCHEMA if col in df_merged.columns]
            missing_final_cols = [col for col in FINAL_SCHEMA if col not in df_merged.columns]
            if missing_final_cols:
                logging.warning(f"Final DF missing some expected columns for gold file: {missing_final_cols}. Saving available columns.")

            df_to_save = df_merged[final_export_cols].copy() # Select only existing columns from the desired schema and make a copy

            # --- Convert list-like columns to string representation for CSV ---
            # Ensure the conversion happens safely, handling None or non-list types
            def list_to_str_safe(x):
                if isinstance(x, list):
                    return str(x)
                elif x is None:
                    return '[]' # Represent None as empty list string
                else:
                    # Log unexpected type, return as basic string
                    logging.debug(f"Unexpected type in list column for CSV export: {type(x)}, value: {x}. Converting using str().")
                    return str(x)

            if 'corrected_steps' in df_to_save.columns:
                df_to_save['corrected_steps'] = df_to_save['corrected_steps'].apply(list_to_str_safe)
            if 'parsed_steps' in df_to_save.columns:
                 df_to_save['parsed_steps'] = df_to_save['parsed_steps'].apply(list_to_str_safe)

            df_to_save.to_csv(gold_file_path, index=False, quoting=csv.QUOTE_ALL, encoding='utf-8')
            print(f"🎉 Dataset complete! Final output saved to: {gold_file_path}");
            run_stats['finalization_status'] = 'Completed' # Update status
        except Exception as e:
            logging.error(f"Failed to save final gold dataset: {e}", exc_info=True);
            run_stats['finalization_status'] = 'Stage 2 Failed (Save Error)'
            try:
                with open(stats_file_path, 'w', encoding='utf-8') as f: json.dump(run_stats, f, indent=2, default=str)
            except Exception as dump_e: logging.error(f"Could not save final status: {dump_e}")
            sys.exit(1)

        # Save final stats after successful save
        try:
            with open(stats_file_path, 'w', encoding='utf-8') as f: json.dump(run_stats, f, indent=2, default=str)
            logging.info(f"Final statistics saved to: {stats_file_path}")
        except Exception as e: logging.error(f"Failed to save final run statistics: {e}", exc_info=True)

        print("\n========================================"); print(f"✅ {mode.upper()} Completed Successfully! Gold dataset: {gold_file_path}"); print("========================================")


    else:
        # --- Generate Mode (Default) ---
        # (Generate Mode logic IS modified here)
        print(f"--- Running in GENERATION mode for input: {args.input_jsonl} ---")
        print(f"--- All output files will be saved relative to: {output_base_path} ---")
        df = pd.DataFrame(); loaded_pages = []; loaded_doc_id = ""; loaded_language = ""
        # Initialize run_stats, ensure config is serializable
        serializable_config = {}
        for k, v in config.items():
            try: json.dumps({k: v}) # Test serializability
            except TypeError: serializable_config[k] = str(v) # Convert non-serializable to string
            else: serializable_config[k] = v
        run_stats = {'manual_id': output_base_path.name, 'config_used': serializable_config}

        try: # Wrap pipeline steps
            # Step 1: Load
            print("\n--- Step 1: Loading & Setup ---")
            loaded_pages, loaded_doc_id, loaded_language = load_manual(args.input_jsonl); run_stats['input_pages'] = len(loaded_pages); logging.info(f"Loaded {len(loaded_pages)} pages.")

            # Step 2: Get Raw Rows
            print("\n--- Step 2: Obtaining Raw Generated Rows ---"); loaded_raw_rows = []
            if raw_output_file_path.is_file():
                 logging.info(f"Raw file found: {raw_output_file_path}. Skipping generation..."); raw_text = raw_output_file_path.read_text(encoding='utf-8'); loaded_raw_rows = [l for l in raw_text.splitlines() if l.strip()]; logging.info(f"Loaded {len(loaded_raw_rows)} lines.")
            else:
                 logging.info(f"Raw file not found. Generating..."); full_prompt = build_prompt(loaded_pages); logging.info(f"Prompt length: {len(full_prompt)} chars."); generation_temp = 0.8; raw_generated_rows = over_generate(full_prompt, OVERGEN_FACTOR, gemini_model, generation_temp); logging.info(f"Target raw output file: {raw_output_file_path}")
                 if raw_generated_rows: raw_output_content = "\n".join(raw_generated_rows); raw_output_file_path.write_text(raw_output_content, encoding='utf-8'); logging.info(f"Saved {len(raw_generated_rows)} rows to {raw_output_file_path}"); loaded_raw_rows = raw_generated_rows
                 else: raise ValueError("Generation failed to produce any rows.")
            if not loaded_raw_rows: raise ValueError("No raw rows obtained (loaded or generated)."); run_stats['raw_rows_generated'] = 0

            run_stats['raw_rows_obtained'] = len(loaded_raw_rows) # Use 'obtained' as it could be loaded or generated

            # Step 3: Parse
            print("\n--- Step 3: Parsing Raw Rows ---")
            df = parse_rows(loaded_raw_rows, SCHEMA, loaded_doc_id, loaded_language)
            run_stats['rows_parsed'] = len(df); run_stats['parse_failures'] = run_stats.get('raw_rows_obtained',0) - run_stats['rows_parsed']
            if df.empty: raise ValueError("Parsing failed, no valid rows produced.")
            print(f"\n--- Step 3 Verification ---"); print(f"Parsed {df.shape[0]} rows.");
            if 'category' in df.columns: print("Category Distribution (Initial):\n", df['category'].value_counts().to_string());
            else: print("Category column missing after parse.");
            print("\n--- Step 3 Complete (Parsing) ---")

            # --- NEW Step 3a: Filter Invalid Metadata ---
            rows_before_meta_filter = len(df)
            df = filter_invalid_metadata(df)
            run_stats['rows_after_meta_filter'] = len(df)
            run_stats['rows_removed_invalid_metadata'] = rows_before_meta_filter - len(df)
            if df.empty: raise ValueError("DataFrame empty after filtering invalid metadata.")
            print(f" Shape after Metadata Filter: {df.shape}")
            if 'category' in df.columns: print("Category Distribution (After Meta Filter):\n", df['category'].value_counts().to_string())


            # Step 3b: Add Parsed Steps via LLM
            df = add_parsed_steps_llm(df, STEP_PARSING_MODEL) # Use config model
            proc_found = int((df['category'] == PROCEDURAL_CATEGORY_NAME).sum()); proc_parsed = int(df['parsed_steps'].apply(lambda x: isinstance(x, list) and len(x) > 0).sum()) if 'parsed_steps' in df.columns else 0
            run_stats['procedural_rows_found'] = proc_found; run_stats['procedural_rows_llm_parsed'] = proc_parsed
            print(f"\n--- Step 3b Verification ---"); print(f"LLM parsed steps for {proc_parsed}/{proc_found} procedural rows.")
            # (Already prints step complete message inside function)

            # Step 4: Deduplicate
            # print("\n--- Step 4: Deduplicating Questions ---"); # Moved print inside function
            rows_before = len(df)
            if not df.empty: df = deduplicate(df, embedding_model, DUP_THRESHOLD, EMBED_MODEL)
            run_stats['rows_after_dedupe'] = len(df); run_stats['rows_removed_dedupe'] = rows_before - len(df)
            if df.empty: logging.warning("DataFrame empty after Deduplication.")
            # (Already prints step complete message inside function)

            # Step 5: Annotate Page Check
            # print("\n--- Step 5: Annotating Snippet Grounding ---") # Moved print inside function
            if not df.empty: df = page_check_annotate(df, loaded_pages)
            if not df.empty and 'passed_strict_check' in df.columns:
                passed_strict_count = int(df['passed_strict_check'].sum()); failed_strict_count = len(df) - passed_strict_count
                run_stats['stats_after_annotate'] = {'total_rows': len(df), 'passed_strict': passed_strict_count, 'failed_strict': failed_strict_count }
                try: pass_rate_by_cat = df.groupby('category')['passed_strict_check'].mean().round(3).to_dict(); run_stats['strict_check_rate_by_category'] = {k:v for k,v in pass_rate_by_cat.items() if pd.notna(v)}
                except Exception as e: logging.warning(f"Could not calc per-cat strict pass rate: {e}"); run_stats['strict_check_rate_by_category'] = {}
            else: run_stats['stats_after_annotate'] = {'total_rows': len(df), 'passed_strict': 0, 'failed_strict': len(df) }; run_stats['strict_check_rate_by_category'] = {}
            if df.empty: logging.warning("DataFrame empty after Page Check annotation.")
            # (Already prints step complete message inside function)

            # Step 6: RAGAS Filter
            # print("\n--- Step 6: Filtering with Ragas ---"); # Moved print inside function
            rows_before = len(df)
            if not df.empty: df = ragas_filter(df, loaded_pages, RAGAS_THRESHOLD)
            run_stats['rows_after_ragas'] = len(df); run_stats['rows_removed_ragas'] = rows_before - len(df)
            if df.empty: logging.warning("DataFrame empty after Ragas filtering.")
            # (Already prints step complete message inside function)

            # Step 7: Judge Filter
            # print("\n--- Step 7: Filtering Answerable with LLM Judge ---"); # Moved print inside function
            rows_before = len(df)
            if not df.empty: df = judge_filter(df, loaded_pages, JUDGE_MODEL, JUDGE_THRESHOLD)
            run_stats['rows_after_judge'] = len(df); run_stats['rows_removed_judge'] = rows_before - len(df)
            if df.empty: logging.warning("DataFrame empty after LLM Judge filtering.")
            # (Already prints step complete message inside function)

            # Step 7b: Verify Unanswerable
            # print("\n--- Step 7b: Verifying Unanswerable Questions ---"); # Moved print inside function
            rows_before = len(df); unans_before = (df['category'] == UNANSWERABLE_CATEGORY_NAME).sum() if 'category' in df.columns else 0; run_stats['unanswerable_initially'] = int(unans_before)
            if not df.empty:
                df_ans = df[df['category'] != UNANSWERABLE_CATEGORY_NAME].copy(); df_unans = df[df['category'] == UNANSWERABLE_CATEGORY_NAME].copy()
                if not df_unans.empty: full_ctx = get_full_manual_text(loaded_pages); verified_idx = verify_unanswerable(df_unans, full_ctx, JUDGE_MODEL); df_veri_unans = df_unans.loc[verified_idx]; df = pd.concat([df_ans, df_veri_unans]).sort_index().reset_index(drop=True)
                else: logging.info("No Unanswerable rows to verify.")
            unans_after = (df['category'] == UNANSWERABLE_CATEGORY_NAME).sum() if 'category' in df.columns else 0; run_stats['unanswerable_verified'] = int(unans_after); run_stats['rows_after_unanswerable_verify'] = len(df);
            print(f" Shape after Unanswerable check: {df.shape}");
            # (Already prints step complete message inside function)

            # Step 8: Quota Select
            # print("\n--- Step 8: Selecting Final Rows per Category Quota ---") # Moved print inside function
            if df.empty: raise ValueError("No rows left for quota selection.")
            df, quota_info = quota_select(df, CATEGORY_TARGETS, FINAL_DATASET_SIZE) # Use config targets/size
            run_stats['quota_selection_details'] = quota_info; run_stats['quota_met_fully'] = quota_info.get('_summary', {}).get('all_quotas_met', False); run_stats['final_dataset_size_generated'] = len(df)
            if len(df) == 0: raise ValueError("Quota selection resulted in zero rows. Cannot proceed.") # Stop if quota selection fails completely
            # (Already prints step complete message inside function)

            # Step 9: Save Candidates & Export Audit Slice
            # print("\n--- Step 9: Saving Candidates & Exporting Audit Files ---") # Moved print inside function
            if df.empty or len(df) < 1: raise ValueError(f"Quota selection failed or resulted in empty DataFrame. Cannot export.") # Check size > 0
            # Call export_audit_slice (which now includes saving candidates)
            export_audit_slice(df, output_base_path, AUDIT_FRACTION);
            # Correct calculation of audit sample size for stats
            non_proc_count_final = len(df[df['category'] != PROCEDURAL_CATEGORY_NAME]) # Calculate non-proc count on final DF
            k_stage1_base = max(3, int(round(non_proc_count_final * AUDIT_FRACTION + 0.5)))
            k_stage1_capped = min(k_stage1_base, non_proc_count_final)
            # Add sentinel if real rows > 0 OR if no real rows exist at all (to ensure at least sentinel is audited)
            k_stage1_final = k_stage1_capped + (NUM_SENTINELS if k_stage1_capped > 0 or non_proc_count_final == 0 else 0)
            run_stats['audit_sample_size_stage1'] = k_stage1_final
            num_proc_exported = len(df[df['category'] == PROCEDURAL_CATEGORY_NAME]); run_stats['audit_sample_size_stage2'] = num_proc_exported
            # (Already prints step complete message inside function)

            # --- Save Stats for Generate Mode ---
            run_stats['pipeline_status'] = 'Generated'
            run_stats['rater_A_sentinel_accuracy'] = None; run_stats['rater_B_sentinel_accuracy'] = None
            run_stats['kappa_scores_stage1'] = None; run_stats['min_kappa_stage1'] = None; run_stats['finalization_passed'] = None
            try:
                # Ensure all values are JSON serializable before saving
                def make_serializable(obj):
                    if isinstance(obj, (int, float, str, bool, type(None))): return obj
                    if isinstance(obj, (list, tuple)): return [make_serializable(item) for item in obj]
                    if isinstance(obj, dict): return {str(k): make_serializable(v) for k, v in obj.items()}
                    # Handle pandas/numpy types explicitly if needed
                    if hasattr(obj, 'item'): return obj.item() # For numpy scalar types
                    if isinstance(obj, Path): return str(obj) # Convert Path objects
                    return str(obj) # Default conversion for other types

                serializable_stats = make_serializable(run_stats)
                with open(stats_file_path, 'w', encoding='utf-8') as f: json.dump(serializable_stats, f, indent=2)
                logging.info(f"Run statistics saved to: {stats_file_path}")
            except Exception as e: logging.error(f"Failed to save run statistics: {e}", exc_info=True)

            # --- Final Instructions (Unchanged) ---
            print("\n*** GENERATION COMPLETE ***")
            print("Candidate data and run stats saved.")
            print(f"Audit files ({audit_file_a.name}, {audit_file_b.name}) exported for Stage 1 review.")
            print(f"Procedural review file ({procedural_review_file_path.name}) exported for Stage 2 expert correction.")
            print("\nACTION REQUIRED:")
            print("1. Perform Stage 1 audit using 'rater_instructions.md' and save results.")
            print(f"2. Perform Stage 2 procedural correction by directly editing the 'corrected_steps'")
            print(f"   and 'procedural_comments' columns within the '{procedural_review_file_path.name}' file.")
            print(f"   Ensure you save your changes to '{procedural_review_file_path.name}'.")
            print("3. Once BOTH stages are complete and files saved, run finalize mode:")
            # Construct the finalize command carefully, quoting paths if they might contain spaces
            script_path_str = str(Path(sys.argv[0]).resolve())
            input_jsonl_str = str(input_file_path.resolve())
            finalize_cmd = f"python \"{script_path_str}\" --input \"{input_jsonl_str}\" --finalize"
            print(f"\n   {finalize_cmd}\n")

        except Exception as e:
             logging.error(f"An error occurred during the generation pipeline: {e}", exc_info=True)
             # Try to save any stats collected so far before exiting
             try:
                 if 'run_stats' in locals() and run_stats:
                      run_stats['pipeline_status'] = 'Failed'; run_stats['error_message'] = str(e)
                      # Make serializable before saving on error too
                      serializable_stats = {}
                      for k, v in run_stats.items():
                          try: json.dumps({k: v})
                          except TypeError: serializable_stats[k] = str(v)
                          else: serializable_stats[k] = v
                      with open(stats_file_path, 'w', encoding='utf-8') as f: json.dump(serializable_stats, f, indent=2)
                      logging.info(f"Partial run statistics saved to {stats_file_path} due to error.")
             except Exception as dump_e: logging.error(f"Could not save partial stats on error: {dump_e}")
             sys.exit(1)

    # --- End of Script ---
    if not args.finalize and not args.finalize_test: print("\nScript finished generate mode.")
    # Finalize modes print their own success messages or exits on error