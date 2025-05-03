import pymupdf4llm
import sys # To handle command-line arguments
import yaml
from dotenv import load_dotenv
import logging
import json
import re
from pathlib import Path # Using pathlib for easier path manipulation

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# --- Load configuration ---
try:
    # Use Path object for config path
    config_path = Path('config/settings.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logging.info("Configuration loaded successfully.")
except FileNotFoundError:
    logging.error(f"Error: {config_path} not found.")
    sys.exit(1) # Use sys.exit for clearer exit
except Exception as e:
    logging.error(f"Error loading configuration: {e}")
    sys.exit(1)

# --- Get paths from config ---
# Use Path objects for directories
PDF_DIR = Path(config['paths']['pdf_directory'])
MARKDOWN_PAGE_OUTPUT_DIR = Path(config['paths']['processed_data_dir']) / 'markdown_pages'
# Create output directory if it doesn't exist
MARKDOWN_PAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_language_from_filename(filename: str) -> str:
    """Extracts language (DE or EN) from filename based on common patterns."""
    if re.search(r'_DE_', filename, re.IGNORECASE):
        return "de"
    elif re.search(r'_EN_', filename, re.IGNORECASE):
        return "en"
    else:
        # Use Path(filename).name to get just the filename if a full path was passed
        logging.warning(f"Could not determine language for {Path(filename).name}, defaulting to 'en'.")
        return "en"

# --- Modified function to process ONE PDF ---
def process_single_pdf(pdf_path: Path, output_dir: Path):
    """
    Processes a single PDF, converting each page to Markdown and saving
    structured data to a specific JSONL file named after the PDF.
    """
    if not pdf_path.is_file():
        logging.error(f"Input PDF not found: {pdf_path}")
        return False # Indicate failure

    filename = pdf_path.name
    language = get_language_from_filename(filename)
    output_filename = output_dir / f"{pdf_path.stem}_pages.jsonl" # e.g., manual_A_pages.jsonl
    total_pages_saved = 0
    first_dict_keys_logged = False # Flag per file now

    logging.info(f"Processing: {filename} (Language: {language})")

    try:
        # Expecting a list of dictionaries from page_chunks=True
        markdown_pages_list = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True) # Pass path as string
        num_md_pages = len(markdown_pages_list)
        logging.info(f"Converted {filename} to Markdown, received {num_md_pages} page chunks (dictionaries).")

        with open(output_filename, 'w', encoding='utf-8') as outfile:
            for page_num_zero_based, page_dict in enumerate(markdown_pages_list):
                if isinstance(page_dict, dict):
                    if not first_dict_keys_logged:
                        logging.info(f"Keys in first page dictionary: {list(page_dict.keys())}")
                        first_dict_keys_logged = True

                    page_md = page_dict.get('text', '') # Use the correct key 'text'

                    if isinstance(page_md, str) and page_md.strip():
                        page_data = {
                            "doc_id": filename, # Store the original filename
                            "language": language,
                            "page_num": page_num_zero_based + 1,
                            "markdown_content": page_md.strip()
                        }
                        outfile.write(json.dumps(page_data, ensure_ascii=False) + '\n')
                        total_pages_saved += 1
                    elif not isinstance(page_md, str):
                         logging.warning(f"Value for key 'text' on page {page_num_zero_based+1} is not a string: type={type(page_md)}")

                else:
                     logging.warning(f"Item at index {page_num_zero_based} for {filename} is not a dictionary: type={type(page_dict)}")

        logging.info(f"Successfully processed {num_md_pages} pages.")
        logging.info(f"Total pages with content saved: {total_pages_saved}.")
        logging.info(f"Structured page data saved to: {output_filename}")
        return True # Indicate success

    except Exception as e:
        logging.error(f"Error processing {filename}: {e}", exc_info=True)
        return False # Indicate failure


# --- Main execution block modified ---
if __name__ == "__main__":
    # Check if a command-line argument (PDF filename) is provided
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <pdf_filename_in_{PDF_DIR}>")
        print(f"Example: python {sys.argv[0]} my_manual.pdf")
        sys.exit(1)

    pdf_filename_arg = sys.argv[1]
    pdf_full_path = PDF_DIR / pdf_filename_arg # Construct full path using pathlib

    logging.info(f"Starting PDF processing for single file: {pdf_full_path}")

    # Process the specified PDF
    success = process_single_pdf(pdf_full_path, MARKDOWN_PAGE_OUTPUT_DIR)

    if success:
        logging.info("PDF processing finished successfully.")
        # Optional: Print first few lines of the created file
        output_jsonl = MARKDOWN_PAGE_OUTPUT_DIR / f"{pdf_full_path.stem}_pages.jsonl"
        try:
            with open(output_jsonl, 'r', encoding='utf-8') as f:
                print(f"\n--- First 3 lines of output file ({output_jsonl.name}) ---")
                for i, line in enumerate(f):
                    if i >= 3: break
                    print(line.strip()[:200] + "...")
        except FileNotFoundError:
             logging.warning(f"Could not read output file {output_jsonl} for verification.")
    else:
        logging.error("PDF processing failed.")
        sys.exit(1) # Exit with error status if processing failed
