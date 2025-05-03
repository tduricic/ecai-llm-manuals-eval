# Procedural QA Step Correction Instructions (Stage 2 Audit)

## 1. Goal

Thank you for reviewing the procedural questions for this dataset creation project. The primary goal of this specific task is to create the definitive **gold standard list of steps** for question-answer pairs categorized as "Procedural Step Inquiry".

You are acting as the expert adjudicator. You will review the original question, the answer snippet extracted by an LLM, the original source page content (from the manual), and an initial attempt by another LLM to parse the snippet into steps. Your careful review and correction are crucial for ensuring the accuracy of the procedural ground truth used in later evaluations.

## 2. Files & Task Overview

You will work with one CSV file per manual:
* `[manual_name]_procedural_review.csv`

This file contains only the rows identified as procedural questions that passed the automated filtering stages (typically 5 rows per manual, but could be fewer if some were filtered out earlier).

**Your task is to:**

1.  Open the `_procedural_review.csv` file.
2.  For **each row**, carefully review these columns:
    * `question_id`: Unique identifier.
    * `question_text`: The question asked.
    * `gt_answer_snippet`: The verbatim text extracted by the generation LLM as the answer. This might contain formatting artifacts (``, `...`, odd newlines) from the source.
    * `gt_page_number`: The page number in the original manual where the snippet was found.
    * `parsed_steps`: The list of steps automatically parsed from `gt_answer_snippet` by an LLM. **This may be inaccurate or fragmented.**
3.  **Access Source Context:** Use the `gt_page_number` to view the original `markdown_content` for that page from the corresponding `[manual_name]_pages.jsonl` file (see Section 3 below). You **must** refer to the source page content to understand the original formatting and context accurately.
4.  **Determine Correct Steps:** Based *primarily* on the `gt_answer_snippet` and informed by the source page `markdown_content`, identify the sequence of distinct, semantic actions or instructions that correctly answer the `question_text`.
5.  **Fill `corrected_steps` Column:** Enter the definitive list of steps into this column for **every row**. Format the output **exactly** as a Python list literal string, using single quotes for the strings inside the list. See detailed guidelines and examples below.
6.  **Fill `procedural_comments` Column:** Add comments **only if necessary** (see guidelines below).
7.  **Save As:** Save the completed file as `[manual_name]_procedural_corrected.csv`. Ensure it remains a valid CSV file.

## 3. Accessing Source Context (Page Content)

To perform this task accurately, you *must* refer to the original page content.

* You should have access to the source `[manual_name]_pages.jsonl` file for the manual you are reviewing.
* Each line in this file represents a page and looks like:
    `{"doc_id": "...", "language": "...", "page_num": N, "markdown_content": "Text of page N..."}`
* **To find the context for a row in your review CSV:**
    1.  Note the `gt_page_number` (e.g., 40).
    2.  Open the `.jsonl` file in a text editor capable of handling potentially large lines.
    3.  Search for the line containing `"page_num": 40` (using the number from the CSV).
    4.  The text within the `"markdown_content": "..."` field on that line is the source context.
    5.  Compare the `gt_answer_snippet` from the CSV to this source context to understand the original layout, surrounding text, and potential artifacts.

## 4. Detailed Guidelines for Filling Columns

**a) `corrected_steps` Column (Mandatory for all rows)**

* **Format:** MUST be a valid Python list literal string. Use single quotes inside for the step strings.
    * *Correct Examples:* `['Step 1.', 'Step 2.']`, `['Do this first.', 'Then this.']`, `[]`
    * *Incorrect Examples:* `Step 1, Step 2`, `["Step 1", "Step 2"]` (uses double quotes inside), `None`, *(empty cell)*
* **Content - Identifying Steps:**
    * A "step" should represent a distinct semantic action or instruction required to answer the `question_text`.
    * Focus on the core actions described in the `gt_answer_snippet`, using the source page context for clarification.
    * **Merge Fragments:** Combine text clearly belonging to the same action, especially if split by ellipses (`...`) or awkward line breaks/hyphenation in the `gt_answer_snippet`.
    * **Ignore Markers:** Do not include non-standard bullets (``, `–`, etc.) or standard list numbering (`1.`, `a)`) in the final step strings *unless* they are essential to the meaning of the step itself (rare).
    * **Exclude Non-Action Text:** Omit purely descriptive sentences, introductory phrases, or result descriptions unless they are critically necessary to understand *how* to perform an action.
    * **Preserve Wording:** Maintain the original phrasing from the `gt_answer_snippet` / source text within each step where possible and appropriate. Minor rephrasing for clarity or merging fragments is acceptable.
* **Specific Cases:**
    * **If `parsed_steps` (auto-generated) is Correct:** Review it against the snippet/source. If it accurately represents the steps according to the guidelines above, **copy its exact string representation** (e.g., `['Step A', 'Step B']`) into the `corrected_steps` column.
    * **If `parsed_steps` needs Correction:** Manually create the correct list string (following the formatting rules) and enter it into `corrected_steps`.
    * **If NO Valid Steps Apply:** If your review determines the `gt_answer_snippet` does not actually contain procedural steps relevant to the question (e.g., the question was miscategorized, the snippet is wrong/irrelevant, or describes a state, not actions), enter the literal string for an empty list: **`[]`**

**b) `procedural_comments` Column (Optional, except when entering `[]`)**

* **If `corrected_steps` contains a valid list (copied or corrected):** You only need to add comments if there was significant ambiguity, difficulty in deciding step boundaries, or other issues worth noting for future reference. **Leaving this blank implies you reviewed and approved/corrected the steps.**
* **If `corrected_steps` is `[]`:** You **must** add a comment explaining *why* no valid steps were extracted (e.g., "Original snippet is not procedural", "Question category seems incorrect", "Answer snippet irrelevant to question", "Source text too ambiguous to segment").

## 5. Examples

**Example A (Merging Ellipses/Markers):**
* `gt_answer_snippet`: `" Remove fluff . . .\n. . . from surfaces and de-\nflector."`
* `parsed_steps`: `['Remove fluff . . .', '. . . from surfaces and de- flector.']` (Example initial parse)
* `corrected_steps` (Your input): `"['Remove fluff from surfaces and deflector.']"`
* `procedural_comments`: `"Merged fragments based on ellipsis and hyphen."`

**Example B (Auto-parse Correct):**
* `gt_answer_snippet`: `"1. Attach tool.\n2. Unscrew."`
* `parsed_steps`: `['Attach tool.', 'Unscrew.']`
* `corrected_steps`: `"['Attach tool.', 'Unscrew.']"`
* `procedural_comments`: *`(Leave Blank)`*

**Example C (No Valid Steps):**
* `gt_answer_snippet`: `"Refer to diagram 5 for details."`
* `parsed_steps`: `['Refer to diagram 5 for details.']`
* `corrected_steps`: `"[]"`
* `procedural_comments`: `"Snippet is not a procedure, just a reference."`

## 6. Saving

When finished, save the file as `[manual_name]_procedural_corrected.csv` in the same directory where the review file was found. Ensure it is saved as a valid CSV file.

## 7. Contact

If you have questions about these instructions or encounter difficult cases, please contact **[Your Name/Contact Info Here]**.

---
Thank you! Your careful review is essential for creating high-quality procedural ground truth.