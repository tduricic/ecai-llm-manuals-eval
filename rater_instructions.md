# QA Dataset Audit Instructions

## 1. Goal

Thank you for helping audit this question-answering dataset generated from technical manuals. The goal of this audit is to verify the quality and consistency of the generated questions and answers before finalizing the dataset. Your careful judgment is crucial!

## 2. Task Overview

You will be provided with two CSV files:
* `[manual_name]_audit_A.csv`
* `[manual_name]_audit_B.csv`

(You might be assigned Rater A or Rater B, or asked to do both sequentially).

Your task is to:
1.  Open your assigned CSV file(s).
2.  For **each row**, review the provided information (`question_id`, `question_text`, `gt_answer_snippet`, `gt_page_number`, `category`, `persona`).
3.  You will need to **compare the `gt_answer_snippet` against the content of the original manual on the specified `gt_page_number`**. (See Section 3 below).
4.  Fill in the **last five columns** (`answer_correct?`, `grounded?`, `question_clear?`, `category_correct?`, `persona_tone_ok?`) with **"Yes"** or **"No"**. Case doesn't matter (yes/no, Yes/No are all fine). Please only use these two words.
5.  **Save** the CSV file(s) once you have rated all rows.
6.  Notify the coordinator (or press Enter in the script if you are running it) once **both** files are saved.

## 3. Accessing Source Context (Page Content)

To perform the `grounded?` check (and sometimes to verify `answer_correct?`), you need to see the original text from the manual page cited in the `gt_page_number` column.

* You should have access to the source `[manual_name]_pages.jsonl` file.
* This file contains the manual's content, page by page, in JSON format. Each line looks like:
    `{"doc_id": "...", "language": "...", "page_num": N, "markdown_content": "Text of page N..."}`
* **To find the context for a row:**
    1.  Note the `gt_page_number` (e.g., 40).
    2.  Open the `.jsonl` file in a text editor that can handle large files.
    3.  Search for the line containing `"page_num": 40` (using the number from the CSV).
    4.  The text within the `"markdown_content": "..."` field on that line is the source context you need.
    5.  You can use search (Ctrl+F) within this `markdown_content` to find the `gt_answer_snippet`.

*(Note: A helper script or tool might be provided later to make viewing page content easier if needed).*

## 4. Detailed Guidelines for Each Check

Please apply these definitions consistently:

**a) `answer_correct?` (Yes/No)**
* **Definition:** Does the `gt_answer_snippet`, based *only* on the information it contains, accurately and fully answer the core question asked in `question_text`?
* **Rate Yes if:** The snippet directly and correctly answers the question. Minor awkwardness in the snippet's phrasing (if extracted verbatim) is okay if the information is correct.
    * *Example Yes:* Q: "What is the max load?" A: "Max. load 8.0 kg"
    * *Example Yes:* Q: "How to clean filters?" A: "1. Open door. 2. Remove filter." (Assuming these are the correct first steps)
* **Rate No if:** The snippet is factually incorrect, answers a *different* question, only *partially* answers the question (leaving out critical info implied by the question), or is completely irrelevant.
    * *Example No:* Q: "What is max load?" A: "Clean the fluff filters." (Irrelevant)
    * *Example No:* Q: "What voltage is needed?" A: "Voltage See data plate" (Doesn't provide the value itself - partial answer).
    * *Example No:* Q: "Max load for cottons?" A: "8.0 kg" (If the manual actually says 9kg for cottons - factually incorrect snippet).

**b) `grounded?` (Yes/No)**
* **Definition:** Is the *entire* text in `gt_answer_snippet` present *exactly* (verbatim, case-sensitive, punctuation-sensitive) as a *contiguous block* of text within the `markdown_content` of the page specified by `gt_page_number`?
* **How to Check:** Use the method in Section 3 to find the page content. Use your text editor's search function (Ctrl+F) to find the *exact* text from the `gt_answer_snippet`.
* **Rate Yes if:** You find an exact, character-for-character match for the *entire* snippet on that page. Internal newlines (`\n`) in the snippet must match newlines in the source.
    * *Example Yes:* Snippet `"Clean the filter"` is found exactly as `"Clean the filter"` on the cited page.
    * *Example Yes:* Snippet `"1. Step One\n2. Step Two"` is found exactly matching those two lines in the source.
* **Rate No if:** There is *any* difference - even a single changed/missing word, different punctuation, different case, paraphrasing, or if the snippet text is split non-contiguously in the source, or if the snippet cannot be found on the cited page *at all*.
    * *Example No:* Snippet is `"Clean filter"` but source says `"Clean the filter"`.
    * *Example No:* Snippet is `"Step 1. Do X"` but source says `"1. Do X"`.
    * *Example No:* Snippet is `"See page 10"` but `gt_page_number` is 9.

**c) `question_clear?` (Yes/No)**
* **Definition:** Is the `question_text` grammatically correct, unambiguous, and is its intent easily understandable?
* **Rate Yes if:** The question is well-formed and you immediately understand what information is being sought.
* **Rate No if:** The question is grammatically incorrect, contains significant typos that obscure meaning, is vague (e.g., "What about the setting?"), or could be reasonably interpreted in multiple ways.
    * *Example No:* "How adjust setting it?" (Grammar)
    * *Example No:* "Tell me about the main feature." (Vague)
    * *Example No:* "Does it work with it?" (Ambiguous pronoun)

**d) `category_correct?` (Yes/No)**
* **Definition:** Is the assigned `category` the single best fit for the `question_text` based on the category definitions below?
    * **Specification Lookup:** Asks for specific quantitative or qualitative properties, technical data, capacities, dimensions, ratings, error codes, model numbers, or standards mentioned.
    * **Tool/Material Identification:** Asks to identify required tools, parts, materials, consumables, software, or specific chemical agents needed or mentioned.
    * **Procedural Step Inquiry:** Asks how to perform a task involving multiple steps (e.g., "How do I install X?", "What are the steps to clean Y?").
    * **Location/Definition:** Asks *where* a physical component/control is located OR *what* a specific term, symbol, indicator, message, or menu option *means* or *does*.
    * **Conditional Logic/Causal Reasoning:** Asks about situations dependent on specific conditions ('What if...?', 'When should I...?'), prerequisites, or cause/effect ('Why did...?', 'What happens if...?').
    * **Safety Information Lookup:** Asks about potential hazards, safety precautions, warnings, PPE, risks, emergency procedures, or safe disposal.
    * **Unanswerable:** Asks a plausible question, but the answer cannot be found in the manual.
* **Rate Yes if:** The assigned category accurately reflects the question's primary intent.
* **Rate No if:** A different category would be a significantly better fit.
    * *Example No:* Q: "How do I clean the filter?" Category: `Location/Definition` (Should be `Procedural Step Inquiry`)
    * *Example No:* Q: "What does error E-05 mean?" Category: `Safety Information Lookup` (Should be `Specification Lookup` or maybe `Location/Definition`)

**e) `persona_tone_ok?` (Yes/No)**
* **Definition:** Does the `question_text`'s wording, complexity, and focus reasonably align with the assigned `persona` based on the style hints below?
    * **Novice User:** simple wording, asks basic “what/where” questions
    * **Technician:** uses precise technical jargon, asks detailed procedural/spec questions
    * **SafetyOfficer:** asks about/emphasises risks, warnings, and protective measures
* **Rate Yes if:** The style generally fits the persona. It doesn't have to be a perfect caricature.
* **Rate No if:** There is a clear mismatch (e.g., extremely technical jargon for a Novice, overly simplistic question for a Technician, a safety question not assigned to SafetyOfficer when it clearly fits).
    * *Example No:* Persona: `Novice User`, Question: "What is the nominal impedance curve under phased harmonic load?"
    * *Example No:* Persona: `Technician`, Question: "Where is the on button?" (Could be Yes if context implies difficulty finding it, but generally too simple).

## 5. Handling Uncertainty

If you are genuinely unsure how to rate a specific check after carefully considering the guidelines and the source text, please make your best judgment and, if possible, note the `question_id` separately for later discussion. If forced to choose Yes/No, slightly favor "No" if quality/correctness is uncertain.

## 6. Contact

If you have questions about these instructions, please contact [Your Name/Contact Info Here].

---

Thank you for your help in ensuring the quality of this dataset!