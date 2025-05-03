# ECAI 2025 Paper: Evaluating LLMs for Technical Document Interaction

This repository contains the code, data, and manuscript for our ECAI 2024 submission evaluating the capabilities of Large Language Models (LLMs) on technical instruction manuals.

## Goal

The primary goal is to develop and apply a comprehensive evaluation framework to assess LLM performance across multiple critical dimensions when interacting with a corpus of technical documents. We focus on:

1.  **Structured Data Extraction:** Generating valid JSON outputs according to specified schemas.
2.  **Grounded Question Answering:** Providing accurate answers based on manual content, including correct source identification (document and page).
3.  **Procedural Task Guidance:** Generating reliable step-by-step instructions for tasks described in the manuals.

## Repository Structure

*   `/paper`: Contains the LaTeX source files for the ECAI paper.
*   `/data`: Holds the raw instruction manuals (PDFs - *add locally, do not commit if large/confidential*), processed data, and ground truth annotations.
    *   `/data/pdfs`: Raw PDF manuals (local only).
    *   `/data/processed`: Intermediate processed data (e.g., text chunks).
    *   `/data/ground_truth`: Manually created ground truth data (JSONs, Q&A pairs, Guidance sequences).
*   `/src`: Contains the Python source code for data processing, RAG pipeline, experiments, and evaluation.
*   `/notebooks`: Jupyter notebooks for exploration, analysis, and visualization (optional).
*   `/config`: Configuration files (e.g., model names, paths, parameters - API keys stored locally/securely).
*   `requirements.txt`: Python dependencies.

## Setup

(Instructions to be added later on how to set up the environment and run the code.)

## TODO

(High-level tasks remaining.)