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

Manuals:

[NanoDrop One](https://tools.thermofisher.com/content/sfs/manuals/3091-NanoDrop-One-Help-UG-en.pdf)
[Omron Monitor](https://medaval.ie/docs/manuals/Omron-HEM-7200-E-Manual.pdf)
[Miele G5000e](https://media.miele.com/downloads/54/42/00_F61547C6F7151EDDB0FCF677A67E5442.pdf)
[DeWalt Saw](https://www.dewalt.com/GLOBALBOM/QU/DWE575/1/Instruction_Manual/EN/NA477368_DWE574_DWE575_DWE575SB_T1__NA.pdf)
[Bosch Oven](https://media3.bosch-home.com/Documents/9001011473_B.pdf)
[Dyson v12](https://www.dyson.co.uk/content/dam/dyson/maintenance/user-guides/en-vn/DYS_P211757_J0004_T01_Dyson_620D_TH_VN_ID_user_guide_W210XH297_EN-VN_Digital.pdf)
[Prusa 3d printer](https://www.prusa3d.com/downloads/manual/prusa3d_manual_mk4s_en_1_0.pdf)
[Makita drill](https://cdn.makitatools.com/apps/cms/doc/prod/XFD/e2dc6a61-63d3-4455-b4dc-940ffa84e99d_XFD12_IM_885510-943.pdf)
[Hilti Hammer](https://www.hilti.com/medias/sys_master/documents/h76/ha2/10131929923614/Operating-Instruction-TE-60-04-Operating-Instruction-PUB-5277961-000.pdf)
[]()

(High-level tasks remaining.)