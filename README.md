# 🏦 Transformer-Based Summarization of SEC 10-K Filings

Fine-tuning **BART** to generate concise summaries of SEC 10-K annual filings using a weakly supervised NLP pipeline built on the MD&A section of real financial documents.

---

## 📌 Overview

This project was completed as part of **CS 583 – Deep Learning** at Stevens Institute of Technology (Fall 2025).

SEC 10-K filings often exceed 100 pages of dense financial, operational, and risk-related content. This project builds an end-to-end summarization pipeline using `facebook/bart-large-cnn`, fine-tuned on the MD&A (Management's Discussion and Analysis) section of 191 real filings from the SEC EDGAR dataset. Since human-written summaries are unavailable, pseudo-summaries are generated for weak supervision.

---

## 🗂️ Project Structure

```
sec-10k-summarization/
│
├── data/
│   └── raw/                        # Raw .htm SEC filings from Kaggle
│
├── outputs/
│   └── all_filing_summaries.csv    # Generated summaries for all filings
│
├── CS583_Project.ipynb         # Full pipeline: preprocessing → training → evaluation
├── ProjectReport_NehaNagaraj.pdf   # Final project report
└── README.md
```

---

## 🔄 Pipeline

```
Raw SEC Filing (.htm)
       ↓
HTML Parsing (BeautifulSoup)
       ↓
MD&A Section Extraction
       ↓
Cleaning (whitespace, XBRL, tables, scripts)
       ↓
Tokenization (BART — 512 token truncation)
       ↓
BART Encoder–Decoder (fine-tuned)
       ↓
Generated Summary
```

---

## 🔍 Methodology

### Dataset
- **Source:** [SEC EDGAR Annual 10-K Filings (2021) — Kaggle](https://www.kaggle.com/)
- **Format:** Raw HTML `.htm` filings
- **Final dataset:** 191 cleaned MD&A excerpts, each paired with a pseudo-summary

### Preprocessing
- HTML parsing with `BeautifulSoup`
- Removal of styling, tables, scripts, and XBRL blocks
- Unicode normalization and whitespace cleanup
- MD&A section extraction
- Truncation to 512 tokens (BART max: 1024)

### Pseudo-Labels
- Reference summaries generated from the **first three sentences** of each cleaned document
- Provides weak but structured supervision

### Model & Training
| Parameter | Value |
|-----------|-------|
| Model | `facebook/bart-large-cnn` |
| Epochs | 5 |
| Learning rate | 2e-5 |
| Batch size | 1 (gradient accumulation: 8) |
| Mixed precision | fp16 |
| Hardware | Google Colab T4 GPU |
| Trainer | HuggingFace `Seq2SeqTrainer` |

---

## 📊 Results

### Loss Curves
Evaluation loss decreased from **0.498 → 0.091** across 5 epochs, with training loss at epoch 5 of **0.098** — indicating stable learning and minimal overfitting.

### ROUGE Scores

| Metric | Score |
|--------|-------|
| ROUGE-1 | 0.5808 |
| ROUGE-2 | 0.5340 |
| ROUGE-L | 0.5763 |

Given pseudo-label supervision, these scores indicate the model effectively learns the structural summarization patterns of MD&A sections — including operational performance, risks, and financial trends.

---

## ✅ Strengths & ⚠️ Limitations

**Strengths**
- Full end-to-end pipeline from raw HTML filings to generated summaries
- BART performs well even under noisy weak supervision
- Stable loss curves with no signs of significant overfitting

**Limitations**
- Pseudo-summaries may inflate ROUGE scores relative to human-written references
- Only the first 512 tokens of each MD&A are used due to model constraints
- Model occasionally repeats formatting or introductory tokens

---

## 🛠️ Tech Stack

- **Language:** Python
- **Libraries:** `transformers`, `datasets`, `torch`, `beautifulsoup4`, `rouge-score`, `pandas`
- **Hardware:** Google Colab (T4 GPU)

---

## 📄 References

- Lewis, M. et al. "BART: Denoising Sequence-to-Sequence Pretraining for Natural Language Generation." ACL 2020.
- Raffel, C. et al. "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer." JMLR 2020.
- Yang, Y. et al. "FinBERT: Financial Domain BERT." arXiv:2006.08097.
- Kaggle: "SEC EDGAR Annual Financial Filings 2021."

---

## 👩‍💻 Author

**Neha Nagaraj**  
M.S. Data Science, Stevens Institute of Technology  
[LinkedIn](https://www.linkedin.com/in/neha-nagaraj-23j2002/) · neha23nagaraj@gmail.com
