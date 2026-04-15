# Shikshya Chautari — Question Paper Generator

An AI-powered Streamlit app that generates predicted exam question papers from a syllabus, past question papers, and (optionally) a reference book. Built for Nepali university exams (BSc CSIT, BIT, BIM, BCA) but subject-agnostic.

## How it works

1. **Upload** syllabus PDF + past question papers + (optional) reference book
2. **Process** — text is extracted, cleaned, chunked, and indexed in a FAISS vector store
3. **Generate** — the app produces 3 candidate papers using `gpt-4o`, an LLM verifier scores each, and the best is selected
4. **Download** the result as PDF

When a book is uploaded, every generated question is grounded in the book's content. Without a book, past papers provide style and the syllabus provides topics.

## Requirements

- Python 3.9+
- OpenAI API key

## Setup

```bash
# 1. Create a virtual environment
python -m venv venv
venv\Scripts\activate         # Windows
source venv/bin/activate      # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create a .env file in the project root
echo OPENAI_API_KEY=sk-your-key-here > .env

# 4. Run the app
streamlit run app.py
```

The app opens at http://localhost:8501.

## Usage

1. In the sidebar, upload:
   - **Syllabus PDF** (required) — filename becomes the subject name
   - **Past question papers** (required, one or more)
   - **Reference book PDF** (optional) — questions will be strictly grounded in book content when provided
2. Click **Process Files**
3. Click **Generate Predicted Paper**
4. The app shows all 3 candidate scores and picks the best
5. Download the result as PDF

## Architecture

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI, file upload, generation pipeline |
| `utils.py` | PDF extraction, FAISS indexing, generation, verifier, PDF export |
| `api.py` | FastAPI REST endpoint with S3 integration (optional) |
| `faiss_index/` | Persisted vector store (regenerated on each "Process Files") |

### Generation pipeline
1. Past papers → chunked (1000 chars) → embedded → FAISS (tagged `source: past_paper`)
2. Book → chunked → FAISS (tagged `source: book`)
3. On generate: retrieve past-paper chunks for style + book chunks for content
4. Best-of-N: generate 3 candidates with `gpt-4o` → verifier (`gpt-4o-mini`) scores each 0–100 → pick max

### Scoring rubric
- Marks present on every question (25 pts)
- Section headers formatted `[X x Y = Z]` (20 pts)
- Structure consistency (15 pts)
- Question quality / non-duplication (15 pts)
- Book/syllabus grounding (25 pts)

## Cost per paper

~$0.05 on OpenAI (3 generations with `gpt-4o` + 3 verifier calls with `gpt-4o-mini`).

## Optional: FastAPI + S3

The `api.py` file exposes a REST endpoint that pulls past papers and syllabi from an S3 bucket, generates a paper, and uploads the result back. Requires:

```
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=eu-north-1
```

Run with:
```bash
uvicorn api:app --reload
```
Swagger UI at http://localhost:8000/docs.

## Project structure

```
shikshychautari-baseversion1/
├── app.py                  # Streamlit web app
├── api.py                  # FastAPI REST backend
├── utils.py                # Core logic: extraction, indexing, generation
├── requirements.txt
├── .env                    # OPENAI_API_KEY (create this)
├── data/
│   └── raw/                # Example syllabi and past papers
│       ├── DWDM/
│       ├── JAVA/
│       ├── POM/
│       └── SPM/
└── faiss_index/            # Generated at runtime
```

## Troubleshooting

**"OPENAI_API_KEY is not set"**
Create a `.env` file in the project root with your key.

**Scanned / image-based PDFs**
PyPDF2 can't read images. Run scanned PDFs through an OCR tool (`ocrmypdf`, Tesseract, or Adobe Acrobat's OCR) before uploading.

**All candidates score low**
Indicates the past papers don't parse well or the syllabus is too short. Check the raw text extraction by running `extract_text_from_pdfs` on your file in a Python shell.

**Questions look generic despite uploading a book**
The book may be a scanned PDF (no extractable text). Confirm by opening the book PDF and trying to select text — if you can't, it needs OCR.

## Notes

- FAISS index is overwritten on each "Process Files" — only one subject is active at a time
- Generator model: `gpt-4o`. Verifier model: `gpt-4o-mini`
- Temperature: 0.3 for generation, 0.0 for verifier
