# Deployment & Development Instructions

## Dependency files — why there are two

| File | Purpose | Used by |
|---|---|---|
| `requirements.txt` | **Trimmed** — only packages the code actually imports (~15). Fast install. | GitHub **and** Hugging Face Space |
| `requirements-full.txt` | The original full list, including heavy ML libs (`torch`, `transformers`, `sentence-transformers`) kept for future features (local models, OCR, fine-tuning). | Reference only — install manually if you add those features |

**Why trim?** The full list pulls ~2 GB (torch + transformers). On Hugging Face the build then takes ~10 minutes. None of those packages are imported by `app.py`, `utils.py`, or `api.py`, so the trimmed list builds in ~2 minutes with identical behavior.

**If you add a feature that needs the heavy libs** (e.g. local Llama inference, OCR with pytesseract), either:
- add just that package to `requirements.txt`, or
- switch to the full list: `pip install -r requirements-full.txt`

## Run locally

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
streamlit run app.py
```

Enter your OpenAI API key in the app sidebar (or put `OPENAI_API_KEY=sk-...` in a `.env` file).

## Deploy to Hugging Face Spaces

1. Create a Space at https://huggingface.co/new-space — SDK: **Streamlit**, hardware: **CPU basic (free)**.
2. Create a **Write** token at https://huggingface.co/settings/tokens (never paste it anywhere except the git password prompt).
3. Push:
   ```bash
   git remote add space https://huggingface.co/spaces/<USERNAME>/<SPACE_NAME>
   git push space main
   ```
   Username = your HF username. Password = the **write token**.
4. HF reads the YAML header in `README.md` to configure the Space, installs `requirements.txt`, and runs `app.py`.

### API key on the Space
The app has a password-type input in the sidebar. **Each visitor enters their own OpenAI key** — it is used only for that session and never stored. The Space owner pays nothing. Do **not** add `OPENAI_API_KEY` as a Space secret unless the Space is private.

## What is NOT shipped to the Space

Configured in `.gitignore`:
- `venv/`, `env1/` — virtual environments
- `faiss_index/` — regenerated at runtime on each "Process Files"
- `data/`, sample `*.pdf`, `*.docx`, `Web Technology*/` — local test data; users upload their own
- `.env` — secrets
- `interview_prep.pdf`, `generate_interview_prep.py` — local interview-prep artifacts

The Space runs fine without sample data — the workflow is upload-your-own-files.

## Files that ARE shipped

`app.py`, `utils.py`, `api.py`, `requirements.txt`, `requirements-full.txt`, `README.md` (with HF YAML header), `LICENSE`, `.gitignore`, `INSTRUCTIONS.md`.
