"""
Evaluation harness for the Question Paper Generator.

Measures, per test case and in aggregate:
  - BERTScore  (generated paper vs. held-out REAL paper)        -> semantic fidelity
  - Syllabus Precision  (% of generated questions on-syllabus)
  - Syllabus Recall     (% of syllabus topics covered)
  - Syllabus F1         (harmonic mean of the two)
  - Hallucination rate  (% of questions not grounded in source)

METHODOLOGY (read before quoting numbers anywhere):
  * "On-syllabus" / "covered" / "grounded" are decided by cosine similarity
    between OpenAI embeddings, thresholded. Thresholds are heuristics and
    MUST be calibrated on a labelled sample before the numbers are
    publication-grade. Defaults are conservative starting points, not truth.
  * BERTScore uses the `bert-score` library if installed (the metric the
    LinkedIn commenter asked for). If it is not installed, the script falls
    back to OpenAI-embedding cosine similarity and labels it clearly as a
    PROXY -- never report a proxy as "BERTScore".
  * Reference papers are HELD OUT: a real past paper that is NOT supplied to
    the generator as context. Leakage invalidates the metric.

Folder layout (create under eval/cases/):

  eval/cases/
    <case_name>/
      syllabus.pdf          (or syllabus.txt)        REQUIRED
      past/                  *.pdf  context papers    REQUIRED (>=1)
      reference.pdf          (or reference.txt)       REQUIRED  <- ground truth
      book/                  *.pdf  optional textbook OPTIONAL

Run:
  python eval/evaluate.py                 # single-shot generation (cheap)
  python eval/evaluate.py --best-of-n 3   # evaluate the full Best-of-N path
  python eval/evaluate.py --syllabus-threshold 0.80 --ground-threshold 0.75
"""

import os
import re
import sys
import json
import glob
import argparse
import tempfile
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings
from utils import (
    extract_text_from_pdfs, chunk_text, create_vector_store, load_vector_store,
    strip_headers_and_footers, clean_body_keep_all_marks,
    generate_predicted_paper, generate_predicted_paper_best_of_n,
)

CASES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cases")
RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")


# ----------------------------- helpers --------------------------------------

def read_source(path_no_ext):
    """Read a .pdf or .txt source given a path without extension."""
    if os.path.exists(path_no_ext + ".pdf"):
        with open(path_no_ext + ".pdf", "rb") as f:
            return extract_text_from_pdfs([f.read()])
    if os.path.exists(path_no_ext + ".txt"):
        with open(path_no_ext + ".txt", "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    return ""


def read_dir_pdfs(folder):
    """Extract concatenated text from every PDF in a folder."""
    if not os.path.isdir(folder):
        return ""
    pdfs = sorted(glob.glob(os.path.join(folder, "*.pdf")))
    if not pdfs:
        return ""
    blobs = []
    for p in pdfs:
        with open(p, "rb") as f:
            blobs.append(f.read())
    return extract_text_from_pdfs(blobs)


def extract_questions(paper_text):
    """Pull individual question strings from a paper.

    Heuristic: a question is a line that starts with a number-dot, OR a line
    that contains a marks bracket like [5] / [2+3]. Section headers and
    'Attempt any ...' instructions are excluded.
    """
    questions = []
    for raw in paper_text.splitlines():
        line = raw.strip()
        if len(line) < 15:
            continue
        if re.match(r"^(section|group)\s+[a-z]", line, re.I):
            continue
        if re.match(r"^(attempt|answer)\s+any", line, re.I):
            continue
        is_numbered = bool(re.match(r"^\(?\d{1,2}[\.\)]", line))
        has_marks = bool(re.search(r"\[\s*\d+(?:\s*\+\s*\d+)*\s*\]", line))
        if is_numbered or has_marks:
            # strip leading numbering and trailing marks bracket
            q = re.sub(r"^\(?\d{1,2}[\.\)]\s*", "", line)
            q = re.sub(r"\[\s*\d+(?:\s*\+\s*\d+)*\s*\]\s*$", "", q).strip()
            if len(q) > 15:
                questions.append(q)
    return questions


def extract_topics(syllabus_text):
    """Pull syllabus topic lines."""
    topics = []
    for raw in syllabus_text.splitlines():
        line = raw.strip()
        line = re.sub(r"^[\d\.\)\-\*•]+\s*", "", line).strip()
        if len(line) >= 8:
            topics.append(line)
    return topics[:60]


def cosine_matrix(a, b):
    """Cosine similarity matrix between row-vectors of a and b."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    a /= (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b /= (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return a @ b.T


def embed(embedder, texts):
    if not texts:
        return np.zeros((0, 1536), dtype=np.float32)
    return np.asarray(embedder.embed_documents(texts), dtype=np.float32)


def bert_score_or_proxy(generated, reference, embedder):
    """Return (value, metric_name). Real BERTScore if lib present, else proxy."""
    try:
        from bert_score import score as _bs
        P, R, F1 = _bs([generated], [reference], lang="en", verbose=False)
        return float(F1.mean()), "bertscore_f1"
    except Exception:
        g = embed(embedder, [generated])
        r = embed(embedder, [reference])
        if len(g) == 0 or len(r) == 0:
            return 0.0, "embedding_similarity_PROXY"
        return float(cosine_matrix(g, r)[0, 0]), "embedding_similarity_PROXY"


# ----------------------------- per-case eval --------------------------------

def evaluate_case(name, path, args, embedder):
    syllabus = read_source(os.path.join(path, "syllabus"))
    reference = read_source(os.path.join(path, "reference"))
    past_text = read_dir_pdfs(os.path.join(path, "past"))
    book_text = read_dir_pdfs(os.path.join(path, "book"))

    missing = [k for k, v in
               {"syllabus": syllabus, "reference": reference, "past": past_text}.items()
               if not v or len(v.strip()) < 50]
    if missing:
        return {"case": name, "error": f"missing/empty: {', '.join(missing)}"}

    # Build the vector store exactly like the app does
    clean_past = clean_body_keep_all_marks(strip_headers_and_footers(past_text))
    past_chunks = chunk_text(clean_past)
    book_chunks = chunk_text(book_text) if book_text and len(book_text.strip()) > 100 else None
    has_book = book_chunks is not None

    tmp_index = os.path.join(tempfile.gettempdir(), f"faiss_eval_{name}")
    create_vector_store(past_chunks, db_path=tmp_index, book_chunks=book_chunks)
    db = load_vector_store(tmp_index)

    if args.best_of_n > 1:
        generated, _, _ = generate_predicted_paper_best_of_n(
            db, syllabus, has_book=has_book, n=args.best_of_n)
    else:
        generated, _ = generate_predicted_paper(db, syllabus, has_book=has_book)

    gen_qs = extract_questions(generated)
    topics = extract_topics(syllabus)
    if not gen_qs:
        return {"case": name, "error": "no questions parsed from generated paper"}
    if not topics:
        return {"case": name, "error": "no topics parsed from syllabus"}

    # --- embeddings ---
    q_emb = embed(embedder, gen_qs)
    t_emb = embed(embedder, topics)
    src_text = (book_text if has_book else clean_past)
    src_chunks = chunk_text(src_text)[:200] or [src_text[:1000]]
    s_emb = embed(embedder, src_chunks)

    # --- syllabus precision / recall / F1 ---
    q_t = cosine_matrix(q_emb, t_emb)                     # questions x topics
    on_syllabus = (q_t.max(axis=1) >= args.syllabus_threshold)
    covered = (q_t.max(axis=0) >= args.syllabus_threshold)
    precision = float(on_syllabus.mean())
    recall = float(covered.mean())
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    # --- hallucination rate ---
    q_s = cosine_matrix(q_emb, s_emb)                     # questions x source
    grounded = (q_s.max(axis=1) >= args.ground_threshold)
    hallucination_rate = float(1.0 - grounded.mean())

    # --- BERTScore (or labelled proxy) ---
    bscore, bname = bert_score_or_proxy(generated, reference, embedder)

    return {
        "case": name,
        "has_book": has_book,
        "n_questions": len(gen_qs),
        "n_topics": len(topics),
        bname: round(bscore, 4),
        "syllabus_precision": round(precision, 4),
        "syllabus_recall": round(recall, 4),
        "syllabus_f1": round(f1, 4),
        "hallucination_rate": round(hallucination_rate, 4),
    }


# ----------------------------- main -----------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Evaluate the Question Paper Generator.")
    ap.add_argument("--best-of-n", type=int, default=1,
                    help="1 = single generation (cheap). >1 = evaluate Best-of-N path.")
    ap.add_argument("--syllabus-threshold", type=float, default=0.80,
                    help="Cosine sim threshold for 'on-syllabus' / 'covered'. CALIBRATE.")
    ap.add_argument("--ground-threshold", type=float, default=0.75,
                    help="Cosine sim threshold for 'grounded in source'. CALIBRATE.")
    ap.add_argument("--skip", nargs="*", default=[],
                    help="Subject prefixes to skip, e.g. --skip dwdm java")
    ap.add_argument("--rerun", nargs="*", default=[],
                    help="Force re-run specific cases even if cached, e.g. --rerun cprog-2077")
    ap.add_argument("--rerun-all", action="store_true",
                    help="Ignore cache and re-run every case (full fresh eval)")
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set (.env or environment).")
        sys.exit(1)
    if not os.path.isdir(CASES_DIR):
        print(f"ERROR: no cases directory at {CASES_DIR}. See module docstring.")
        sys.exit(1)

    skip_prefixes = tuple(s.lower() for s in (args.skip or []))
    case_names = sorted(
        d for d in os.listdir(CASES_DIR)
        if os.path.isdir(os.path.join(CASES_DIR, d))
        and not any(d.lower().startswith(p) for p in skip_prefixes)
    )
    if not case_names:
        print(f"ERROR: no case folders in {CASES_DIR}. See module docstring.")
        sys.exit(1)

    # Load cached results so we can skip already-evaluated cases
    cached = {}
    if not args.rerun_all and os.path.exists(RESULTS_PATH):
        try:
            with open(RESULTS_PATH, encoding="utf-8") as f:
                prev = json.load(f)
            for r in prev.get("per_case", []):
                if "error" not in r:
                    cached[r["case"]] = r
            if cached:
                print(f"[cache] loaded {len(cached)} existing result(s) from results.json")
        except Exception:
            pass  # corrupt file — just re-run everything

    force_rerun = set(args.rerun or [])

    embedder = OpenAIEmbeddings()
    rows = []
    for nm in case_names:
        if nm in cached and nm not in force_rerun:
            print(f"-> skipping (cached): {nm}")
            rows.append(cached[nm])
            continue
        print(f"-> evaluating: {nm} ...")
        try:
            rows.append(evaluate_case(nm, os.path.join(CASES_DIR, nm), args, embedder))
        except Exception as e:
            rows.append({"case": nm, "error": repr(e)})

    ok = [r for r in rows if "error" not in r]

    def avg(key):
        vals = [r[key] for r in ok if key in r]
        return round(sum(vals) / len(vals), 4) if vals else None

    bert_key = next((k for r in ok for k in r
                     if k in ("bertscore_f1", "embedding_similarity_PROXY")), None)
    aggregate = {
        "n_cases_ok": len(ok),
        "n_cases_failed": len(rows) - len(ok),
        "avg_" + (bert_key or "bert"): avg(bert_key) if bert_key else None,
        "avg_syllabus_precision": avg("syllabus_precision"),
        "avg_syllabus_recall": avg("syllabus_recall"),
        "avg_syllabus_f1": avg("syllabus_f1"),
        "avg_hallucination_rate": avg("hallucination_rate"),
    }

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": vars(args),
        "methodology_note": (
            "Thresholds are uncalibrated heuristics. "
            "'embedding_similarity_PROXY' is NOT BERTScore -- install "
            "`bert-score` for the real metric. Reference papers must be "
            "held out from generator context."
        ),
        "aggregate": aggregate,
        "per_case": rows,
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # console table
    print("\n" + "=" * 78)
    print(f"{'case':<22}{'BERT/proxy':>12}{'Prec':>8}{'Rec':>8}{'F1':>8}{'Halluc':>9}")
    print("-" * 78)
    for r in rows:
        if "error" in r:
            print(f"{r['case']:<22}  ERROR: {r['error'][:46]}")
            continue
        bv = r.get("bertscore_f1", r.get("embedding_similarity_PROXY", 0))
        print(f"{r['case']:<22}{bv:>12.3f}{r['syllabus_precision']:>8.3f}"
              f"{r['syllabus_recall']:>8.3f}{r['syllabus_f1']:>8.3f}"
              f"{r['hallucination_rate']:>9.3f}")
    print("-" * 78)
    a = aggregate
    print(f"{'AVERAGE (' + str(a['n_cases_ok']) + ' ok)':<22}"
          f"{(a.get('avg_' + (bert_key or 'bert')) or 0):>12.3f}"
          f"{(a['avg_syllabus_precision'] or 0):>8.3f}"
          f"{(a['avg_syllabus_recall'] or 0):>8.3f}"
          f"{(a['avg_syllabus_f1'] or 0):>8.3f}"
          f"{(a['avg_hallucination_rate'] or 0):>9.3f}")
    print("=" * 78)
    if bert_key == "embedding_similarity_PROXY":
        print("NOTE: BERT column is an embedding PROXY. `pip install bert-score` "
              "for the real metric before quoting it.")
    print(f"Full report -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()
