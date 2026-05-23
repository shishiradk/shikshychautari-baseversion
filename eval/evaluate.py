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

# Q5: use sentence-transformers for EVALUATION embeddings (non-circular)
# Falls back to OpenAI embeddings if not installed
def make_eval_embedder():
    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        class STEmbedder:
            def embed_documents(self, texts):
                return _model.encode(texts, show_progress_bar=False).tolist()
        print("[eval embedder] using sentence-transformers/all-MiniLM-L6-v2 (non-circular)")
        return STEmbedder(), "sentence-transformers"

    except ImportError:
        print("[eval embedder] sentence-transformers not installed — falling back to OpenAI embeddings (circular)")
        return OpenAIEmbeddings(), "openai"

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
    """Extract concatenated text from every PDF and TXT in a folder."""
    if not os.path.isdir(folder):
        return ""
    text = ""
    for p in sorted(glob.glob(os.path.join(folder, "*.pdf"))):
        with open(p, "rb") as f:
            text += extract_text_from_pdfs([f.read()])
    for p in sorted(glob.glob(os.path.join(folder, "*.txt"))):
        with open(p, encoding="utf-8") as f:
            text += f.read()
    return text


def read_dir_pdfs_individually(folder):
    """Return list of (filename, text) for each PDF in a folder."""
    if not os.path.isdir(folder):
        return []
    results = []
    for p in sorted(glob.glob(os.path.join(folder, "*.pdf"))):
        with open(p, "rb") as f:
            txt = extract_text_from_pdfs([f.read()])
        if txt and len(txt.strip()) > 50:
            results.append((os.path.basename(p), txt))
    return results


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
    # Cap at 2000 chunks — OpenAI embedding API limit is 300k tokens/request.
    # FAISS retrieves top-k anyway so extra chunks beyond 2000 add no value.
    book_chunks = chunk_text(book_text)[:2000] if book_text and len(book_text.strip()) > 100 else None
    has_book = book_chunks is not None

    tmp_index = os.path.join(tempfile.gettempdir(), f"faiss_eval_{name}")
    create_vector_store(past_chunks, db_path=tmp_index, book_chunks=book_chunks)
    db = load_vector_store(tmp_index)

    # Q2 ablation: --no-rag generates with only the syllabus (no retrieval)
    if getattr(args, "no_rag", False):
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import PromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        llm = ChatOpenAI(model="gpt-4o", temperature=0.3, max_tokens=1400)
        prompt = PromptTemplate.from_template(
            "You are an exam paper generator. Using ONLY the syllabus below, "
            "generate a realistic TU BSc.CSIT exam paper with sections, questions, and marks.\n\n"
            "SYLLABUS:\n{syllabus}\n\nEXAM PAPER:"
        )
        generated = (prompt | llm | StrOutputParser()).invoke({"syllabus": syllabus})
    elif args.best_of_n > 1:
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

    # --- Baseline BERTScore: past papers vs reference (answers Q1) ---
    past_papers = read_dir_pdfs_individually(os.path.join(path, "past"))
    baseline_scores = []
    for _, ptxt in past_papers:
        bs, _ = bert_score_or_proxy(ptxt, reference, embedder)
        baseline_scores.append(bs)
    baseline_avg = round(sum(baseline_scores) / len(baseline_scores), 4) if baseline_scores else None
    baseline_max = round(max(baseline_scores), 4) if baseline_scores else None

    # --- Reference recall: syllabus recall of the REAL paper (answers Q3) ---
    ref_qs = extract_questions(reference)
    if ref_qs:
        ref_emb = embed(embedder, ref_qs)
        ref_covered = (cosine_matrix(ref_emb, t_emb).max(axis=0) >= args.syllabus_threshold)
        ref_recall = round(float(ref_covered.mean()), 4)
    else:
        ref_recall = None

    return {
        "case": name,
        "has_book": has_book,
        "n_questions": len(gen_qs),
        "n_topics": len(topics),
        bname: round(bscore, 4),
        "baseline_bertscore_avg": baseline_avg,
        "baseline_bertscore_max": baseline_max,
        "bert_vs_baseline": round(bscore - baseline_avg, 4) if baseline_avg else None,
        "syllabus_precision": round(precision, 4),
        "syllabus_recall": round(recall, 4),
        "ref_recall": ref_recall,
        "recall_vs_ref": round(recall - ref_recall, 4) if ref_recall is not None else None,
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
    ap.add_argument("--add-baselines", action="store_true",
                    help="Add baseline BERTScore + ref_recall to cached results without regenerating")
    ap.add_argument("--no-rag", action="store_true",
                    help="Q2 ablation: generate with syllabus only, no retrieval. Compare vs normal run.")
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

    # Eval embedder: OpenAI for now (thresholds calibrated for it).
    # Switch to sentence-transformers via make_eval_embedder() once thresholds
    # are recalibrated for all-MiniLM-L6-v2 (0.80/0.75 are too strict for ST).
    embedder = OpenAIEmbeddings()

    # --add-baselines: enrich cached results without regenerating papers
    if args.add_baselines:
        if not cached:
            print("ERROR: no cached results found. Run a normal eval first.")
            sys.exit(1)
        print(f"[baselines] enriching {len(cached)} cached result(s) ...")
        for nm, row in cached.items():
            if "baseline_bertscore_avg" in row:
                print(f"  -> already has baselines: {nm}")
                continue
            case_path = os.path.join(CASES_DIR, nm)
            if not os.path.isdir(case_path):
                print(f"  -> case folder missing, skipping: {nm}")
                continue
            print(f"  -> adding baselines: {nm} ...")
            try:
                reference = read_source(os.path.join(case_path, "reference"))
                syllabus  = read_source(os.path.join(case_path, "syllabus"))
                topics    = extract_topics(syllabus)
                t_emb     = embed(embedder, topics) if topics else None

                past_papers = read_dir_pdfs_individually(os.path.join(case_path, "past"))
                bname = "bertscore_f1" if "bertscore_f1" in row else "embedding_similarity_PROXY"
                bscore = row.get(bname, 0)
                baseline_scores = []
                for _, ptxt in past_papers:
                    bs, _ = bert_score_or_proxy(ptxt, reference, embedder)
                    baseline_scores.append(bs)
                row["baseline_bertscore_avg"] = round(sum(baseline_scores)/len(baseline_scores), 4) if baseline_scores else None
                row["baseline_bertscore_max"] = round(max(baseline_scores), 4) if baseline_scores else None
                row["bert_vs_baseline"] = round(bscore - row["baseline_bertscore_avg"], 4) if row["baseline_bertscore_avg"] else None

                ref_qs = extract_questions(reference)
                if ref_qs and t_emb is not None:
                    ref_emb   = embed(embedder, ref_qs)
                    ref_cov   = (cosine_matrix(ref_emb, t_emb).max(axis=0) >= args.syllabus_threshold)
                    row["ref_recall"]     = round(float(ref_cov.mean()), 4)
                    row["recall_vs_ref"]  = round(row["syllabus_recall"] - row["ref_recall"], 4)
                cached[nm] = row
                # save after each case so a crash doesn't lose all progress
                with open(RESULTS_PATH, encoding="utf-8") as rf:
                    _prev = json.load(rf)
                with open(RESULTS_PATH, "w", encoding="utf-8") as wf:
                    json.dump({**_prev, "per_case": list(cached.values())}, wf, indent=2)
            except Exception as e:
                print(f"  -> ERROR on {nm}: {e}")

        # write enriched results back
        rows = list(cached.values())
        ok   = [r for r in rows if "error" not in r]
        def avg(key):
            vals = [r[key] for r in ok if key in r and r[key] is not None]
            return round(sum(vals)/len(vals), 4) if vals else None
        bert_key = next((k for r in ok for k in r if k in ("bertscore_f1","embedding_similarity_PROXY")), None)
        aggregate = {
            "n_cases_ok": len(ok),
            "n_cases_failed": len(rows)-len(ok),
            "avg_"+(bert_key or "bert"): avg(bert_key) if bert_key else None,
            "avg_baseline_bertscore_avg": avg("baseline_bertscore_avg"),
            "avg_bert_vs_baseline": avg("bert_vs_baseline"),
            "avg_syllabus_precision": avg("syllabus_precision"),
            "avg_syllabus_recall": avg("syllabus_recall"),
            "avg_ref_recall": avg("ref_recall"),
            "avg_recall_vs_ref": avg("recall_vs_ref"),
            "avg_syllabus_f1": avg("syllabus_f1"),
            "avg_hallucination_rate": avg("hallucination_rate"),
        }
        with open(RESULTS_PATH, encoding="utf-8") as rf:
            prev = json.load(rf)
        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump({**prev, "aggregate": aggregate, "per_case": rows}, f, indent=2)

        print("\n" + "=" * 100)
        print(f"{'case':<22}{'BERT':>8}{'Baseline':>10}{'BERT+':>7}{'Rec':>7}{'RefRec':>8}{'Rec+':>7}")
        print("-" * 100)
        for r in sorted(rows, key=lambda x: x["case"]):
            if "error" in r: continue
            bv  = r.get("bertscore_f1", r.get("embedding_similarity_PROXY", 0))
            bb  = r.get("baseline_bertscore_avg") or 0
            bdb = r.get("bert_vs_baseline") or 0
            rc  = r.get("syllabus_recall", 0)
            rr  = r.get("ref_recall") or 0
            dr  = r.get("recall_vs_ref") or 0
            print(f"{r['case']:<22}{bv:>8.3f}{bb:>10.3f}{bdb:>+7.3f}{rc:>7.3f}{rr:>8.3f}{dr:>+7.3f}")
        print("-" * 100)
        print(f"{'AVERAGE':<22}"
              f"{(aggregate.get('avg_bertscore_f1') or 0):>8.3f}"
              f"{(aggregate.get('avg_baseline_bertscore_avg') or 0):>10.3f}"
              f"{(aggregate.get('avg_bert_vs_baseline') or 0):>+7.3f}"
              f"{(aggregate.get('avg_syllabus_recall') or 0):>7.3f}"
              f"{(aggregate.get('avg_ref_recall') or 0):>8.3f}"
              f"{(aggregate.get('avg_recall_vs_ref') or 0):>+7.3f}")
        print("=" * 100)
        print(f"Full report -> {RESULTS_PATH}")
        return

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
