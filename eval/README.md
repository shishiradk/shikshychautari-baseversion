# Evaluation Harness

Measures generation quality with the metrics requested publicly:
**BERTScore**, syllabus **precision / recall / F1**, and **hallucination rate**.

## Quick start

1. **Install the real BERTScore metric** (otherwise you get a labelled proxy):
   ```bash
   pip install bert-score
   ```

2. **Create test cases.** For each subject, make a folder under `eval/cases/`:

   ```
   eval/cases/
     web-technology-2080/
       syllabus.pdf            # the syllabus
       past/                   # CONTEXT papers given to the generator
         2076.pdf
         2077.pdf
         2078.pdf
       reference.pdf           # GROUND TRUTH: a real paper NOT in past/
       book/                   # optional textbook PDFs
         webtech.pdf
   ```

   The `reference.pdf` must be a **real** past paper that is **not** placed in
   `past/`. If the generator sees it, the BERTScore is meaningless (data leak).
   `.txt` files work too (`syllabus.txt`, `reference.txt`).

3. **Run:**
   ```bash
   python eval/evaluate.py                 # cheap: one generation per case
   python eval/evaluate.py --best-of-n 3   # evaluate the full Best-of-N path
   ```

   Output: a console table + `eval/results.json` with per-case + aggregate
   numbers and a methodology note.

## What each metric means here

| Metric | Definition in this project |
|---|---|
| **BERTScore F1** | Semantic similarity between the generated paper and the held-out real paper. High = the prediction reads like a real exam. |
| **Syllabus Precision** | Of the questions generated, the fraction that map to a syllabus topic (embedding cosine ≥ threshold). High = few off-syllabus questions. |
| **Syllabus Recall** | Of the syllabus topics, the fraction that at least one question covers. High = broad coverage. |
| **Syllabus F1** | Harmonic mean of precision and recall. |
| **Hallucination rate** | Fraction of questions whose nearest source chunk (book if provided, else past papers) is below the grounding threshold. Low = well grounded. |

## Honest caveats — read before quoting numbers

- **Thresholds are uncalibrated.** `--syllabus-threshold` (default 0.80) and
  `--ground-threshold` (default 0.75) are starting guesses. To make the
  numbers defensible: hand-label ~30 (question, on-syllabus?) pairs, pick the
  threshold that best matches your labels, then report *that* threshold
  alongside the metric.
- **Proxy vs. real BERTScore.** Without `bert-score` installed, the script
  reports `embedding_similarity_PROXY`. Never present a proxy as BERTScore.
- **Reference leakage** invalidates BERTScore. Keep `reference.pdf` out of `past/`.
- **Sample size.** Three cases is a smoke test, not evidence. Aim for 10+
  held-out papers across subjects before publishing aggregate numbers.

## For the LinkedIn follow-up

Report: the metric values, the **thresholds used**, the **number of held-out
cases**, and **at least one failure case**. "F1 0.71 over 12 held-out papers
at threshold 0.80, worst case X because Y" is credible. A bare "30%" is not.
