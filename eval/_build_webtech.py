"""Build leave-one-out eval cases for Web Technology from the 5 clean
pure question papers. Reference is always a held-out pure question paper
that does NOT appear in past/ (no leakage)."""
import os, shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WT = os.path.join(ROOT, "data", "raw", "WebTechnology")
SYLLABUS = os.path.join(WT, "CSC318-Web-Technology-Syllabus.pdf")
CASES = os.path.join(ROOT, "eval", "cases")

# KEC WebTech book (OCR'd) — TU-aligned, written by Nepali authors for CSC318.
# Covers Bootstrap, jQuery, AJAX, JSON — topics missing from all Western books.
KEC_OCR = os.path.join(WT, "book", "WebTechKecSem5_222_ocr.txt")

# year -> source filename (pure question papers only)
PAPERS = {
    "2073": "2073 questionpaper.pdf",
    "2074": "2074 questionpaper.pdf.pdf",
    "2075": "2075 questionpaper.pdf.pdf",
    "2076": "2076 questionpaper (new).pdf",
    "2078": "2078 questionpaper (new).pdf",
}

years = list(PAPERS)
for held_out in years:
    case = os.path.join(CASES, f"webtech-{held_out}")
    past = os.path.join(case, "past")
    if os.path.exists(case):
        shutil.rmtree(case)
    os.makedirs(past)

    shutil.copy(SYLLABUS, os.path.join(case, "syllabus.pdf"))
    shutil.copy(os.path.join(WT, PAPERS[held_out]),
                os.path.join(case, "reference.pdf"))
    for y in years:
        if y != held_out:
            shutil.copy(os.path.join(WT, PAPERS[y]),
                        os.path.join(past, f"{y}.pdf"))
    if os.path.exists(KEC_OCR):
        book_dir = os.path.join(case, "book")
        os.makedirs(book_dir)
        shutil.copy(KEC_OCR, os.path.join(book_dir, "kec_webtech_ocr.txt"))
    print(f"built webtech-{held_out}: reference={held_out}, "
          f"past={[y for y in years if y != held_out]}, "
          f"book={'KEC OCR txt' if os.path.exists(KEC_OCR) else 'none'}")

print(f"\n{len(years)} cases created under {CASES}")
