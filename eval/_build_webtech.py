"""Build leave-one-out eval cases for Web Technology from the 5 clean
pure question papers. Reference is always a held-out pure question paper
that does NOT appear in past/ (no leakage)."""
import os, shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WT = os.path.join(ROOT, "Web Technology-20260415T132741Z-3-001", "Web Technology")
SYLLABUS = os.path.join(ROOT, "CSC318-Web-Technology.pdf")
CASES = os.path.join(ROOT, "eval", "cases")

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
    print(f"built webtech-{held_out}: reference={held_out}, "
          f"past={[y for y in years if y != held_out]}")

print(f"\n{len(years)} cases created under {CASES}")
