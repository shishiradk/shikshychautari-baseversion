"""Build leave-one-out eval cases for every configured subject.
Deletes and rebuilds existing case folders for the listed subjects."""
import os, shutil, glob, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASES = os.path.join(ROOT, "eval", "cases")

# Each entry: (case_prefix, syllabus_path, [(label, paper_path), ...], [book_paths])
SUBJECTS = [
    (
        "cprog",
        "data/raw/cprogramming/c_programming_syllabus.pdf",
        [
            ("2077", "data/raw/cprogramming/TU_C_Programming_2077_Exam.pdf"),
            ("2078", "data/raw/cprogramming/TU_C_Programming_2078_Exam.pdf"),
            ("2079", "data/raw/cprogramming/TU_C_Programming_2079_Exam.pdf"),
            ("2081", "data/raw/cprogramming/TU_C_Programming_2081_Exam.pdf"),
        ],
        ["data/raw/cprogramming/c_programming_notes.pdf"],
    ),
    (
        "dl",
        "data/raw/DL/digital_logic_syllabus.pdf",
        [
            ("2077", "data/raw/DL/TU_Digital_Logic_2077_Exam.pdf"),
            ("2078", "data/raw/DL/TU_Digital_Logic_2078_Exam.pdf"),
            ("2080", "data/raw/DL/TU_Digital_Logic_2080_Exam.pdf"),
            ("2081", "data/raw/DL/TU_Digital_Logic_2081_Exam.pdf"),
        ],
        ["data/raw/DL/DL.pdf"],
    ),
    (
        "iit",
        "data/raw/IIT/iit syllabus.pdf",
        [
            ("2077", "data/raw/IIT/TU_IIT_2077_Exam.pdf"),
            ("2078", "data/raw/IIT/TU_IIT_2078_Exam.pdf"),
            ("2080", "data/raw/IIT/TU_IIT_2080_Exam.pdf"),
            ("2081", "data/raw/IIT/TU_IIT_2081_Exam.pdf"),
        ],
        ["data/raw/IIT/iitbook.pdf"],
    ),
    (
        "dwdm",
        "data/raw/DWDM/7 SemSyllabus-DWDM.pdf",
        [
            ("2078", "data/raw/DWDM/dwdm 2078.pdf"),
            ("2079", "data/raw/DWDM/dwdm 2079.pdf"),
            ("2080", "data/raw/DWDM/dwdm2080.pdf"),
            ("2081", "data/raw/DWDM/dwdm4.pdf"),
            ("csc410", "data/raw/DWDM/CSC410-Data-Warehousing-and-Data-Mining.pdf"),
            ("modelset", "data/raw/DWDM/dwdm model set.pdf"),
        ],
        sorted(glob.glob(os.path.join(ROOT, "data/raw/DWDM/book/DW Unit*.pdf"))),
    ),
    (
        "java",
        "data/raw/JAVA/7 SemSyllabus-JAVA - converted.pdf",
        [
            ("2079", "data/raw/JAVA/java_2079.pdf"),
            ("2080", "data/raw/JAVA/java 2080.pdf"),
            ("modelset1", "data/raw/JAVA/java_modelset.pdf"),
            ("modelset2", "data/raw/JAVA/model set II.pdf"),
        ],
        sorted(glob.glob(os.path.join(ROOT, "data/raw/JAVA/book/*.pdf"))),
    ),
    (
        "pom",
        "data/raw/POM/7 SemSyllabus-POM.pdf",
        [
            ("2078", "data/raw/POM/pom 2078.pdf"),
            ("2079", "data/raw/POM/pom 2079.pdf"),
            ("2080", "data/raw/POM/POM 2080.pdf"),
            ("2081", "data/raw/POM/POM 2081.pdf"),
            ("mgt411", "data/raw/POM/MGT411-Principles-of-Management.pdf"),
        ],
        sorted(glob.glob(os.path.join(ROOT, "data/raw/POM/book/*.pdf"))),
    ),
    (
        "ds",
        "data/raw/DS/TU_Discrete_Structures_Syllabus.pdf",
        [
            ("2078", "data/raw/DS/TU_Discrete_Structures_2078_Exam.pdf"),
            ("2079", "data/raw/DS/TU_Discrete_Structures_2079_Exam.pdf"),
            ("2080", "data/raw/DS/TU_Discrete_Structures_2080_Exam.pdf"),
            ("2081", "data/raw/DS/TU_Discrete_Structures_2081_Exam.pdf"),
        ],
        [],  # Rosen book hurts DS — style mismatch (theorem-heavy vs TU exam-style questions)
    ),
    (
        "oop",
        "data/raw/OOP/TU_OOP_Syllabus.pdf",
        [
            ("2077", "data/raw/OOP/TU_OOP_2077_Exam.pdf"),
            ("2078", "data/raw/OOP/TU_OOP_2078_Exam.pdf"),
            ("2080", "data/raw/OOP/TU_OOP_2080_Exam.pdf"),
            ("2081", "data/raw/OOP/TU_OOP_2081_Exam.pdf"),
        ],
        ["data/raw/OOP/OOP_EBOOK.pdf"],
    ),
    (
        "micro",
        "data/raw/Microprocessor/TU_Microprocessor_Syllabus.pdf",
        [
            ("2077", "data/raw/Microprocessor/TU_Microprocessor_2077_Exam.pdf"),
            ("2078", "data/raw/Microprocessor/TU_Microprocessor_2078_Exam.pdf"),
            ("2080", "data/raw/Microprocessor/TU_Microprocessor_2080_Exam.pdf"),
            ("2081", "data/raw/Microprocessor/TU_Microprocessor_2081_Exam.pdf"),
        ],
        ["data/raw/Microprocessor/Microprocessor_CSIT_Complete.pdf"],
    ),
]


def build_one_subject(prefix, syllabus, labelled_papers, book_files):
    n = len(labelled_papers)
    if n < 2:
        print(f"[{prefix}] skipped: need >=2 papers, got {n}")
        return 0

    built = 0
    for held in range(n):
        label = labelled_papers[held][0]
        case = os.path.join(CASES, f"{prefix}-{label}")
        if os.path.exists(case):
            shutil.rmtree(case)
        os.makedirs(os.path.join(case, "past"))

        shutil.copy(os.path.join(ROOT, syllabus), os.path.join(case, "syllabus.pdf"))
        shutil.copy(os.path.join(ROOT, labelled_papers[held][1]),
                    os.path.join(case, "reference.pdf"))
        for j, (lbl, p) in enumerate(labelled_papers):
            if j != held:
                shutil.copy(os.path.join(ROOT, p),
                            os.path.join(case, "past", f"{lbl}.pdf"))
        if book_files:
            os.makedirs(os.path.join(case, "book"))
            for k, b in enumerate(book_files):
                shutil.copy(os.path.join(ROOT, b),
                            os.path.join(case, "book", f"b{k:02d}.pdf"))
        built += 1
        print(f"  built {os.path.basename(case)}  (book: {len(book_files)} file(s))")
    return built


total = 0
for prefix, syl, papers, book in SUBJECTS:
    print(f"\n[{prefix}] {len(papers)} papers, {len(book)} book file(s)")
    total += build_one_subject(prefix, syl, papers, book)

print(f"\n=== {total} new case(s) built. Total cases: "
      f"{len(os.listdir(CASES))} ===")
