import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
 
import   re
from io import BytesIO



try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.enums import TA_CENTER
    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False

def extract_text_from_pdfs(pdfs):
    full_text = ""
    for pdf in pdfs:
        # Handle both file objects and bytes data
        if isinstance(pdf, bytes):
            from io import BytesIO
            pdf_stream = BytesIO(pdf)
            reader = PdfReader(pdf_stream)
        else:
            reader = PdfReader(pdf)
        
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text
    return full_text

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    return splitter.split_text(text)

def create_vector_store(chunks, db_path="faiss_index", book_chunks=None):
    embeddings = OpenAIEmbeddings()
    all_texts = list(chunks)
    metadatas = [{"source": "past_paper"}] * len(chunks)
    if book_chunks:
        all_texts.extend(book_chunks)
        metadatas.extend([{"source": "book"}] * len(book_chunks))
    vectorstore = FAISS.from_texts(all_texts, embedding=embeddings, metadatas=metadatas)
    vectorstore.save_local(db_path)

def load_vector_store(db_path="faiss_index"):
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)


# -------- Cleaning --------
def strip_headers_and_footers(text: str) -> str:
    lines = text.splitlines()
    clean = []
    for line in lines:
        if any(kw in line.lower() for kw in [
            "tribhuvan", "university", "institute", "science",
            "technology", "full marks", "pass marks", "time",
            "semester", "year", "candidates are required"
        ]):
            continue
        clean.append(line)
    return "\n".join(clean).strip()

def clean_body_keep_all_marks(text: str) -> str:
    text = re.sub(r"\*+|_+|`+|#+", "", text)      # remove artifacts
    text = re.sub(r"\.\.\.+", "", text)           # remove ...
    text = re.sub(r"\s{2,}", " ", text)           # collapse spaces
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text) # collapse big gaps
    return text.strip()

def enforce_structure(text: str) -> str:
    text = re.sub(r"(Section\s+[A-Z])", r"\n\1", text, flags=re.I)
    text = re.sub(r"(GROUP\s+[A-Z])", r"\n\1", text, flags=re.I)
    text = re.sub(r"(?<!^)(\s*)(\d+\.)", r"\n\2", text)  # each question new line
    return text

def split_group_instructions(text: str) -> str:
    """
    Put 'Answer any ...' instructions on a new line after Group/Section headings,
    but keep them left aligned (not centered).
    """
    text = re.sub(r"((Group|Section)\s+[A-Z])\s+(Answer any.+)", r"\1\n\3", text, flags=re.I)
    return text

def enforce_pom1_style(text: str) -> str:
    """
    Remove double numbering (standalone digits and duplicates before 'n.').
    Handles patterns like '1 1.', '2) 2.' etc.
    """
    out_lines = []
    for line in text.splitlines():
        l = line.strip()
        if re.fullmatch(r"\d+", l):
            continue
        l = re.sub(r"^(\d+)[\s\.)]+(\1\.)", r"\2", l)
        out_lines.append(l)
    return "\n".join(out_lines)

def center_sections(text: str) -> str:
    lines = text.splitlines()
    out = []
    for line in lines:
        if re.match(r"^(Section\s+[A-Z])", line.strip(), flags=re.I):
            out.append(line.strip().center(80))
        else:
            out.append(line.strip())
    return "\n".join(out)


def generate_predicted_paper(past_db, syllabus_text, has_book=False):
    # Retrieve past-paper docs for STYLE (structure, numbering, marks)
    style_docs = past_db.similarity_search(
        "exam structure sections numbering marks distribution typical topics", k=12
    )
    past_docs = [d for d in style_docs if d.metadata.get("source") != "book"]
    past_context = "\n".join([d.page_content for d in past_docs]) or \
                   "\n".join([d.page_content for d in style_docs])

    # Retrieve book docs for CONTENT grounding (only if book uploaded)
    book_context = ""
    if has_book:
        book_docs = []
        # Query by syllabus lines so retrieval stays on-syllabus
        queries = [q.strip() for q in syllabus_text.splitlines() if len(q.strip()) > 8][:20]
        if not queries:
            queries = ["definition", "concept", "theory", "example", "explanation"]
        for q in queries:
            try:
                hits = past_db.similarity_search(q, k=3)
                book_docs.extend([h for h in hits if h.metadata.get("source") == "book"])
            except Exception:
                pass
        # Dedupe
        seen = set()
        unique = []
        for d in book_docs:
            h = hash(d.page_content[:120])
            if h not in seen:
                seen.add(h)
                unique.append(d)
        book_context = "\n\n".join([d.page_content for d in unique[:15]])

    chain = get_question_generator_chain(has_book=has_book)
    invoke_args = {
        "past_questions": past_context,
        "syllabus": syllabus_text,
    }
    if has_book:
        invoke_args["book_content"] = book_context or "(no book content retrieved)"
    return chain.invoke(invoke_args), past_context





def get_question_generator_chain(has_book=False):
    book_rules = ""
    book_section = ""
    input_vars = ["past_questions", "syllabus"]
    if has_book:
        input_vars = ["past_questions", "syllabus", "book_content"]
        book_section = """

--- BOOK CONTENT (PRIMARY SOURCE FOR QUESTIONS) ---
{book_content}
"""
        book_rules = """

--- BOOK-GROUNDING RULES (STRICT) ---
1. Every question MUST be answerable strictly from the BOOK CONTENT above.
2. Do NOT generate questions about topics absent from the BOOK CONTENT, even if they appear in the syllabus.
3. Do NOT invent facts, code, or examples outside the BOOK CONTENT.
4. Use the SYLLABUS only to decide WHICH book topics are exam-relevant.
5. Use the PAST PAPER only for style, structure, numbering, and marks — not for content."""
    template = """
 You are an expert academic exam paper generator for university-level exams.

Task:
Generate **exactly one complete future exam paper** strictly following the structure, style, and formatting of the provided past paper.

--- STRUCTURE RULES ---
1. Use the exact same number of sections as in the past paper.
2. Preserve all section titles exactly (e.g., Section A, Section B).
3. Place section instructions (e.g., "Attempt any 2 questions") on a separate line immediately below the section title.
4. For section instructions, calculate total marks dynamically using the formula:
   [Number_of_questions x Marks_per_question = Total_marks_for_section]
   Example: "Attempt any 2 questions [2 x 10 = 20]"
5. Preserve exact numbering style (global or per section) and number of questions per section.
6. Keep per-question and sub-question marks exactly as in the past paper (e.g., [2+3], [5], [1+4+5]).
7. Center all section titles.
8. Start each section on a new line.
9. Replicate multi-type questions (objective, short answer, long answer) exactly as in past papers.
10. Do NOT include headers, footers, university name, course name, or total marks table.

--- CONTENT RULES ---
1. Generate **fresh exam questions** based on the syllabus topics provided.
2. Match the **difficulty, phrasing, and style** of the past paper.
3. Include sub-questions exactly if the past paper has them (e.g., 1(a), 1(b)).
4. Use **formal academic language** suitable for university-level exams.
5. Include sub-questions exactly if the past paper has them (e.g., 1(a), 1(b)).
6. Handle short notes or multi-part questions like this example:
    Write short notes on:[marks per question]
    a. question
    b. question
    

--- MARKS RULES (MANDATORY) ---
- EVERY question MUST end with its marks in square brackets. No exceptions.
  Examples: "... explain with example. [5]" or "... derive the formula. [2+3]"
- If a question has sub-parts, write the combined mark like [2+3] or [1+4+5].
- Section headers MUST include total marks formatted as `[X x Y = Z]`
  (e.g., "Section B Attempt any Eight Questions [8 x 5 = 40]").
- Dynamically calculate section totals using multiplication.
- Preserve sub-question marks as they appear in the past paper.
- Do NOT output incorrect formats like `[85 = 40]`.
- If a question has NO marks bracket, the paper is invalid — always include one.

--- OUTPUT FORMAT ---
- Only output:
  1. Section titles (centered)
  2. Section instructions with dynamically calculated marks
  3. Questions with numbering, sub-questions, and marks
- No extra text, no explanations, no headers or footers.

--- INPUT VARIABLES ---
- {past_questions}: Past paper questions to replicate style and formatting
- {syllabus}: Syllabus topics to generate fresh questions
""" + book_section + book_rules + """

Output:
A single, fully formatted exam paper following all rules above.

"""
    prompt = PromptTemplate(
        input_variables=input_vars,
        template=template,
    )
    llm = ChatOpenAI(model="gpt-4o", temperature=0.3, max_tokens=1400)
    output_parser = StrOutputParser()
    return prompt | llm | output_parser




def score_paper(paper: str, syllabus_text: str, has_book: bool, book_excerpt: str = "") -> tuple:
    """Return (score 0-100, feedback) for a generated paper."""
    book_check = ""
    if has_book and book_excerpt:
        book_check = f"""
5. Book grounding (0-25): is every question answerable from the BOOK EXCERPT?
BOOK EXCERPT:
{book_excerpt[:2500]}
"""
    else:
        book_check = "5. Syllabus grounding (0-25): is every question on a syllabus topic?"

    rubric = f"""You are a strict exam-paper reviewer. Score the paper below on a 0-100 scale using this rubric:

1. Marks present (0-25): does EVERY question end with [marks]? -5 per missing bracket.
2. Section headers (0-20): do headers include [X x Y = Z]?
3. Structure (0-15): are sections and numbering consistent?
4. Question quality (0-15): are questions clear, non-duplicate, well-phrased?
{book_check}

SYLLABUS:
{syllabus_text[:2000]}

PAPER:
{paper}

Respond in EXACTLY this format:
SCORE: <integer 0-100>
FEEDBACK: <one-line summary of main issues, or "none">
"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, max_tokens=200)
    resp = llm.invoke(rubric).content
    m = re.search(r"SCORE:\s*(\d+)", resp)
    score = int(m.group(1)) if m else 0
    fb = re.search(r"FEEDBACK:\s*(.+)", resp)
    return score, (fb.group(1).strip() if fb else "")


def generate_predicted_paper_best_of_n(past_db, syllabus_text, has_book=False, n=3):
    """Generate N candidate papers, score them, return the best one."""
    candidates = []
    book_excerpt = ""
    for i in range(n):
        paper, past_context = generate_predicted_paper(past_db, syllabus_text, has_book=has_book)
        if has_book and not book_excerpt:
            # Grab book excerpt once for verifier reuse
            try:
                queries = [q.strip() for q in syllabus_text.splitlines() if len(q.strip()) > 8][:10]
                book_docs = []
                for q in queries or ["concept"]:
                    hits = past_db.similarity_search(q, k=2)
                    book_docs.extend([h for h in hits if h.metadata.get("source") == "book"])
                seen = set()
                for d in book_docs:
                    h = hash(d.page_content[:120])
                    if h not in seen:
                        seen.add(h)
                        book_excerpt += d.page_content + "\n\n"
                    if len(book_excerpt) > 3000:
                        break
            except Exception:
                pass
        try:
            score, feedback = score_paper(paper, syllabus_text, has_book, book_excerpt)
        except Exception as e:
            score, feedback = 0, f"scoring failed: {e}"
        candidates.append({"paper": paper, "score": score, "feedback": feedback, "past_context": past_context})

    best = max(candidates, key=lambda c: c["score"])
    return best["paper"], best["past_context"], candidates


def _pdf_escape(text: str) -> str:
    """Escape <, >, & so reportlab Paragraph doesn't eat unknown HTML tags
    like <canvas>, <br>, <img>, etc. in question text."""
    import html as _h
    return _h.escape(text, quote=False)


def paper_to_pdf_bytes(content: str, subject: str) -> bytes:
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "title_style",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=16,
        spaceAfter=20
    )
    q_style = ParagraphStyle("q_style", parent=styles["Normal"], leading=18)
    sec_style = ParagraphStyle("sec_style", parent=styles["Heading2"],
                               alignment=TA_CENTER, spaceBefore=12, spaceAfter=12)
    instr_style = ParagraphStyle("instr_style", parent=styles["Normal"],
                                 alignment=0, spaceBefore=6, spaceAfter=6)

    story = []
    pdf_title = f"Predicted Questions for {_pdf_escape(subject)}"
    story.append(Paragraph(pdf_title, title_style))
    story.append(Spacer(1, 24))

    lines = content.splitlines()
    for line in lines:
        s = line.strip()
        if not s:
            continue
        esc = _pdf_escape(s)
        if re.match(r"^(Section\s+[A-Z])", s, flags=re.I):
            story.append(Paragraph(esc, sec_style))
            story.append(Spacer(1, 12))
        elif re.match(r"(GROUP\s+[A-Z])", s, flags=re.I):
            story.append(Spacer(1, 18))
            story.append(Paragraph(esc, sec_style))
            story.append(Spacer(1, 6))
        elif re.match(r"^Answer any", s, flags=re.I):
            story.append(Paragraph(esc, instr_style))
            story.append(Spacer(1, 6))
        elif re.match(r"^\d+\.", s):
            q_text = re.sub(r"^(\d+\.)", r"<b>\1</b>", esc)
            story.append(Paragraph(q_text, q_style))
            story.append(Spacer(1, 12))
        else:
            story.append(Paragraph(esc, q_style))

    doc.build(story)
    return buf.getvalue()
