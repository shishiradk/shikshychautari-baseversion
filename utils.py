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

def create_vector_store(chunks, db_path="faiss_index"):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
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


def generate_predicted_paper(past_db, syllabus_text):
    docs = past_db.similarity_search(
        "exam structure sections numbering marks distribution typical topics", k=12
    )
    past_context = "\n".join([doc.page_content for doc in docs])
    chain = get_question_generator_chain()
    return chain.invoke({
        "past_questions": past_context,
        "syllabus": syllabus_text
    }), past_context





def get_question_generator_chain():
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
    

--- MARKS RULES ---
- Dynamically calculate all **section totals** using multiplication.
- Preserve all sub-question marks as they appear in the past paper.
- Format section totals as `[X x Y = Z]`.
- Do NOT output incorrect formats like `[85 = 40]`.

--- OUTPUT FORMAT ---
- Only output:
  1. Section titles (centered)
  2. Section instructions with dynamically calculated marks
  3. Questions with numbering, sub-questions, and marks
- No extra text, no explanations, no headers or footers.

--- INPUT VARIABLES ---
- {past_questions}: Past paper questions to replicate style and formatting
- {syllabus}: Syllabus topics to generate fresh questions

Output:
A single, fully formatted exam paper following all rules above.

"""
    prompt = PromptTemplate(
        input_variables=["past_questions", "syllabus"],
        template=template,
    )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, max_tokens=1400)
    output_parser = StrOutputParser()
    return prompt | llm | output_parser




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
    pdf_title = f"Predicted Questions for {subject}"
    story.append(Paragraph(pdf_title, title_style))
    story.append(Spacer(1, 24))

    lines = content.splitlines()
    for line in lines:
        if not line.strip():
            continue
        if re.match(r"^(Section\s+[A-Z])", line.strip(), flags=re.I):
            story.append(Paragraph(line.strip(), sec_style))
            story.append(Spacer(1, 12))
        elif re.match(r"(GROUP\s+[A-Z])", line.strip(), flags=re.I):
            story.append(Spacer(1, 18))
            story.append(Paragraph(line.strip(), sec_style))
            story.append(Spacer(1, 6))
        elif re.match(r"^Answer any", line.strip(), flags=re.I):
            story.append(Paragraph(line.strip(), instr_style))
            story.append(Spacer(1, 6))
        elif re.match(r"^\d+\.", line.strip()):
            q_text = re.sub(r"^(\d+\.)", r"<b>\1</b>", line.strip())
            story.append(Paragraph(q_text, q_style))
            story.append(Spacer(1, 12))
        else:
            story.append(Paragraph(line.strip(), q_style))

    doc.build(story)
    return buf.getvalue()
