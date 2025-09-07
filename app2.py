import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os, re
from io import BytesIO

# Optional PDF generation
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.enums import TA_CENTER
    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False

load_dotenv()

# -------- Utilities --------
def extract_text_from_pdfs(pdfs):
    full_text = ""
    for pdf in pdfs:
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

def get_question_generator_chain():
    template = """
You are an expert academic exam paper predictor.

Generate exactly ONE complete future exam paper.

Requirements:
1. Use the same structure as past papers (sections, numbering, marks distribution).
2. Do not copy verbatim but follow topic/phrasing patterns.
3. Output should contain ONLY Section titles + instructions + questions.
4. Exclude all headers/footers (university, course name, marks table, time).
5. Keep section-wide marks (e.g., [2 x 10 = 20]) and per-question marks ([4+6], [5], [2+3]).
6. Each section must start on a new line. Each question must be numbered clearly.
7. Center section headings.
8. Use 'Answer any ...' instructions as in past papers but not with Group or Section but in another line .

--- PAST QUESTIONS CONTEXT ---
{past_questions}

--- SYLLABUS CONTENT ---
{syllabus}

Output strictly the structured question paper body.
"""
    prompt = PromptTemplate(
        input_variables=["past_questions", "syllabus"],
        template=template,
    )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, max_tokens=1400)
    output_parser = StrOutputParser()
    return prompt | llm | output_parser

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

# -------- PDF Generator with dynamic title --------
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

# -------- Streamlit App --------
def main():
    st.set_page_config("📘 Question Paper Generator", layout="wide")
    st.title("📘 Structured Question Paper Generator (Marks Preserved, POM1 Style)")

    with st.sidebar:
        st.header("Upload Files")
        syllabus_file = st.file_uploader("Upload Syllabus PDF", type=["pdf"], key="syllabus")
        past_files = st.file_uploader("Upload Past Questions PDF(s)", type=["pdf"], accept_multiple_files=True, key="past_questions")

        if st.button("Process Files"):
            if syllabus_file and past_files:
                st.session_state.syllabus_text = extract_text_from_pdfs([syllabus_file])
                st.session_state.subject = os.path.splitext(syllabus_file.name)[0]

                past_text = extract_text_from_pdfs(past_files)
                past_text = strip_headers_and_footers(past_text)
                past_text = clean_body_keep_all_marks(past_text)
                past_chunks = chunk_text(past_text)
                create_vector_store(past_chunks)
                st.session_state.ready = True
                st.success("Syllabus + past questions processed (marks preserved).")
            else:
                st.error("Upload syllabus + past papers first.")

    if st.button("Generate Predicted Paper", disabled=not st.session_state.get("ready", False)):
        with st.spinner("Generating..."):
            past_db = load_vector_store()
            raw_paper, _ = generate_predicted_paper(past_db, st.session_state.syllabus_text)

            paper = strip_headers_and_footers(raw_paper)
            paper = clean_body_keep_all_marks(paper)
            paper = enforce_structure(paper)
            paper = split_group_instructions(paper)
            paper = enforce_pom1_style(paper)
            paper = center_sections(paper)

            st.subheader("Predicted Paper (Structured, Marks Preserved, POM1 Style)")

            html_lines = []
            subject = st.session_state.get("subject", "Subject")
            html_lines.append(f"<h3 style='text-align:center;'>Predicted Questions for {subject}</h3><br>")

            for line in paper.splitlines():
                if re.match(r"^(Section\s+[A-Z])", line.strip(), flags=re.I):
                    html_lines.append(f"<div style='text-align:center; margin-top:20px;'><b>{line.strip()}</b></div>")
                elif re.match(r"(GROUP\s+[A-Z])", line.strip(), flags=re.I):
                    html_lines.append(f"<div style='text-align:center; margin-top:20px;'><b>{line.strip()}</b></div>")
                elif re.match(r"^Answer any", line.strip(), flags=re.I):
                    html_lines.append(f"<p style='margin-bottom:12px;'><i>{line.strip()}</i></p>")
                elif re.match(r"^\d+\.", line.strip()):
                    q_text = re.sub(r"^(\d+\.)", r"<b>\1</b>", line.strip())
                    html_lines.append(f"<p style='margin-bottom:18px;'>{q_text}</p>")
                else:
                    html_lines.append(f"<p style='margin-bottom:12px;'>{line.strip()}</p>")

            styled_output = "\n".join(html_lines)
            st.markdown(styled_output, unsafe_allow_html=True)

            if REPORTLAB_OK:
                pdf_bytes = paper_to_pdf_bytes(paper, subject)
                st.download_button("⬇️ Download PDF", pdf_bytes, "predicted_paper.pdf", "application/pdf")

if __name__ == "__main__":
    main()
