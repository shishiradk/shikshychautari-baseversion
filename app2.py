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
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
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
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.3, max_tokens=1400)
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
    text = re.sub(r"(?<!^)(\s*)(\d+\.)", r"\n\2", text)  # each question new line
    return text

def center_sections(text: str) -> str:
    lines = text.splitlines()
    out = []
    for line in lines:
        if re.match(r"^(Section\s+[A-Z])", line.strip(), flags=re.I):
            out.append(line.strip().center(80))
        else:
            out.append(line.strip())
    return "\n".join(out)

# -------- Streamlit App --------
def main():
    st.set_page_config("📘 Question Paper Generator", layout="wide")
    st.title("📘 Structured Question Paper Generator (Marks Preserved, No Header)")

    with st.sidebar:
        st.header("Upload Files")
        syllabus_file = st.file_uploader("Upload Syllabus PDF", type=["pdf"], key="syllabus")
        past_files = st.file_uploader("Upload Past Questions PDF(s)", type=["pdf"], accept_multiple_files=True, key="past_questions")

        if st.button("Process Files"):
            if syllabus_file and past_files:
                st.session_state.syllabus_text = extract_text_from_pdfs([syllabus_file])
                past_text = extract_text_from_pdfs(past_files)
                # clean before sending to FAISS
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
            paper = center_sections(paper)

            st.subheader("Predicted Paper (Structured, Marks Preserved)")
            st.text_area("Output", paper, height=600)

            if REPORTLAB_OK:
                pdf_bytes = paper_to_pdf_bytes("Predicted Paper", paper)
                st.download_button("⬇️ Download PDF", pdf_bytes, "predicted_paper.pdf", "application/pdf")

def paper_to_pdf_bytes(title: str, content: str) -> bytes:
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, title=title)
    styles = getSampleStyleSheet()
    story = [Paragraph(content.replace("\n", "<br/>"), styles["BodyText"])]
    doc.build(story)
    return buf.getvalue()

if __name__ == "__main__":
    main()
