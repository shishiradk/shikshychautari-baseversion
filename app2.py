import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
import re
from io import BytesIO

# Optional PDF generation (reportlab)
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False

load_dotenv()

# LangSmith Tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Question Paper Generator"

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

Generate exactly **one** complete future exam paper.

Requirements:
1) Mirror the sections, numbering, and marks distribution seen in past papers.
2) Maximize recurrence likelihood: pick topics and phrasings consistent with past patterns without copying verbatim.
3) Fill all sections fully. Produce one paper only.
4) Include high-importance syllabus areas that were underrepresented historically to keep the paper realistic.
5) Maintain academic tone, clarity, and logical flow.
6) Do NOT include any header (university, year, marks, instructions). Leave the header blank.

--- PAST QUESTION CONTEXT ---
{past_questions}

--- SYLLABUS CONTENT ---
{syllabus}

Output only the exam questions with section titles, instructions, and numbering.
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

def strip_exam_header(text: str) -> str:
    lines = text.splitlines()
    clean = []
    for line in lines:
        if any(kw in line.lower() for kw in [
            "tribhuvan university", "institute of science",
            "full marks", "pass marks", "time", "candidates are required"
        ]):
            continue
        clean.append(line)
    return "\n".join(clean).strip()

def clean_question_body(text: str) -> str:
    text = re.sub(r"\*+", "", text)           # remove asterisks
    text = re.sub(r"_+", "", text)            # remove underscores
    text = re.sub(r"`+", "", text)            # remove backticks
    text = re.sub(r"#+", "", text)            # remove markdown headings
    text = re.sub(r"\.\.\.+", "", text)       # remove ellipses (...)
    text = re.sub(r"\s{2,}", " ", text)       # collapse multiple spaces
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)  # collapse big blank areas
    return text.strip()

def force_section_breaks(text: str) -> str:
    """Ensure Section or Part headings start on new lines."""
    return re.sub(r"(Section\s+[A-Z])", r"\n\1", text, flags=re.I)

def force_newlines(text: str) -> str:
    """Ensure every numbered question starts on its own line."""
    return re.sub(r"(?<!^)(\s*)(\d+\.)", r"\n\2", text)

def detect_numbering_style(past_text: str) -> str:
    sections = re.split(r"(Section\s+[A-Z])", past_text, flags=re.I)
    restart_counts = 0
    total_sections = 0
    for sec in sections:
        q_nums = re.findall(r"(\d+)\.", sec)
        if q_nums:
            total_sections += 1
            if q_nums[0] == "1":
                restart_counts += 1
    if total_sections > 1 and restart_counts == total_sections:
        return "section"
    return "global"

def format_questions(text: str, style: str = "global") -> str:
    lines = text.splitlines()
    out = []
    q_counter = 1
    section_q = 1
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith("section") or line.lower().startswith("part"):
            out.append("\n" + line)
            if style == "section":
                section_q = 1
            continue
        if re.match(r"^\d+\.", line):
            if style == "section":
                out.append(f"{section_q}. " + re.sub(r"^\d+\.\s*", "", line))
                section_q += 1
            else:
                out.append(f"{q_counter}. " + re.sub(r"^\d+\.\s*", "", line))
                q_counter += 1
        else:
            out.append(line)
    return "\n".join(out).strip()

def render_question_paper(raw_text):
    st.text_area("Predicted Paper (Structured)", raw_text, height=600)

def paper_to_pdf_bytes(title: str, content: str) -> bytes:
    if not REPORTLAB_OK:
        raise RuntimeError("ReportLab not installed. Run: pip install reportlab")
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, title=title)
    styles = getSampleStyleSheet()
    story = [Paragraph(title, styles["Title"]), Spacer(1, 12)]
    story.append(Paragraph(content.replace("\n", "<br/>"), styles["BodyText"]))
    doc.build(story)
    pdf_data = buf.getvalue()
    buf.close()
    return pdf_data

# -------- Streamlit App --------
def main():
    st.set_page_config("📘 Question Paper Generator", layout="wide")

    st.markdown("""
        <style>
            .centered-title { text-align: center; }
            .centered-button button { display: block; margin: 0 auto; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 class='centered-title'> Predict One High-Density Question Paper</h1>", unsafe_allow_html=True)

    with st.sidebar:
        st.header("📘 Upload and Process Syllabus")
        syllabus_file = st.file_uploader("Upload Syllabus PDF", type=["pdf"], key="syllabus")
        if st.button("Submit Syllabus"):
            if syllabus_file:
                st.session_state.syllabus_text = extract_text_from_pdfs([syllabus_file])
                st.success("Syllabus processed successfully.")
            else:
                st.error("Please upload a syllabus PDF.")

        st.header("📄 Upload and Process Past Questions")
        past_files = st.file_uploader("Upload Past Questions PDF(s)", type=["pdf"], accept_multiple_files=True, key="past_questions")
        if st.button("Submit Past Questions"):
            if past_files:
                past_text = extract_text_from_pdfs(past_files)
                past_chunks = chunk_text(past_text)
                create_vector_store(past_chunks)
                st.session_state.past_questions_processed = True
                st.success("Past questions processed and stored.")
            else:
                st.error("Please upload past question PDF(s).")

    st.markdown("---")
    st.markdown("<h2 class='centered-title'>📄 Predicted Paper</h2>", unsafe_allow_html=True)

    ready = bool(st.session_state.get("syllabus_text")) and bool(st.session_state.get("past_questions_processed"))

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        generate_clicked = st.button("📚 Generate Predicted Paper", disabled=not ready)

    if generate_clicked:
        if not ready:
            st.warning("Upload and process both syllabus and past questions first.")
        else:
            with st.spinner("Generating predicted paper..."):
                past_db = load_vector_store()
                paper, past_context = generate_predicted_paper(past_db, st.session_state.syllabus_text)
                style = detect_numbering_style(past_context)

                paper = strip_exam_header(paper)
                paper = clean_question_body(paper)
                paper = force_section_breaks(paper)
                paper = force_newlines(paper)
                paper = format_questions(paper, style=style)

                st.success(f"Generated predicted paper with '{style}' numbering.")

                with st.expander("📄 Predicted Paper"):
                    render_question_paper(paper)

                    if not REPORTLAB_OK:
                        st.info("To enable PDF downloads, install ReportLab:  \n`pip install reportlab`")

                    if REPORTLAB_OK:
                        try:
                            pdf_bytes = paper_to_pdf_bytes("Predicted Question Paper", paper)
                            st.download_button(
                                label="⬇️ Download Paper as PDF",
                                data=pdf_bytes,
                                file_name="predicted_paper.pdf",
                                mime="application/pdf",
                                use_container_width=True,
                            )
                        except Exception as e:
                            st.error(f"PDF generation failed: {e}")

if __name__ == "__main__":
    main()
