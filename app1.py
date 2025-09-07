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

# Optional PDF generation (ReportLab). If missing, fallback to TXT.
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
Generate exactly **one** complete future exam paper with the highest probability of appearing questions, using the inputs below.

Requirements:
1) Mirror the sections, numbering, and marks distribution seen in past papers.
2) Maximize recurrence likelihood: pick topics and phrasings consistent with past patterns without copying verbatim.
3) Fill all sections fully. Produce one paper only.
4) Include high-importance syllabus areas that were underrepresented historically to keep the paper realistic.
5) Maintain academic tone, clarity, and logical flow.

--- PAST QUESTION CONTEXT ---
{past_questions}

--- SYLLABUS CONTENT ---
{syllabus}

Output the predicted exam paper only.
"""
    prompt = PromptTemplate(
        input_variables=["past_questions", "syllabus"],
        template=template,
    )
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.3, max_tokens=1400)
    output_parser = StrOutputParser()
    return prompt | llm | output_parser

def generate_predicted_paper(past_db, syllabus_text):
    # Retrieve rich past context to drive structure + topic frequencies
    docs = past_db.similarity_search("exam structure sections numbering marks distribution typical topics", k=12)
    past_context = "\n".join([doc.page_content for doc in docs])
    chain = get_question_generator_chain()
    return chain.invoke({
        "past_questions": past_context,
        "syllabus": syllabus_text
    })

def render_question_paper(raw_text):
    parts = re.split(r"```([a-zA-Z0-9]*)\n?", raw_text)
    i = 0
    while i < len(parts):
        if i == 0:
            st.write(parts[i].strip())
            i += 1
        else:
            language = parts[i].strip() or "text"
            code = parts[i + 1].strip() if (i + 1) < len(parts) else ""
            st.code(code, language=language)
            i += 2

def paper_to_pdf_bytes(title: str, content: str) -> bytes:
    if not REPORTLAB_OK:
        raise RuntimeError("ReportLab not installed. Run: pip install reportlab")
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, title=title)
    styles = getSampleStyleSheet()
    story = [Paragraph(title, styles["Title"]), Spacer(1, 12)]
    parts = re.split(r"```([a-zA-Z0-9]*)\n?", content)
    for idx, part in enumerate(parts):
        if idx == 0:
            text = part.strip()
            if text:
                story.append(Paragraph(text.replace("\n", "<br/>"), styles["BodyText"]))
                story.append(Spacer(1, 8))
        else:
            if idx % 2 == 1:
                continue
            code_text = part.strip()
            if code_text:
                story.append(Paragraph("<b>Block:</b>", styles["BodyText"]))
                story.append(Paragraph(code_text.replace("\n", "<br/>"), styles["Code"]))
                story.append(Spacer(1, 8))
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
                paper = generate_predicted_paper(past_db, st.session_state.syllabus_text)
                st.success("Generated predicted paper.")

                with st.expander("📄 Predicted Paper"):
                    render_question_paper(paper)

                    # Always show download button
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
                    else:
                        # Fallback TXT download
                        st.download_button(
                            label="⬇️ Download Paper as TXT",
                            data=paper.encode("utf-8"),
                            file_name="predicted_paper.txt",
                            mime="text/plain",
                            use_container_width=True,
                        )

if __name__ == "__main__":
    main()
