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

from utils import extract_text_from_pdfs, chunk_text, create_vector_store, load_vector_store, generate_predicted_paper
from utils import strip_headers_and_footers, clean_body_keep_all_marks, enforce_structure, split_group_instructions, enforce_pom1_style, center_sections
from utils import paper_to_pdf_bytes
from utils import REPORTLAB_OK

# Optional PDF generation


load_dotenv()

 

# -------- Streamlit App --------
def main():
    st.set_page_config("📘 Question Paper Generator", layout="wide")
    st.title("📘 Structured Question Paper Generator ")

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
                try:
                    pdf_bytes = paper_to_pdf_bytes(paper, subject)
                    st.download_button(
                        "⬇️ Download PDF", 
                        pdf_bytes, 
                        f"predicted_paper_{subject.replace(' ', '_')}.pdf", 
                        "application/pdf",
                        help="Click to download the generated question paper as a PDF file"
                    )
                except Exception as e:
                    st.error(f"Error generating PDF: {str(e)}")
            else:
                st.warning("PDF generation is not available. Please install reportlab: pip install reportlab")

if __name__ == "__main__":
    main()
