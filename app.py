import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os, re, html
from io import BytesIO

from utils import extract_text_from_pdfs, chunk_text, create_vector_store, load_vector_store, generate_predicted_paper_best_of_n
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
        book_files = st.file_uploader("Upload Reference Book(s) PDF (optional)", type=["pdf"], accept_multiple_files=True, key="book_files")

        if st.button("Process Files"):
            if syllabus_file and past_files:
                st.session_state.syllabus_text = extract_text_from_pdfs([syllabus_file])
                st.session_state.subject = os.path.splitext(syllabus_file.name)[0]

                past_text = extract_text_from_pdfs(past_files)
                past_text = strip_headers_and_footers(past_text)
                past_text = clean_body_keep_all_marks(past_text)
                past_chunks = chunk_text(past_text)

                book_chunks = None
                st.session_state.has_book = False
                if book_files:
                    book_text = extract_text_from_pdfs(book_files)
                    if book_text and len(book_text.strip()) > 100:
                        book_chunks = chunk_text(book_text)
                        st.session_state.has_book = True

                create_vector_store(past_chunks, book_chunks=book_chunks)
                st.session_state.ready = True
                msg = "Syllabus + past questions processed (marks preserved)."
                if book_chunks:
                    msg += f" Book indexed ({len(book_chunks)} chunks) — questions will be grounded in book content."
                st.success(msg)
            else:
                st.error("Upload syllabus + past papers first.")

    if st.button("Generate Predicted Paper", disabled=not st.session_state.get("ready", False)):
        with st.spinner("Generating 3 candidates and picking the best..."):
            past_db = load_vector_store()
            raw_paper, _, candidates = generate_predicted_paper_best_of_n(
                past_db,
                st.session_state.syllabus_text,
                has_book=st.session_state.get("has_book", False),
                n=3,
            )
            scores = [c["score"] for c in candidates]
            best = max(scores)
            st.info(f"Candidate scores: {scores} — picked best ({best}/100).")
            for i, c in enumerate(candidates):
                if c["feedback"] and c["feedback"].lower() != "none":
                    st.caption(f"Candidate {i+1} ({c['score']}/100): {c['feedback']}")

            paper = strip_headers_and_footers(raw_paper)
            paper = clean_body_keep_all_marks(paper)
            paper = enforce_structure(paper)
            paper = split_group_instructions(paper)
            paper = enforce_pom1_style(paper)
            paper = center_sections(paper)

            html_lines = []
            subject = st.session_state.get("subject", "Subject")
            html_lines.append(f"<h3 style='text-align:center;'>Predicted Questions for {html.escape(subject)}</h3><br>")

            for line in paper.splitlines():
                s = line.strip()
                esc = html.escape(s)
                if re.match(r"^(Section\s+[A-Z])", s, flags=re.I):
                    html_lines.append(f"<div style='text-align:center; margin-top:20px;'><b>{esc}</b></div>")
                elif re.match(r"(GROUP\s+[A-Z])", s, flags=re.I):
                    html_lines.append(f"<div style='text-align:center; margin-top:20px;'><b>{esc}</b></div>")
                elif re.match(r"^Answer any", s, flags=re.I):
                    html_lines.append(f"<p style='margin-bottom:12px;'><i>{esc}</i></p>")
                elif re.match(r"^\d+\.", s):
                    q_text = re.sub(r"^(\d+\.)", r"<b>\1</b>", esc)
                    html_lines.append(f"<p style='margin-bottom:18px;'>{q_text}</p>")
                else:
                    html_lines.append(f"<p style='margin-bottom:12px;'>{esc}</p>")

            styled_output = "\n".join(html_lines)
            st.markdown(styled_output, unsafe_allow_html=True)

            if REPORTLAB_OK:
                pdf_bytes = paper_to_pdf_bytes(paper, subject)
                st.download_button("⬇️ Download PDF", pdf_bytes, "predicted_paper.pdf", "application/pdf")

if __name__ == "__main__":
    main()
