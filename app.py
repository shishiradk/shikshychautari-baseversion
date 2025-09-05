import streamlit as st
from utils import (
    extract_text_from_pdfs, 
    chunk_text, 
    create_vector_store, 
    load_vector_store, 
    generate_predicted_paper, 
    render_question_paper, 
    paper_to_pdf_bytes, 
    REPORTLAB_OK
)

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
                    render_question_paper(paper, st)

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