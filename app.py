import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from dotenv import load_dotenv
import os
import re

load_dotenv()

# LangSmith Tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Question Paper Generator"

# Extract text from uploaded PDFs
def extract_text_from_pdfs(pdfs):
    full_text = ""
    for pdf in pdfs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text
    return full_text

# Split text into manageable chunks
def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    return splitter.split_text(text)

# Create FAISS vector store
def create_vector_store(chunks, db_path="faiss_index"):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    vectorstore.save_local(db_path)

# Load vector store
def load_vector_store(db_path="faiss_index"):
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)

# Build the LLM chain using PromptTemplate + Runnable syntax (LangSmith compatible)
def get_question_generator_chain():
    template = """
You are a smart question paper generator.

Given the following:
- Past question context
- Syllabus content
- Difficulty level: {difficulty}

Your task is to generate a future question paper with the **same number and format of questions as in the past questions**.

Prediction logic by difficulty level:
- Easy: 80% questions similar to past papers, 20% new based on syllabus
- Hard: 60% similar, 40% new

Guidelines:
- Reuse the structure, numbering, and format of past questions
- Follow academic tone, vary question types, and avoid repetition
- Ensure new questions come from syllabus topics not seen in past questions
- Maintain clarity and logical flow throughout

--- PAST QUESTIONS CONTEXT ---
{past_questions}

--- SYLLABUS CONTENT ---
{syllabus}

Now generate the predicted question paper according to the given difficulty level.
"""
    prompt = PromptTemplate(
        input_variables=["past_questions", "syllabus", "difficulty"],
        template=template,
    )
    llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0.5, max_tokens=1024)
    output_parser = StrOutputParser()

    return prompt | llm | output_parser  # Runnable chain (LangSmith compatible)

# Generate question papers
def generate_question_papers(past_db, syllabus_text):
    chain = get_question_generator_chain()
    papers = []
    for level in ["Easy", "Hard"]:
        docs = past_db.similarity_search("generate exam questions", k=10)
        past_context = "\n".join([doc.page_content for doc in docs])
        response = chain.invoke({
            "past_questions": past_context,
            "syllabus": syllabus_text,
            "difficulty": level
        })
        papers.append((level, response))
    return papers

# Render output text and formatted code
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

# Main UI
def main():
    st.set_page_config("📘 Question Paper Generator", layout="wide")

    st.markdown("""
        <style>
            .centered-title { text-align: center; }
            .centered-button button { display: block; margin: 0 auto; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 class='centered-title'> Predict Question Papers from Syllabus + Past Papers</h1>", unsafe_allow_html=True)

    with st.sidebar:
        st.header("📘 Upload and Process Syllabus")
        syllabus_file = st.file_uploader("Upload Syllabus PDF", type=["pdf"], key="syllabus")
        if st.button(" Submit Syllabus"):
            if syllabus_file:
                st.session_state.syllabus_text = extract_text_from_pdfs([syllabus_file])
                st.success("Syllabus processed successfully!")
            else:
                st.error("Please upload a syllabus PDF.")

        st.header("📄 Upload and Process Past Questions")
        past_files = st.file_uploader("Upload Past Questions PDF(s)", type=["pdf"], accept_multiple_files=True, key="past_questions")
        if st.button(" Submit Past Questions"):
            if past_files:
                past_text = extract_text_from_pdfs(past_files)
                past_chunks = chunk_text(past_text)
                create_vector_store(past_chunks)
                st.session_state.past_questions_processed = True
                st.success("Past questions processed and stored!")
            else:
                st.error("Please upload past question PDF(s).")

    st.markdown("---")
    st.markdown("<h2 class='centered-title'>📄 Predicted Question Papers</h2>", unsafe_allow_html=True)

    ready = st.session_state.get("syllabus_text") and st.session_state.get("past_questions_processed")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        generate_clicked = st.button("📚 Generate Question Papers", disabled=not ready)

    if generate_clicked:
        if not ready:
            st.warning("Upload and process both syllabus and past questions first.")
        else:
            with st.spinner("Generating question papers..."):
                past_db = load_vector_store()
                papers = generate_question_papers(past_db, st.session_state.syllabus_text)
                st.success("Generated 2 Question Papers!")

                for level, paper in papers:
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        with st.expander(f"📄 {level} Paper"):
                            render_question_paper(paper)

if __name__ == "__main__":
    main()
