import os
import re
from io import BytesIO
from typing import List
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Optional PDF generation (reportlab)
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# LangSmith Tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Question Paper Generator"

def extract_text_from_pdfs(pdfs: List[bytes]) -> str:
    """Extract text from multiple PDF files"""
    full_text = ""
    for pdf_bytes in pdfs:
        reader = PdfReader(BytesIO(pdf_bytes))
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text
    return full_text

def chunk_text(text: str) -> List[str]:
    """Split text into chunks for vector storage"""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    return splitter.split_text(text)

def create_vector_store(chunks: List[str], db_path: str = "faiss_index") -> str:
    """Create and save vector store"""
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    vectorstore.save_local(db_path)
    return db_path

def load_vector_store(db_path: str = "faiss_index") -> FAISS:
    """Load vector store"""
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)

def get_question_generator_chain():
    """Create the question generation chain"""
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

{additional_instructions}

Output the predicted exam paper only.
"""
    prompt = PromptTemplate(
        input_variables=["past_questions", "syllabus", "additional_instructions"],
        template=template,
    )
    llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0.3, max_tokens=1400)
    output_parser = StrOutputParser()
    return prompt | llm | output_parser

def generate_predicted_paper(past_db: FAISS, syllabus_text: str, additional_instructions: str = "") -> str:
    """Generate predicted question paper"""
    # Retrieve rich past context to drive structure + topic frequencies
    docs = past_db.similarity_search("exam structure sections numbering marks distribution typical topics", k=12)
    past_context = "\n".join([doc.page_content for doc in docs])
    
    chain = get_question_generator_chain()
    additional_prompt = f"\nAdditional Instructions: {additional_instructions}" if additional_instructions else ""
    
    return chain.invoke({
        "past_questions": past_context,
        "syllabus": syllabus_text,
        "additional_instructions": additional_prompt
    })

def paper_to_pdf_bytes(title: str, content: str) -> bytes:
    """Convert paper content to PDF bytes"""
    if not REPORTLAB_OK:
        raise RuntimeError("ReportLab not installed. Run: pip install reportlab")
    
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, title=title)
    styles = getSampleStyleSheet()
    story = [Paragraph(title, styles["Title"]), Spacer(1, 12)]
    
    # Simple text processing (can be enhanced)
    lines = content.split('\n')
    for line in lines:
        if line.strip():
            story.append(Paragraph(line.strip(), styles["BodyText"]))
            story.append(Spacer(1, 6))
    
    doc.build(story)
    pdf_data = buf.getvalue()
    buf.close()
    return pdf_data

def render_question_paper(raw_text: str, st):
    """Render question paper content in Streamlit"""
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
