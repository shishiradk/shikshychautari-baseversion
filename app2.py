import streamlit as st
import os
import re
from io import BytesIO
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# Optional PDF generation
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.enums import TA_CENTER
    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False

# ---------- Load keys ----------
load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Question Paper Generator"


# ---------- Text Extraction ----------
def extract_text(pdf_file):
    text = ""
    reader = PdfReader(pdf_file)
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + "\n"
    return text


# ---------- Exam Structure Recognition ----------
def recognize_exam_structure(pdf_file):
    text = extract_text(pdf_file)

    # ---------- HEADER ----------
    header_match = re.search(r"^(.*?)(?=Section\s+[A-Z]|Group\s+[A-Z])", text, re.S | re.I)
    header = header_match.group(0).strip() if header_match else "\n".join(text.splitlines()[:10])

    # ---------- SECTIONS ----------
    sections = []
    section_pattern = re.compile(r"((Section|Group)\s+[A-Z].*?)(?=(Section|Group)\s+[A-Z]|$)", re.S | re.I)
    matches = section_pattern.findall(text)

    for sec in matches:
        block = sec[0].strip()
        lines = block.splitlines()

        # Title = first line
        title = lines[0].strip() if lines else "Section"

        # Instruction = lines before first numbered question
        instruction = ""
        questions = []
        for i, line in enumerate(lines[1:], 1):
            if re.match(r"^\d+\.", line.strip()):
                question_block = "\n".join(lines[i:])
                questions = re.findall(r"(\d+\..*?)(?=\n\d+\.|\Z)", question_block, re.S)
                break
            elif line.strip():
                instruction += (" " + line.strip()) if instruction else line.strip()

        sections.append({
            "title": title,
            "instruction": instruction,
            "questions": [q.strip() for q in questions]
        })

    # ---------- FOOTER ----------
    footer_match = re.search(r"(Time:.*|Best of Luck.*|Figures.*marks.*)$", text, re.I | re.M)
    footer = footer_match.group(0).strip() if footer_match else ""

    return {"header": header, "sections": sections, "footer": footer}


# ---------- Header/Footer Builder ----------
def build_header(structure_json):
    raw_header = structure_json.get("header", "")

    # Flexible field extraction
    university = re.search(r"([A-Za-z ]+University)", raw_header, re.I)
    faculty = re.search(r"(Institute of .*?|Faculty of .*?)", raw_header, re.I)
    level_year_semester = re.search(r"(Bachelor.*Year.*Semester|Bachelor Level.*)", raw_header, re.I)
    program = re.search(r"(Science.*Technology|Computer Science.*|BSc.*)", raw_header, re.I)
    paper_type = re.search(r"(Model Question|Final Exam|Exam Paper)", raw_header, re.I)

    course_title = re.search(r"Course Title:\s*(.+)", raw_header, re.I)
    course_code = re.search(r"Course Code:\s*([A-Z0-9]+)", raw_header, re.I)
    time_val = re.search(r"Time:\s*([0-9]+.*hours)", raw_header, re.I)
    full_marks = re.search(r"Full Marks:\s*([0-9]+)", raw_header, re.I)
    pass_marks = re.search(r"Pass Marks:\s*([0-9]+)", raw_header, re.I)

    # Form-style header
    header = f"""{university.group(1) if university else ""}
{faculty.group(1) if faculty else ""}

{level_year_semester.group(1) if level_year_semester else ""}
{program.group(1) if program else ""}

{paper_type.group(1) if paper_type else "Exam Paper"}
Course Title: {course_title.group(1).strip() if course_title else ""}
Course Code: {course_code.group(1).strip() if course_code else ""}        Time: {time_val.group(1).strip() if time_val else ""}
Full Marks: {full_marks.group(1).strip() if full_marks else ""}          Pass Marks: {pass_marks.group(1).strip() if pass_marks else ""}
"""
    return header.strip()


def build_footer(structure_json):
    raw_footer = structure_json.get("footer", "")
    if not raw_footer:
        return ""
    if "best of luck" in raw_footer.lower():
        return "Best of Luck"
    return raw_footer.strip()


# ---------- Vector Store ----------
def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return splitter.split_text(text)


def create_vector_store_from_docs(past_text, syllabus_text, db_path="faiss_index"):
    embeddings = OpenAIEmbeddings()
    past_chunks = chunk_text(past_text)
    syllabus_chunks = chunk_text(syllabus_text)
    tagged_chunks = [f"[PAST] {c}" for c in past_chunks] + [f"[SYLLABUS] {c}" for c in syllabus_chunks]
    vectorstore = FAISS.from_texts(tagged_chunks, embedding=embeddings)
    vectorstore.save_local(db_path)


def load_vector_store(db_path="faiss_index"):
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)


# ---------- LLM Chain ----------
def get_question_generator_chain():
    template = """
You are an expert academic exam paper predictor.

Rules:
- Keep HEADER, SECTION TITLES, and INSTRUCTIONS exactly as provided.
- Do not rewrite numbering, section labels, or marks distribution.
- Replace ONLY the text of the QUESTIONS with new ones.
- Ensure questions strictly come from the SYLLABUS.
- Keep formatting identical to input.
- Do not duplicate the footer.

{header}

{sections}

{footer}

Reference context (syllabus + past papers):
{context}

Output only the formatted exam paper.
"""
    prompt = PromptTemplate(input_variables=["header", "sections", "footer", "context"], template=template)
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2, max_tokens=3000)
    return prompt | llm | StrOutputParser()


def generate_predicted_paper(past_db, structure_json):
    docs = past_db.similarity_search("exam questions format marks distribution", k=20)
    context = "\n".join([doc.page_content for doc in docs])
    chain = get_question_generator_chain()

    sections_text = ""
    for sec in structure_json["sections"]:
        sections_text += sec["title"] + "\n"
        if sec.get("instruction"):
            sections_text += sec["instruction"] + "\n"
        for q in sec["questions"]:
            sections_text += q + "\n"
        sections_text += "\n"

    return chain.invoke({
        "header": build_header(structure_json),
        "sections": sections_text.strip(),
        "footer": build_footer(structure_json),
        "context": context
    })


# ---------- PDF export ----------
def paper_to_pdf_bytes(title: str, structure_json: dict, body_text: str) -> bytes:
    if not REPORTLAB_OK:
        raise RuntimeError("ReportLab not installed. Run: pip install reportlab")

    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4, title=title,
        topMargin=72, bottomMargin=72, leftMargin=72, rightMargin=72
    )
    styles = getSampleStyleSheet()

    header_style = ParagraphStyle("HeaderStyle", parent=styles["Normal"], alignment=TA_CENTER,
                                  fontSize=11, spaceAfter=18, fontName="Helvetica-Bold")

    section_style = ParagraphStyle("SectionStyle", parent=styles["Normal"], fontSize=11,
                                   spaceAfter=10, fontName="Helvetica-Bold")

    instruction_style = ParagraphStyle("InstructionStyle", parent=styles["Normal"],
                                       fontSize=10, fontName="Helvetica-Oblique", spaceAfter=8)

    question_style = ParagraphStyle("QuestionStyle", parent=styles["Normal"], fontSize=10, spaceAfter=6)

    footer_style = ParagraphStyle("FooterStyle", parent=styles["Normal"], alignment=TA_CENTER,
                                  fontSize=9, spaceBefore=20)

    story = []
    if structure_json.get("header"):
        story.append(Paragraph(build_header(structure_json).replace("\n", "<br/>"), header_style))
        story.append(Spacer(1, 15))

    # Body (sections)
    for sec in structure_json.get("sections", []):
        story.append(Paragraph(sec["title"], section_style))
        if sec.get("instruction"):
            story.append(Paragraph(sec["instruction"], instruction_style))
        for q in sec.get("questions", []):
            story.append(Paragraph(q, question_style))
        story.append(Spacer(1, 10))

    # Footer
    footer_text = build_footer(structure_json)
    if footer_text:
        story.append(Paragraph(footer_text, footer_style))

    doc.build(story)
    pdf_data = buf.getvalue()
    buf.close()
    return pdf_data


# ---------- Streamlit App ----------
def main():
    st.set_page_config(page_title="📘 Question Paper Predictor", layout="wide")
    st.title("📘 Question Paper Predictor (Format-Aware, Syllabus-Aligned)")

    with st.sidebar:
        st.header("Upload Files")
        syllabus_file = st.file_uploader("Upload Syllabus PDF", type=["pdf"])
        past_files = st.file_uploader("Upload Past Question PDFs", type=["pdf"], accept_multiple_files=True)

        if st.button("Process Inputs"):
            if not syllabus_file or not past_files:
                st.error("Upload both syllabus and past question papers.")
            else:
                syllabus_text = extract_text(syllabus_file)
                past_text = ""
                first_structure = None
                for idx, f in enumerate(past_files):
                    txt = extract_text(f)
                    past_text += txt
                    if idx == 0:
                        first_structure = recognize_exam_structure(f)

                create_vector_store_from_docs(past_text, syllabus_text)
                st.session_state.structure_json = first_structure
                st.success("✅ Files processed successfully. Syllabus + Past Papers stored in FAISS.")

    if st.button("📚 Generate Predicted Paper"):
        if "structure_json" not in st.session_state:
            st.warning("Please process syllabus and past questions first.")
        else:
            with st.spinner("Generating predicted exam paper..."):
                past_db = load_vector_store()
                paper = generate_predicted_paper(past_db, st.session_state.structure_json)
                st.subheader("📄 Predicted Exam Paper")
                st.text_area("Generated Paper", paper, height=500)

                if REPORTLAB_OK:
                    try:
                        pdf_bytes = paper_to_pdf_bytes("Predicted Question Paper", st.session_state.structure_json, paper)
                        st.download_button("⬇️ Download as PDF", data=pdf_bytes,
                                           file_name="predicted_paper.pdf", mime="application/pdf")
                    except Exception as e:
                        st.error(f"PDF generation failed: {e}")
                else:
                    st.info("Install ReportLab to enable PDF download: `pip install reportlab`")


if __name__ == "__main__":
    main()
