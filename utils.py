import os
import re
from io import BytesIO
from turtle import st
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
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.units import inch
    from reportlab.lib.colors import black, darkblue, darkgreen
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

def extract_text_from_pdfs(pdfs) -> str:
    """Extract text from multiple PDF files (handles both bytes and UploadedFile objects)"""
    full_text = ""
    for pdf in pdfs:
        # Handle both bytes and UploadedFile objects
        if hasattr(pdf, 'read'):  # UploadedFile object
            pdf_bytes = pdf.read()
        else:  # bytes object
            pdf_bytes = pdf
            
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
    """Create the question generation chain with dynamic structure adaptation"""
    template = """
You are an expert academic exam paper predictor for university-level examinations.
Generate exactly **one** complete future exam paper that EXACTLY replicates the structure, format, and style detected from past papers.

**STRUCTURE ANALYSIS FROM PAST PAPERS:**
{structure_analysis}

**CRITICAL DYNAMIC REPLICATION REQUIREMENTS:**
1) **EXACT QUESTION COUNT**: Generate the PRECISE number of questions for each section as detected from past papers - NO MORE, NO LESS
2) **COMPLETE STRUCTURE ANALYSIS**: The system has analyzed the past papers and detected the EXACT format below
3) **FLEXIBLE SECTION HANDLING**: Replicate ANY section type (Section/Part/Group/Unit) with ANY naming (A,B,C or I,II,III or 1,2,3)
4) **DYNAMIC MARKS DISTRIBUTION**: Handle ANY total marks (60, 80, 100, etc.) with ANY distribution per question
5) **ADAPTIVE QUESTION REQUIREMENTS**: Support ANY attempt pattern (all compulsory, any X out of Y, any X questions)
6) **MULTI-TYPE QUESTION SUPPORT**: Generate objective, subjective short, subjective long, or mixed questions as detected
7) **FLEXIBLE NUMBERING**: Use global numbering (1,2,3...) or section-wise (1,2,3 per section) as detected
8) **EXACT FORMAT REPLICATION**: Copy the PRECISE instruction wording, marks format, and layout from past papers

**CONTENT GENERATION RULES:**
1) Generate questions that follow the same difficulty level and style as past papers
2) Use topics from the syllabus that align with past paper patterns
3) Maintain the same question types (theoretical, numerical, practical, etc.) as past papers
4) Ensure questions are fresh but follow established patterns
5) Include appropriate sub-questions if that's the pattern in past papers

**QUALITY REQUIREMENTS:**
- Questions should be academically rigorous and exam-appropriate
- Content should cover important syllabus topics while following past patterns
- Language should match the formal academic tone of past papers
- Difficulty should progress appropriately within and across sections

--- PAST QUESTION CONTEXT ---
{past_questions}

--- SYLLABUS CONTENT ---
{syllabus}

{additional_instructions}

**CRITICAL OUTPUT REQUIREMENTS:**
- Generate the EXACT number of questions for each section as specified in the structure analysis
- If a section shows "3 questions", generate exactly 3 questions - not 2, not 4, but exactly 3
- If a section shows "Answer any 5 out of 8", generate exactly 8 questions in that section
- Maintain the exact section structure, numbering style, and marks distribution
- Use the same instruction format and wording as past papers

**OUTPUT FORMAT:** Generate the complete predicted exam paper that looks identical in structure to past papers but with new, relevant questions.
"""
    prompt = PromptTemplate(
        input_variables=["structure_analysis", "past_questions", "syllabus", "additional_instructions"],
        template=template,
    )
    llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0.1, max_tokens=3500)
    output_parser = StrOutputParser()
    return prompt | llm | output_parser

def generate_predicted_paper(past_db, syllabus_text, additional_instructions=""):
    """Generate predicted paper with dynamic structure analysis and replication"""
    # Get comprehensive context from past papers
    header_docs = past_db.similarity_search(
        "university institute full marks pass marks time exam header format structure", k=6
    )
    structure_docs = past_db.similarity_search(
        "section part group numbering marks distribution question format instructions answer any attempt", k=10
    )
    content_docs = past_db.similarity_search(
        "topics questions syllabus subject areas concepts problems numerical theoretical", k=12
    )
    
    # Combine different types of context
    all_docs = header_docs + structure_docs + content_docs
    past_context = "\n".join([doc.page_content for doc in all_docs])
    
    # Analyze the complete structure of past papers
    structure_analysis = analyze_paper_structure(past_context)
    
    # Extract subject name from combined context
    subject_name = extract_subject_name(syllabus_text, past_context)
    
    # Create detailed structure description for the AI
    structure_description = create_structure_description(structure_analysis)
    
    # Generate the paper with structure-aware prompt
    chain = get_question_generator_chain()
    predicted_paper = chain.invoke({
        "structure_analysis": structure_description,
        "past_questions": past_context,
        "syllabus": syllabus_text,
        "additional_instructions": additional_instructions
    })
    
    return predicted_paper, past_context, subject_name

def create_structure_description(structure_info: dict) -> str:
    """Create a comprehensive description of the paper structure for dynamic AI generation"""
    description = []
    
    # Header information
    description.append("EXAM PAPER STRUCTURE TO REPLICATE EXACTLY:")
    description.append("="*50)
    
    if structure_info.get('university'):
        description.append(f"University: {structure_info['university']}")
    if structure_info.get('institute'):
        description.append(f"Institute: {structure_info['institute']}")
    if structure_info.get('full_marks'):
        description.append(f"Full Marks: {structure_info['full_marks']}")
    if structure_info.get('pass_marks'):
        description.append(f"Pass Marks: {structure_info['pass_marks']}")
    if structure_info.get('time'):
        description.append(f"Time: {structure_info['time']}")
    
    description.append(f"\nPaper Format: {len(structure_info['sections'])} sections total")
    description.append(f"Question Numbering: {structure_info['numbering_style']} style")
    description.append(f"Total Questions Detected: {structure_info['total_questions']}")
    
    # Detailed section analysis with EXACT question count requirements
    if structure_info['sections']:
        description.append("\nSECTION-BY-SECTION STRUCTURE (GENERATE EXACT NUMBERS):")
        description.append("-"*50)
        
        for i, section in enumerate(structure_info['sections']):
            description.append(f"\n{section['type'].upper()} {section['id'].upper()}:")
            
            # CRITICAL: Exact question count requirements
            total_questions = section.get('total_available', section.get('total_questions', 0))
            if total_questions > 0:
                description.append(f"  *** GENERATE EXACTLY {total_questions} QUESTIONS ***")
                
                # Question attempt requirements
                if section.get('questions_to_attempt'):
                    if section['questions_to_attempt'] == 'all':
                        description.append(f"  - All {total_questions} questions are COMPULSORY")
                    else:
                        description.append(f"  - Answer ANY {section['questions_to_attempt']} out of {total_questions} questions")
                else:
                    description.append(f"  - Generate {total_questions} questions")
            
            # Marks information
            if section['marks_per_question']:
                unique_marks = list(set(section['marks_per_question']))
                if len(unique_marks) == 1:
                    description.append(f"  - Each question carries {unique_marks[0]} marks")
                else:
                    description.append(f"  - Variable marks: {sorted(unique_marks)} marks per question")
            
            # Question type
            if section.get('question_type', 'mixed') != 'mixed':
                q_type = section['question_type'].replace('_', ' ').title()
                description.append(f"  - Question Type: {q_type}")
            
            # Sample content hint
            if section.get('raw_content'):
                content_hint = section['raw_content'][:100].replace('\n', ' ')
                description.append(f"  - Content Style: {content_hint}...")
    
    # General instructions
    if structure_info['instructions']:
        description.append("\nGENERAL INSTRUCTIONS TO INCLUDE:")
        for instruction in structure_info['instructions'][:3]:
            description.append(f"  - {instruction}")
    
    # Critical format requirements
    description.append("\nCRITICAL REQUIREMENTS:")
    description.append("- Generate the EXACT number of questions for each section as specified above")
    description.append("- Use IDENTICAL section names and organization")
    description.append("- Maintain EXACT marks distribution")
    description.append("- Follow SAME question numbering style")
    description.append("- Include SAME instruction format")
    description.append("- Generate questions matching detected question types")
    description.append("- DO NOT add or remove questions - match the exact count from past papers")
    
    return "\n".join(description)

def strip_exam_header(text: str) -> str:
    """Remove all exam headers, metadata, and duplicate content completely"""
    lines = text.splitlines()
    clean = []
    
    for line in lines:
        line_lower = line.lower().strip()
        original_line = line.strip()
        
        # Skip ALL header content and metadata
        if any(kw in line_lower for kw in [
            "tribhuvan university", "institute of science", "bachelor level",
            "seventh semester", "computer science and information technology",
            "full marks:", "pass marks:", "time:", "subject:", "title:",
            "candidates are required", "general instructions:",
            "--- general instructions ---", "advanced java programming",
            "figures in the margin", "margin indicate", "instructions:",
            "the figures in the margin indicate full marks"
        ]):
            continue
            
        # Skip ALL bracketed instructions
        if (line_lower.startswith('[') and line_lower.endswith(']')):
            continue
            
        # Skip course codes and marks patterns
        if re.match(r'^(csc\d+|\d+\s*\+\s*\d+|full\s+marks)', line_lower):
            continue
            
        # Skip subject/title combination lines
        if re.match(r'^subject:\s*title:', line_lower):
            continue
            
        # Skip standalone subject names that are duplicates
        if (line_lower == "advanced java programming" or 
            line_lower.startswith("advanced java") or
            re.match(r'^[a-z\s]+programming$', line_lower)):
            continue
            
        # Skip empty lines or lines with just dashes/formatting
        if not original_line or original_line in ['---', '***', '===', '...']:
            continue
            
        # Only keep actual content (sections and questions)
        if (original_line.lower().startswith(('section', 'part', 'group', 'unit')) or
            re.match(r'^\d+\.', original_line) or
            len(original_line) > 10):  # Keep substantial content
            clean.append(original_line)
    
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


def extract_subject_name(syllabus_text: str, past_context: str = "") -> str:
    """Extract subject name from syllabus or past questions"""
    # Combine both texts for better subject detection
    combined_text = f"{syllabus_text}\n{past_context}".lower()
    
    # Enhanced subject patterns for better extraction
    subject_patterns = [
        # Handle "Subject: Title: Advanced Java Programming" format
        r"subject:\s*title:\s*([^.\n]+?)(?:\s+full\s+marks|$)",
        r"subject[:\s]+([^.\n]+?)(?:\s+full\s+marks|$)",
        r"course[:\s]+([^.\n]+?)(?:\s+full\s+marks|$)",
        # Programming languages and technologies
        r"(advanced\s+java\s+programming|java\s+programming|advanced\s+java)",
        r"(data\s+warehousing\s+and\s+data\s+mining|data\s+mining)",
        r"(software\s+project\s+management|project\s+management)",
        r"(principles\s+of\s+management|management)",
        # General patterns
        r"([a-z]+\s+[a-z]+)\s+(?:programming|language|technology|engineering|science|management)",
        r"([a-z\s]{10,50})\s+(?:syllabus|course|paper|exam)"
    ]
    
    for pattern in subject_patterns:
        matches = re.findall(pattern, combined_text, re.IGNORECASE)
        if matches:
            # Clean and format the subject name
            subject = matches[0].strip()
            
            # Remove unwanted words and patterns
            subject = re.sub(r'\b(syllabus|course|paper|exam|subject|code|title|full|marks)\b', '', subject, flags=re.IGNORECASE)
            subject = re.sub(r'\d+\s*\+\s*\d+', '', subject)  # Remove marks like "60 + 20 + 20"
            subject = re.sub(r'\s+', ' ', subject).strip()    # Clean up spaces
            
            if len(subject) > 3:  # Ensure it's a meaningful subject name
                # Special formatting for common subjects
                if 'java' in subject.lower():
                    return "Advanced Java Programming"
                elif 'data' in subject.lower() and ('mining' in subject.lower() or 'warehousing' in subject.lower()):
                    return "Data Warehousing and Data Mining"
                elif 'project' in subject.lower() and 'management' in subject.lower():
                    return "Software Project Management"
                else:
                    return subject.title()
    
    # Fallback: try to extract from common academic patterns
    academic_patterns = [
        r"([A-Z]{2,4}\s*\d{3,4})",  # Course codes like CSC 401, IT 205
        r"([A-Z][a-z]+\s+[A-Z][a-z]+)",  # Two-word subjects
    ]
    
    for pattern in academic_patterns:
        matches = re.findall(pattern, combined_text)
        if matches:
            return matches[0]
    
    return "Computer Science"  # Default fallback


def clean_question_body(text: str) -> str:
    """Clean question text from markdown, formatting artifacts, marks, and duplicate content"""
    # First apply header stripping to remove duplicate content
    text = strip_exam_header(text)
    
    # Remove markdown formatting
    text = re.sub(r"\*+", "", text)           # remove asterisks
    text = re.sub(r"_+", "", text)            # remove underscores
    text = re.sub(r"`+", "", text)            # remove backticks
    text = re.sub(r"#+", "", text)            # remove markdown headings
    text = re.sub(r"\.\.\.+", "", text)       # remove ellipses (...)
    
    # Remove duplicate subject/title lines
    text = re.sub(r'subject:\s*title:\s*[^\n]+full\s+marks:\s*\d+[^\n]*', '', text, flags=re.IGNORECASE)
    
    # Remove margin instruction lines
    text = re.sub(r'instructions?:\s*the figures in the margin indicate full marks\.?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'the figures in the margin indicate full marks\.?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'figures in the margin indicate full marks\.?', '', text, flags=re.IGNORECASE)
    
    # Remove marks from questions - handle various formats
    text = re.sub(r'\(\d+\s*marks?\)', '', text, flags=re.IGNORECASE)  # (20 Marks)
    text = re.sub(r'\[\d+\s*marks?\]', '', text, flags=re.IGNORECASE)  # [20 marks]
    text = re.sub(r'\(\d+\)', '', text)                                # (20)
    text = re.sub(r'\[\d+\]', '', text)                                # [20]
    text = re.sub(r'\d+\s*marks?\s*$', '', text, flags=re.IGNORECASE | re.MULTILINE)  # 20 marks at end of line
    
    # Remove repeated bracket instructions
    lines = text.splitlines()
    clean_lines = []
    seen_brackets = set()
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Handle bracketed instructions - avoid duplicates
        if line.startswith('[') and line.endswith(']'):
            bracket_content = line.lower().strip('[]').strip()
            if bracket_content in seen_brackets:
                continue
            seen_brackets.add(bracket_content)
            
        # Additional marks removal for individual lines
        line = re.sub(r'^(\d+\.)\s*\(\d+\s*marks?\)\s*', r'\1 ', line, flags=re.IGNORECASE)  # "1. (20 Marks) Question" -> "1. Question"
        line = re.sub(r'^(\d+\.)\s*\[\d+\s*marks?\]\s*', r'\1 ', line, flags=re.IGNORECASE)  # "1. [20 marks] Question" -> "1. Question"
        
        clean_lines.append(line)
    
    text = '\n'.join(clean_lines)
    
    # Join question numbers with their content on the same line
    lines = text.splitlines()
    joined_lines = []
    i = 0
    
    while i < len(lines):
        current_line = lines[i].strip()
        
        # If this is a question number line (like "1." or "2.")
        if re.match(r'^\d+\.\s*$', current_line) and i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            # Join the question number with the next line
            joined_lines.append(f"{current_line} {next_line}")
            i += 2  # Skip the next line since we've joined it
        else:
            joined_lines.append(current_line)
            i += 1
    
    text = '\n'.join(joined_lines)
    
    # Final cleanup
    text = re.sub(r"\s{2,}", " ", text)       # collapse multiple spaces
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)  # collapse big blank areas
    
    return text.strip()

def force_section_breaks(text: str) -> str:
    """Ensure Section or Part headings start on new lines"""
    return re.sub(r"(Section\s+[A-Z])", r"\n\1", text, flags=re.I)

def force_newlines(text: str) -> str:
    """Ensure every numbered question starts on its own line"""
    return re.sub(r"(?<!^)(\s*)(\d+\.)", r"\n\2", text)

def detect_numbering_style(past_text: str) -> str:
    """Detect if questions restart numbering per section or continue globally"""
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
    """Format questions with proper numbering based on detected style"""
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

def analyze_paper_structure(past_context: str) -> dict:
    """Analyze the complete structure of past papers including marks, sections, and instructions"""
    structure_info = {
        'university': '',
        'institute': '',
        'course_code': '',
        'subject_name': '',
        'full_marks': '',
        'pass_marks': '',
        'time': '',
        'year': '',
        'semester': '',
        'sections': [],
        'total_questions': 0,
        'numbering_style': 'global',
        'instructions': [],
        'marks_distribution': {}
    }
    
    text_lower = past_context.lower()
    
    # Extract basic header information
    header_patterns = {
        'university': r'(tribhuvan university|kathmandu university|pokhara university|purbanchal university)',
        'institute': r'(institute of [^\n]+|college of [^\n]+|faculty of [^\n]+)',
        'course_code': r'([A-Z]{2,4}[\s-]?\d{3,4})',
        'full_marks': r'full marks?[:\s]+(\d+)',
        'pass_marks': r'pass marks?[:\s]+(\d+)',
        'time': r'time[:\s]+([^\n]+)',
        'year': r'(20\d{2})',
        'semester': r'(\d+(?:st|nd|rd|th)?\s*semester)'
    }
    
    for key, pattern in header_patterns.items():
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        if matches:
            if key in ['university', 'institute']:
                structure_info[key] = matches[0].title()
            else:
                structure_info[key] = matches[0]
    
    # Enhanced section detection - handle any section type
    section_keywords = ['section', 'part', 'group', 'unit', 'chapter', 'question', 'problem']
    section_pattern = f"({'|'.join(section_keywords)})\\s+([a-z0-9]+|[ivxlc]+|one|two|three|four|five)([^\\n]*?)(?=({'|'.join(section_keywords)})\\s+[a-z0-9]+|$)"
    sections = re.findall(section_pattern, past_context, re.IGNORECASE | re.DOTALL)
    
    for section_match in sections:
        section_type = section_match[0].title()
        section_id = section_match[1].upper()
        section_content = section_match[2]
        
        # Enhanced instruction patterns for maximum flexibility
        instruction_patterns = [
            # Answer patterns with numbers and words
            r'answer\s+any\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+questions?',
            r'attempt\s+any\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+questions?',
            r'solve\s+any\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+questions?',
            r'choose\s+any\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+questions?',
            r'select\s+any\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+questions?',
            # Out of patterns
            r'answer\s+any\s+(\d+)\s+(?:questions?\s+)?out\s+of\s+(\d+)\s+questions?',
            r'attempt\s+any\s+(\d+)\s+(?:questions?\s+)?out\s+of\s+(\d+)\s+questions?',
            r'choose\s+(\d+)\s+(?:questions?\s+)?out\s+of\s+(\d+)\s+questions?',
            # Compulsory patterns
            r'all\s+questions?\s+are\s+compulsory',
            r'compulsory\s+questions?',
            r'answer\s+all\s+questions?',
            r'attempt\s+all\s+questions?',
            # Marks patterns - various formats
            r'each\s+(?:question\s+)?carries\s+(\d+)\s+marks?',
            r'(\d+)\s+marks?\s+each',
            r'(\d+)\s+marks?\s+per\s+question',
            r'\[(\d+)\s+marks?\]',
            r'\((\d+)\s+marks?\)',
            # Total questions
            r'out\s+of\s+(\d+)\s+questions?',
            r'from\s+(\d+)\s+questions?',
            r'total\s+(\d+)\s+questions?'
        ]
        
        section_info = {
            'type': section_type,
            'id': section_id,
            'instructions': [],
            'questions': [],
            'marks_per_question': [],
            'total_questions': 0,
            'questions_to_attempt': 0,
            'total_available': 0,
            'question_type': 'mixed',
            'raw_content': ''
        }
        
        # Helper function to convert word numbers to digits
        def word_to_num(word):
            word_map = {
                'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
                'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
            }
            return word_map.get(word.lower(), word if word.isdigit() else 0)
        
        # Extract instructions with enhanced flexibility
        for pattern in instruction_patterns:
            matches = re.findall(pattern, section_content, re.IGNORECASE)
            if matches:
                if 'out of' in pattern and isinstance(matches[0], tuple) and len(matches[0]) == 2:
                    # Pattern like "answer any 3 out of 4 questions"
                    section_info['questions_to_attempt'] = int(matches[0][0])
                    section_info['total_available'] = int(matches[0][1])
                elif 'any' in pattern and not 'out of' in pattern:
                    # Pattern like "answer any 5 questions"
                    match_val = matches[0]
                    if isinstance(match_val, str):
                        if match_val.isdigit():
                            section_info['questions_to_attempt'] = int(match_val)
                        else:
                            section_info['questions_to_attempt'] = word_to_num(match_val)
                elif 'carries' in pattern or 'marks each' in pattern or 'per question' in pattern or '[' in pattern or '(' in pattern:
                    # Extract marks
                    mark_value = int(matches[0]) if matches[0].isdigit() else 0
                    if mark_value > 0:
                        section_info['marks_per_question'].append(mark_value)
                elif 'compulsory' in pattern or 'all questions' in pattern:
                    section_info['questions_to_attempt'] = 'all'
                elif 'out of' in pattern or 'from' in pattern or 'total' in pattern:
                    # Total available questions
                    total_val = int(matches[0]) if matches[0].isdigit() else 0
                    if total_val > 0:
                        section_info['total_available'] = total_val
        
        # Count questions in this section - handle various numbering formats
        question_patterns = [
            r'^\s*(\d+)\.\s',  # Standard: 1. 2. 3.
            r'^\s*([ivxlc]+)\.\s',  # Roman: i. ii. iii.
            r'^\s*\((\d+)\)\s',  # Parentheses: (1) (2) (3)
            r'^\s*([a-z])\.\s',  # Letters: a. b. c.
            r'^\s*([A-Z])\.\s'   # Capital letters: A. B. C.
        ]
        
        all_questions = []
        for pattern in question_patterns:
            questions = re.findall(pattern, section_content, re.MULTILINE)
            all_questions.extend(questions)
        
        section_info['total_questions'] = len(set(all_questions))  # Remove duplicates
        
        # If total_available not set, use detected questions
        if section_info['total_available'] == 0:
            section_info['total_available'] = section_info['total_questions']
        
        # Extract marks from individual questions - multiple formats
        marks_patterns = [
            r'\[(\d+)\s*marks?\]',  # [5 marks]
            r'\((\d+)\s*marks?\)',  # (5 marks)
            r'\[(\d+)\]',           # [5]
            r'\((\d+)\)',           # (5)
            r'marks?[:\s]+(\d+)',   # marks: 5
            r'(\d+)\s*marks?\s*$'   # 5 marks at end of line
        ]
        
        for pattern in marks_patterns:
            marks_found = re.findall(pattern, section_content, re.IGNORECASE)
            if marks_found:
                marks = [int(m) for m in marks_found if m.isdigit()]
                section_info['marks_per_question'].extend(marks)
        
        # Detect question type based on content
        section_info['question_type'] = detect_question_type(section_content)
        
        # Store raw content for reference
        section_info['raw_content'] = section_content.strip()[:300]
        
        structure_info['sections'].append(section_info)
    
    # Extract general instructions
    instruction_patterns = [
        r'candidates?\s+are\s+required\s+to[^\n]+',
        r'answer\s+all\s+questions?[^\n]*',
        r'attempt\s+all\s+questions?[^\n]*',
        r'read\s+the\s+questions?\s+carefully[^\n]*',
        r'figures?\s+in\s+the\s+margin\s+indicate[^\n]*',
        r'the\s+figures?\s+in\s+the\s+margin[^\n]*'
    ]
    
    for pattern in instruction_patterns:
        matches = re.findall(pattern, past_context, re.IGNORECASE)
        structure_info['instructions'].extend(matches)
    
    # Determine numbering style
    structure_info['numbering_style'] = detect_numbering_style(past_context)
    
    # Calculate total questions
    structure_info['total_questions'] = sum(s['total_questions'] for s in structure_info['sections'])
    
    return structure_info

def detect_question_type(content: str) -> str:
    """Detect question type: objective, subjective_short, subjective_long, or mixed"""
    content_lower = content.lower()
    
    # Objective question indicators
    objective_patterns = [
        r'choose\s+the\s+correct',
        r'select\s+the\s+best',
        r'multiple\s+choice',
        r'tick\s+the\s+correct',
        r'circle\s+the\s+correct',
        r'\b[a-d]\)\s+',  # Options like a) b) c) d)
        r'\([a-d]\)\s+',   # Options like (a) (b) (c) (d)
        r'true\s+or\s+false',
        r'fill\s+in\s+the\s+blanks?',
        r'one\s+word\s+answer',
        r'match\s+the\s+following',
        r'state\s+true\s+or\s+false'
    ]
    
    # Short subjective question indicators
    subjective_short_patterns = [
        r'define\s+',
        r'what\s+is\s+',
        r'list\s+the\s+',
        r'mention\s+',
        r'state\s+',
        r'write\s+the\s+formula',
        r'give\s+the\s+meaning',
        r'short\s+answer',
        r'brief\s+answer',
        r'one\s+line\s+answer',
        r'expand\s+',
        r'full\s+form\s+of'
    ]
    
    # Long subjective question indicators
    subjective_long_patterns = [
        r'explain\s+in\s+detail',
        r'describe\s+briefly',
        r'discuss\s+the',
        r'elaborate\s+on',
        r'write\s+an\s+essay',
        r'compare\s+and\s+contrast',
        r'analyze\s+the',
        r'critically\s+examine',
        r'justify\s+your\s+answer',
        r'with\s+suitable\s+examples',
        r'write\s+short\s+notes?\s+on',
        r'explain\s+with\s+examples',
        r'derive\s+the\s+formula',
        r'prove\s+that'
    ]
    
    # Count pattern matches
    obj_count = sum(len(re.findall(p, content_lower)) for p in objective_patterns)
    short_subj_count = sum(len(re.findall(p, content_lower)) for p in subjective_short_patterns)
    long_subj_count = sum(len(re.findall(p, content_lower)) for p in subjective_long_patterns)
    
    # Determine question type based on predominant patterns
    total_patterns = obj_count + short_subj_count + long_subj_count
    
    if total_patterns == 0:
        return 'mixed'
    
    if obj_count > short_subj_count and obj_count > long_subj_count:
        return 'objective'
    elif long_subj_count > obj_count and long_subj_count > short_subj_count:
        return 'subjective_long'
    elif short_subj_count > obj_count and short_subj_count > long_subj_count:
        return 'subjective_short'
    else:
        return 'mixed'

def extract_exam_header_info(past_context: str) -> dict:
    """Extract exam header information from past papers (backward compatibility)"""
    structure = analyze_paper_structure(past_context)
    return {
        'university': structure['university'],
        'institute': structure['institute'],
        'course_code': structure['course_code'],
        'subject_name': structure['subject_name'],
        'full_marks': structure['full_marks'],
        'pass_marks': structure['pass_marks'],
        'time': structure['time'],
        'year': structure['year'],
        'semester': structure['semester']
    }

def paper_to_pdf_bytes(title: str, content: str, past_context: str = "", subject_name: str = "") -> bytes:
    """Convert paper content to PDF bytes with structure-aware academic formatting"""
    if not REPORTLAB_OK:
        raise RuntimeError("ReportLab not installed. Run: pip install reportlab")
    
    # Analyze complete paper structure from past papers
    structure_info = analyze_paper_structure(past_context) if past_context else {}
    
    # Apply enhanced formatting if past context is available
    if past_context:
        # Apply formatting transformations but keep exam headers
        formatted_content = clean_question_body(content)
        formatted_content = force_section_breaks(formatted_content)
        formatted_content = force_newlines(formatted_content)
        formatted_content = format_questions(formatted_content, style=structure_info.get('numbering_style', 'global'))
        content = formatted_content
    
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, 
        pagesize=A4, 
        title=title,
        topMargin=0.5*inch,
        bottomMargin=0.5*inch,
        leftMargin=0.75*inch,
        rightMargin=0.75*inch
    )
    styles = getSampleStyleSheet()
    
    # Create custom styles for structure-aware formatting
    # University header style
    university_style = ParagraphStyle(
        'UniversityHeader',
        parent=styles['Title'],
        fontSize=16,
        spaceAfter=8,
        alignment=1,  # Center alignment
        textColor=black,
        fontName='Helvetica-Bold'
    )
    
    # Institute header style
    institute_style = ParagraphStyle(
        'InstituteHeader',
        parent=styles['Heading1'],
        fontSize=12,
        spaceAfter=8,
        alignment=1,  # Center alignment
        textColor=black,
        fontName='Helvetica-Bold'
    )
    
    # Exam info style
    exam_info_style = ParagraphStyle(
        'ExamInfo',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=4,
        alignment=1,  # Center alignment
        textColor=black,
        fontName='Helvetica'
    )
    
    # Subject title style
    subject_title_style = ParagraphStyle(
        'SubjectTitle',
        parent=styles['Title'],
        fontSize=14,
        spaceBefore=4,
        textColor=black,
        fontName='Helvetica'
    )
    
    # Section header style
    section_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading1'],
        fontSize=12,
        spaceBefore=4,
        spaceAfter=8,
        textColor=black,
        fontName='Helvetica-Bold',
        leftIndent=0
    )
    
    # Section instructions style
    section_instructions_style = ParagraphStyle(
        'SectionInstructions',
        parent=styles['Normal'],
        fontSize=10,
        spaceBefore=4,
        spaceAfter=8,
        leftIndent=10,
        textColor=black,
        fontName='Helvetica'
    )
    
    # Question number style
    question_style = ParagraphStyle(
        'QuestionNumber',
        parent=styles['Normal'],
        fontSize=11,
        spaceBefore=8,
        spaceAfter=4,
        textColor=black,
        fontName='Helvetica',
        leftIndent=0
    )
    
    # Question text style
    question_text_style = ParagraphStyle(
        'QuestionText',
        parent=styles['Normal'],
        fontSize=11,
        spaceBefore=2,
        spaceAfter=6,
        leftIndent=20,
        textColor=black,
        fontName='Helvetica',
        alignment=0  # Left alignment
    )
    
   
    
    story = []
    
    # # Add structure-aware header
    # if structure_info.get('university'):
    #     story.append(Paragraph(structure_info['university'], university_style))
    
    # if structure_info.get('institute'):
    #     story.append(Paragraph(structure_info['institute'], institute_style))
    
    # Add exam information line with detected values
    
    
    # Add clean subject title
    if subject_name:
        # Clean the subject name from any extra formatting
        clean_subject = subject_name.strip()
        # Remove any "Subject:" prefix if it exists
        if clean_subject.lower().startswith('subject:'):
            clean_subject = clean_subject[8:].strip()
        story.append(Paragraph(f"Predicted Paper : {clean_subject}", subject_title_style))
    elif structure_info.get('course_code'):
        story.append(Paragraph(f"Predicted Paper : {structure_info['course_code']}", subject_title_style))
    else:
        story.append(Paragraph(f" Predicted Paper  {title}", subject_title_style))


    exam_info_parts = []
    if structure_info.get('full_marks'):
        exam_info_parts.append(f"Full Marks: {structure_info['full_marks']}")
    if structure_info.get('pass_marks'):
        exam_info_parts.append(f"Pass Marks: {structure_info['pass_marks']}")
    if structure_info.get('time'):
        exam_info_parts.append(f"Time: {structure_info['time']}")
    
    if exam_info_parts:
        story.append(Paragraph(" | ".join(exam_info_parts), exam_info_style))
    
    # Skip general instructions for clean format
    
    # Add a line separator
    story.append(Spacer(1, 12))
    
    # Process content with structure-aware formatting
    lines = content.split('\n')
    current_section = None
    in_question = False
    question_parts = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Handle section headers - support any section type
        if re.match(r'^(section|part|group|unit|chapter|problem)\s+[a-z0-9]+', line, re.IGNORECASE):
            # Process any pending question parts
            if question_parts:
                story.append(Paragraph(' '.join(question_parts), question_text_style))
                question_parts = []
            
            story.append(Paragraph(line.upper(), section_style))
            
            # Add section-specific instructions if available
            section_id = re.search(r'(section|part|group|unit|chapter|problem)\s+([a-z0-9]+)', line, re.IGNORECASE)
            if section_id and structure_info.get('sections'):
                section_letter = section_id.group(2).upper()
                for section_info_item in structure_info['sections']:
                    if section_info_item['id'] == section_letter:
                        instruction_parts = []
                        if section_info_item['questions_to_attempt']:
                            if section_info_item['questions_to_attempt'] == 'all':
                                instruction_parts.append("All questions are compulsory")
                            else:
                                instruction_parts.append(f"Answer any {section_info_item['questions_to_attempt']} questions")
                        
                        if section_info_item['marks_per_question']:
                            unique_marks = list(set(section_info_item['marks_per_question']))
                            if len(unique_marks) == 1:
                                instruction_parts.append(f"Each question carries {unique_marks[0]} marks")
                        
                        if instruction_parts:
                            story.append(Paragraph(f"[{'. '.join(instruction_parts)}]", section_instructions_style))
                        break
            
            story.append(Spacer(1, 8))
            current_section = section_id.group(2).upper() if section_id else None
            
        # Handle instructions lines (but skip duplicate candidate instructions)
        elif (any(keyword in line.lower() for keyword in ['answer any', 'attempt', 'all questions', 'compulsory', 'each question carries']) and 
              'candidates are required to give their answers' not in line.lower()):
            story.append(Paragraph(f"[{line}]", section_instructions_style))
            story.append(Spacer(1, 6))
            
        # Handle numbered questions
        elif re.match(r'^\d+\.', line):
            # Process any pending question parts
            if question_parts:
                story.append(Paragraph(' '.join(question_parts), question_text_style))
                question_parts = []
            
            story.append(Paragraph(line, question_style))
            in_question = True
            
        # Handle sub-questions
        elif re.match(r'^[a-z]\)|^\([a-z]\)', line):
            story.append(Paragraph(line, question_text_style))
            
        # Handle regular question text
        else:
            if in_question:
                question_parts.append(line)
            else:
                story.append(Paragraph(line, question_text_style))
    
    # Process any remaining question parts
    if question_parts:
        story.append(Paragraph(' '.join(question_parts), question_text_style))
    
    doc.build(story)
    pdf_data = buf.getvalue()
    buf.close()
    return pdf_data

def render_question_paper(raw_text: str, st, past_context: str = ""):
    """Render question paper content in Streamlit with dynamic structure analysis"""
    # Apply enhanced formatting if past context is available
    if past_context:
        # Analyze complete paper structure
        structure_info = analyze_paper_structure(past_context)
        
        # Apply formatting transformations
        formatted_text = clean_question_body(raw_text)
        formatted_text = force_section_breaks(formatted_text)
        formatted_text = force_newlines(formatted_text)
        formatted_text = format_questions(formatted_text, style=structure_info['numbering_style'])
        
        # Display comprehensive structure information
        if any([structure_info.get('university'), structure_info.get('full_marks'), structure_info['sections']]):
            st.markdown("### 📋 Analyzed Paper Structure")
            
            # Basic info in columns
            info_cols = st.columns(3)
            with info_cols[0]:
                if structure_info.get('university'):
                    st.markdown(f"**University:** {structure_info['university']}")
                if structure_info.get('institute'):
                    st.markdown(f"**Institute:** {structure_info['institute']}")
            
            with info_cols[1]:
                if structure_info.get('full_marks'):
                    st.markdown(f"**Full Marks:** {structure_info['full_marks']}")
                if structure_info.get('pass_marks'):
                    st.markdown(f"**Pass Marks:** {structure_info['pass_marks']}")
            
            with info_cols[2]:
                if structure_info.get('time'):
                    st.markdown(f"**Time:** {structure_info['time']}")
                st.markdown(f"**Total Questions:** {structure_info['total_questions']}")
            
            # Section structure details
            if structure_info['sections']:
                st.markdown("#### 📊 Section Analysis")
                for section in structure_info['sections']:
                    with st.expander(f"{section['type']} {section['id']} Details"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Total Questions:** {section['total_questions']}")
                            if section['questions_to_attempt']:
                                if section['questions_to_attempt'] == 'all':
                                    st.write("**Attempt:** All questions (Compulsory)")
                                else:
                                    st.write(f"**Attempt:** Any {section['questions_to_attempt']} questions")
                        
                        with col2:
                            if section['marks_per_question']:
                                unique_marks = list(set(section['marks_per_question']))
                                if len(unique_marks) == 1:
                                    st.write(f"**Marks per Question:** {unique_marks[0]}")
                                else:
                                    st.write(f"**Marks Distribution:** {unique_marks}")
            
            st.markdown("---")
        
        # Display formatted text
        st.text_area("Predicted Paper (Structure-Matched Format)", formatted_text, height=600)
        
        # Show analysis results
        st.success(f"✅ Replicated '{structure_info['numbering_style']}' numbering and {len(structure_info['sections'])} section structure from past papers")
        
        # Detailed analysis in expander
        with st.expander("🔍 Complete Structure Analysis"):
            st.json({
                "Paper Structure": {
                    "Sections": len(structure_info['sections']),
                    "Total Questions": structure_info['total_questions'],
                    "Numbering Style": structure_info['numbering_style']
                },
                "Header Info": {k: v for k, v in structure_info.items() if k in ['university', 'institute', 'full_marks', 'pass_marks', 'time'] and v},
                "Section Details": [
                    {
                        "Section": f"{s['type']} {s['id']}",
                        "Questions": s['total_questions'],
                        "Attempt": s['questions_to_attempt'],
                        "Marks": s['marks_per_question']
                    } for s in structure_info['sections']
                ],
                "Detected Instructions": structure_info['instructions'][:3]
            })
    else:
        # Enhanced fallback rendering
        st.text_area("Predicted Paper (Basic Format)", raw_text, height=600)
        st.info("💡 Upload past question papers for dynamic structure analysis and replication")
