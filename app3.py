from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from io import BytesIO


def paper_to_pdf_bytes(title: str, content: str, subject: str) -> bytes:
    """
    Generate a PDF for the predicted question paper with structured sections.
    
    :param title: General PDF title (e.g., "Predicted Paper")
    :param content: Full structured text of the predicted paper
    :param subject: Subject name to include in PDF title
    :return: PDF file in bytes
    """
    buf = BytesIO()
    pdf_title = f"Predicted Questions for {subject}"
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        title=pdf_title,
        rightMargin=30,
        leftMargin=30,
        topMargin=30,
        bottomMargin=30,
    )

    styles = getSampleStyleSheet()
    story = []

    # Title style
    title_style = ParagraphStyle(
        "title_style",
        parent=styles["Title"],
        alignment=TA_CENTER,
        spaceAfter=20,
        fontSize=18,
    )
    story.append(Paragraph(pdf_title, title_style))
    story.append(Spacer(1, 12))

    # Section heading style
    section_style = ParagraphStyle(
        "section_style",
        parent=styles["Heading2"],
        alignment=TA_CENTER,
        spaceBefore=15,
        spaceAfter=10,
        fontSize=14,
    )

    # Question/Body style
    question_style = ParagraphStyle(
        "question_style",
        parent=styles["Normal"],
        alignment=TA_LEFT,
        fontSize=12,
        leading=16,
        spaceAfter=8,
    )

    # Split content into lines
    lines = content.splitlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue  # skip empty lines

        # Detect Section headings
        if line.lower().startswith("section"):
            story.append(Spacer(1, 12))
            story.append(Paragraph(line, section_style))
            story.append(Spacer(1, 8))
        else:
            # Regular question text
            story.append(Paragraph(line.replace("\n", "<br/>"), question_style))

    # Build PDF
    doc.build(story)
    return buf.getvalue()


# define a simple main so script can run without NameError
def main():
    sample_content = """Section A
1. Example question [5]
2. Another one [10]"""
    pdf_bytes = paper_to_pdf_bytes("Predicted Paper", sample_content, "Java")
    with open("predicted_paper.pdf", "wb") as f:
        f.write(pdf_bytes)
    print("PDF saved as predicted_paper.pdf")


if __name__ == "__main__":
    main()
