# 📘 ShikshaChautari - AI Question Paper Generator

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)

An intelligent AI-powered question paper generator that creates structured academic exam papers using past question patterns and syllabus content. Built with Streamlit for the web interface and FastAPI for cloud-based processing.

## 🌟 Features

### 🎯 Core Functionality
- **AI-Powered Generation**: Uses OpenAI GPT-4o-mini with LangChain for intelligent question generation
- **Pattern Recognition**: Analyzes past exam papers to replicate structure, difficulty, and formatting
- **Syllabus Integration**: Generates fresh questions based on provided syllabus topics
- **Mark Preservation**: Maintains exact mark distributions and question types from past papers
- **Structured Formatting**: Applies POM1 (Principles of Management) style formatting automatically

### 📄 Document Processing
- **PDF Text Extraction**: Processes syllabus and past question PDFs
- **Text Cleaning Pipeline**: Removes headers, footers, and artifacts while preserving marks
- **Vector Search**: Uses FAISS for efficient similarity search across past questions
- **Multi-format Support**: Handles various PDF layouts and text structures

### 🎨 Output Formats
- **Web Interface**: Interactive Streamlit app with real-time preview
- **PDF Export**: Generates downloadable PDFs with proper formatting
- **API Integration**: RESTful API for automated processing
- **Cloud Storage**: S3 integration for scalable file handling

### 🔧 Technical Features
- **Dual Interface**: Both web app and API endpoints
- **Session Management**: Isolated processing for concurrent users
- **Error Handling**: Comprehensive error handling and user feedback
- **Modular Architecture**: Clean separation of concerns with reusable utilities

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- OpenAI API key
- AWS credentials (for S3 integration)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/shikshychautari-baseversion.git
   cd shikshychautari-baseversion
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

5. **Run the application**
   ```bash
   # For web interface
   streamlit run app.py

   # For API server
   python api.py
   ```

## 📖 Usage

### Web Interface (Streamlit)

1. **Access the application** at `http://localhost:8501`
2. **Upload Files**:
   - **Syllabus PDF**: Course syllabus or curriculum document
   - **Past Questions PDFs**: Previous exam papers (multiple files supported)
3. **Process Files**: Click "Process Files" to analyze and prepare the content
4. **Generate Paper**: Click "Generate Predicted Paper" to create new questions
5. **Download PDF**: Use the download button to save the formatted question paper

### API Usage

#### Generate Paper from S3

```bash
curl -X POST "http://localhost:8000/generate-paper-s3" \
  -H "Content-Type: application/json" \
  -d '{
    "bucket_name": "your-s3-bucket",
    "old_question_prefix": "past-questions/",
    "syllabus_prefix": "syllabus/",
    "predicted_question_prefix": "generated-papers/"
  }'
```

#### API Endpoints

- `GET /` - API information and Swagger UI link
- `POST /generate-paper-s3` - Generate paper from S3-stored PDFs

## 🏗️ Architecture

```
shikshychautari-baseversion/
├── app.py                 # Streamlit web application
├── api.py                 # FastAPI backend service
├── utils.py              # Core utilities and AI functions
├── requirements.txt      # Python dependencies
├── .env.example         # Environment variables template
├── data/                # Sample data directory
│   ├── raw/            # Raw PDF files
│   │   ├── DWDM/      # Data Warehousing & Mining
│   │   ├── JAVA/      # Java Programming
│   │   ├── POM/       # Principles of Management
│   │   └── SPM/       # Software Project Management
│   └── ...
├── faiss_index/        # Vector database storage
└── README.md           # This file
```

### Data Flow

1. **Input Processing**: PDFs are uploaded and text is extracted
2. **Text Cleaning**: Headers, footers, and artifacts are removed
3. **Vectorization**: Past questions are chunked and embedded using OpenAI
4. **AI Generation**: GPT model generates new questions based on syllabus and past patterns
5. **Formatting**: Questions are structured in POM1 style with preserved marks
6. **Output**: Formatted paper is displayed and/or exported as PDF

## ⚙️ Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# AWS S3 Configuration (for API)
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=eu-north-1
```

### Optional Dependencies

- **reportlab**: For PDF generation (automatically installed)
- **boto3**: For S3 integration (automatically installed)

## 📊 Sample Data Structure

The `data/raw/` directory contains sample PDFs organized by subject:

```
data/raw/
├── DWDM/                    # Data Warehousing & Data Mining
│   ├── 7 SemSyllabus-DWDM.pdf
│   ├── CSC410-Data-Warehousing-and-Data-Mining.pdf
│   └── DWDM_2078.jpg
├── JAVA/                    # Java Programming
│   ├── 7 SemSyllabus-JAVA - converted.pdf
│   └── java_2077.png
├── POM/                     # Principles of Management
│   ├── 7 SemSyllabus-POM.pdf
│   └── MGT411-Principles-of-Management.pdf
└── SPM/                     # Software Project Management
    ├── 7 SemSyllabus-SPM.pdf
    └── CSC415-Software-Project-Management.pdf
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Format code
black .
isort .
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io) for the web interface
- Powered by [OpenAI](https://openai.com) GPT models
- Uses [LangChain](https://langchain.com) for AI orchestration
- Vector search with [FAISS](https://github.com/facebookresearch/faiss)
- PDF processing with [PyPDF2](https://pypi.org/project/PyPDF2/)

## 📞 Support

For questions or issues:
- Open an issue on GitHub
- Check the documentation
- Review the code comments for implementation details

---

