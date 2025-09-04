# 📚 Question Paper Generator - Shikshychautari Base Version

An AI-powered application that generates predicted question papers from syllabus and past question papers using OpenAI GPT-4 and LangChain.

## 🎯 What This Project Does

This project provides **two interfaces** for generating predicted exam papers:
1. **Streamlit Web App** - Interactive web interface for easy use
2. **FastAPI REST API** - Programmatic access with Swagger documentation

Both use the same core AI logic to analyze past papers and syllabus content to predict future exam questions.

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- OpenAI API key (get from [OpenAI Platform](https://platform.openai.com/))

### 1. Setup Environment

```bash
# Clone or download the project
cd shikshychautari-baseversion

# Create virtual environment
py -m venv env

# Activate environment (Windows)
env\Scripts\activate
# Or on Linux/Mac: source env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
LANGCHAIN_API_KEY=your_langsmith_key_here  # Optional for tracking
```

### 3. Choose Your Interface

#### Option A: Streamlit Web App (Beginner Friendly)
```bash
streamlit run app.py
```
- Open browser to `http://localhost:8501`
- Upload PDFs through the web interface
- Generate and download question papers

#### Option B: FastAPI REST API (Advanced/Programmatic)
```bash
py fastapi_app.py
```
- API available at `http://localhost:8000`
- Interactive docs at `http://localhost:8000/swagger`
- **Two endpoints available:**
  - `POST /generate-paper` - Upload both syllabus and old question files
  - `POST /generate-paper-s3` - Upload syllabus + download old questions from S3

## 📁 Project Structure

```
shikshychautari-baseversion/
├── app.py                 # Streamlit web application
├── fastapi_app.py         # FastAPI REST API (single endpoint)
├── requirements.txt       # Python dependencies
├── .env                  # Environment variables (create this)
├── data/raw/             # Sample PDF files
│   ├── DWDM/            # Data Warehousing & Data Mining
│   ├── JAVA/            # Java Programming
│   ├── POM/             # Principles of Management
│   └── SPM/             # Software Project Management
└── env/                  # Virtual environment
```

## 🎓 Learning Path

### For Beginners
1. **Start with Streamlit App** - Easy to understand UI flow
2. **Explore the code** in `app.py` to understand:
   - PDF text extraction with PyPDF2
   - Text chunking with LangChain
   - Vector storage with FAISS
   - AI generation with OpenAI

### For Advanced Users
1. **Use the FastAPI** - RESTful architecture
2. **Study `fastapi_app.py`** to learn:
   - API endpoint design
   - Request/response models with Pydantic
   - Session management
   - Error handling
   - Swagger documentation

### Key Concepts to Learn

#### 1. **RAG (Retrieval Augmented Generation)**
```python
# How it works:
1. Upload PDFs → Extract text
2. Split into chunks → Create embeddings
3. Store in vector database (FAISS)
4. Query similar content → Generate with AI
```

#### 2. **LangChain Framework**
- **Text Splitters**: Break documents into manageable chunks
- **Embeddings**: Convert text to numerical vectors
- **Vector Stores**: Store and search embeddings
- **Chains**: Connect multiple AI operations
- **Prompts**: Structure AI instructions

#### 3. **FastAPI Features**
- **Automatic docs**: Swagger UI generation
- **Type hints**: Pydantic models for validation
- **Async support**: High-performance async/await
- **Dependency injection**: Clean architecture

## 🔧 How to Use

### Streamlit Web App

1. **Upload Syllabus**: Click "Upload Syllabus PDF" in sidebar
2. **Upload Past Papers**: Click "Upload Past Questions PDF(s)"
3. **Generate**: Click "Generate Predicted Paper" button
4. **Download**: Get PDF of the generated paper

### FastAPI Usage

#### Using Swagger UI (Easiest)
1. Go to `http://localhost:8000/swagger`
2. Try out each endpoint interactively
3. Upload files and generate papers

#### Using Python Requests
```python
import requests

# 1. Upload syllabus
with open('syllabus.pdf', 'rb') as f:
    files = [('files', f)]
    response = requests.post('http://localhost:8000/upload/syllabus', files=files)
    session_id = response.json()['session_id']

# 2. Upload old questions
with open('old_questions.pdf', 'rb') as f:
    files = [('files', f)]
    data = {'session_id': session_id}
    requests.post('http://localhost:8000/upload/old-questions', files=files, data=data)

# 3. Generate paper
response = requests.post('http://localhost:8000/generate', 
                        json={"session_id": session_id})
paper = response.json()['predicted_paper']
```

#### Using cURL
```bash
# Upload syllabus
curl -X POST "http://localhost:8000/upload/syllabus" \
  -F "files=@syllabus.pdf"

# Generate paper
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"session_id": "your-session-id"}'
```

## 🧪 Testing

### Test the API
Use the interactive Swagger UI at `http://localhost:8000/swagger` - much better than command line!

### Manual Testing
1. Use sample PDFs from `data/raw/` directory
2. Try different subjects (JAVA, DWDM, POM, SPM)
3. Compare generated papers with actual past papers

## 🔍 Understanding the AI Logic

### Prompt Engineering
The AI uses this structured prompt:
```python
template = """
You are an expert academic exam paper predictor.
Generate exactly **one** complete future exam paper...

Requirements:
1) Mirror the sections, numbering, and marks distribution
2) Maximize recurrence likelihood from past patterns
3) Fill all sections fully
4) Include underrepresented syllabus areas
5) Maintain academic tone and clarity
"""
```

### Vector Search
```python
# How similarity search works:
docs = past_db.similarity_search(
    "exam structure sections numbering marks distribution", 
    k=12  # Get 12 most similar chunks
)
```

## 🛠️ Customization

### Modify AI Behavior
Edit the prompt template in:
- `app.py` line 54-72 (Streamlit)
- `fastapi_app.py` line 108-130 (FastAPI)

### Change AI Model
```python
llm = ChatOpenAI(
    model_name="gpt-4o-mini",  # Try: gpt-4, gpt-3.5-turbo
    temperature=0.3,           # 0.0 = deterministic, 1.0 = creative
    max_tokens=1400           # Adjust output length
)
```

### Adjust Text Chunking
```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,    # Size of each chunk
    chunk_overlap=150   # Overlap between chunks
)
```

## 📚 Learning Resources

### LangChain
- [Official Documentation](https://python.langchain.com/)
- [LangChain Cookbook](https://github.com/langchain-ai/langchain/tree/master/cookbook)

### FastAPI
- [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/)
- [FastAPI Advanced Guide](https://fastapi.tiangolo.com/advanced/)

### Streamlit
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Streamlit Gallery](https://streamlit.io/gallery)

### Vector Databases
- [FAISS Documentation](https://faiss.ai/)
- [Understanding Embeddings](https://platform.openai.com/docs/guides/embeddings)

## 🐛 Troubleshooting

### Common Issues

**1. OpenAI API Key Error**
```bash
Error: No API key provided
```
Solution: Create `.env` file with `OPENAI_API_KEY=your_key`

**2. PDF Reading Error**
```bash
Error: No text could be extracted
```
Solution: Ensure PDFs contain text (not just images)

**3. Memory Error**
```bash
Error: Out of memory
```
Solution: Reduce `chunk_size` in text splitter

**4. Port Already in Use**
```bash
Error: Port 8000 is already in use
```
Solution: Kill existing process or use different port

### Getting Help
1. Check the error messages in terminal
2. Verify your `.env` file setup
3. Test with sample PDFs first
4. Use the test script: `python test_api.py`

## 🎯 Next Steps

### Beginner Projects
1. Modify the prompt to focus on specific question types
2. Add support for different file formats
3. Create a simple frontend for the API

### Advanced Projects
1. Add user authentication
2. Implement caching for faster responses
3. Add support for multiple languages
4. Create a database for storing sessions
5. Deploy to cloud (Heroku, AWS, etc.)

### Learning Exercises
1. Trace through the code step by step
2. Experiment with different AI models
3. Try different chunking strategies
4. Build your own evaluation metrics

## 📄 License

This project is open source. Feel free to modify and learn from it!

---

**Happy Learning! 🚀**

For questions or issues, check the troubleshooting section above or examine the code comments for detailed explanations.
