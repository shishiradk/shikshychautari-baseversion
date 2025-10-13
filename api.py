from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import uuid
import tempfile
import boto3

load_dotenv()
from utils import (
    extract_text_from_pdfs, 
    chunk_text, 
    create_vector_store, 
    load_vector_store, 
    generate_predicted_paper, 
    paper_to_pdf_bytes, 
    REPORTLAB_OK,
    strip_headers_and_footers,
    clean_body_keep_all_marks,
    enforce_structure,
    split_group_instructions,
    enforce_pom1_style,
    center_sections
)

# Pydantic Models
class GeneratePaperRequest(BaseModel):
    bucket_name: str
    old_question_prefix: str
    syllabus_prefix: str
    predicted_question_prefix: str
    additional_instructions: Optional[str] = None

# FastAPI app initialization
app = FastAPI(
    title="🎓 Question Paper Generator API",
    description="API for generating structured question papers with POM1 style formatting", 
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001", 
        "http://localhost:3002",
        "http://localhost:3003",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "http://127.0.0.1:3002", 
        "http://127.0.0.1:3003"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# S3 Utility functions
def get_s3_client():
    """Get S3 client with credentials"""
    return boto3.client(
        's3',
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION", "eu-north-1")
    )

def download_pdfs_from_s3(bucket_name: str, prefix: str) -> List[bytes]:
    """Download all PDF files from S3 bucket with given prefix"""
    s3_client = get_s3_client()
    pdf_contents = []
    temp_files = []  # Keep track of temp files for cleanup
    
    try:
        # List objects with the prefix
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        
        if 'Contents' not in response:
            raise HTTPException(status_code=404, detail=f"No files found in bucket {bucket_name} with prefix {prefix}")
        
        # Download each PDF file
        for obj in response['Contents']:
            key = obj['Key']
            if key.lower().endswith('.pdf'):
                print(f"📥 Downloading: {key}")
                
                # Create temporary file
                tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
                temp_files.append(tmp_file.name)
                tmp_file.close()
                
                try:
                    # Download file to temporary location
                    s3_client.download_file(bucket_name, key, tmp_file.name)
                    
                    # Read the file content
                    with open(tmp_file.name, 'rb') as f:
                        pdf_content = f.read()
                        pdf_contents.append(pdf_content)
                        
                except Exception as e:
                    print(f"⚠️ Error processing {key}: {str(e)}")
                    continue
        
        if not pdf_contents:
            raise HTTPException(status_code=404, detail=f"No PDF files found in bucket {bucket_name} with prefix {prefix}")
        
        print(f"✅ Downloaded {len(pdf_contents)} PDF files from S3")
        return pdf_contents
        
    except Exception as e:
        if "NoSuchBucket" in str(e):
            raise HTTPException(status_code=404, detail=f"Bucket {bucket_name} not found")
        elif "AccessDenied" in str(e):
            raise HTTPException(status_code=403, detail=f"Access denied to bucket {bucket_name}")
        else:
            raise HTTPException(status_code=500, detail=f"Error downloading from S3: {str(e)}")
    
    finally:
        # Clean up all temporary files
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception as e:
                print(f"⚠️ Could not delete temp file {temp_file}: {str(e)}")
                # Continue with cleanup even if one file fails

def upload_pdf_to_s3(pdf_bytes: bytes, bucket_name: str, output_prefix: str, filename: str) -> str:
    """Upload PDF to S3 and return the full S3 key"""
    s3_client = get_s3_client()
    
    # Create the full S3 key
    s3_key = f"{output_prefix.rstrip('/')}/{filename}"
    
    try:
        s3_client.put_object(
            Bucket=bucket_name,
            Key=s3_key,
            Body=pdf_bytes,
            ContentType='application/pdf'
        )
        
        print(f"✅ Uploaded PDF to S3: {s3_key}")
        return s3_key
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading to S3: {str(e)}")

# API Endpoints
@app.get("/", tags=["Info"])
async def root():
   
    return {
        "welcome to question paper generator api": "http://localhost:8000/swagger"
    }

 
@app.post("/generate-paper-s3", 
           tags=["☁️ S3 Question Paper Generation"],
            summary="☁️ Generate Structured Question Paper from S3 (Old Questions Style)",
           description="Download PDFs from S3 (old questions + syllabus), generate structured question paper with POM1 style formatting and preserved marks, then upload result back to S3",
           response_description="Success message with S3 location of generated structured PDF")
async def generate_paper_from_s3(
    data: GeneratePaperRequest
):
 
    try:
        # Generate unique session ID for this request
        session_id = str(uuid.uuid4())
        
        print(f"🚀 Starting S3-based question paper generation...")
        print(f"📦 Bucket: {data.bucket_name}")
        print(f"📥 Old Question prefix: {data.old_question_prefix}")
        print(f"📚 Syllabus prefix: {data.syllabus_prefix}")
        print(f"📤 Predicted Question prefix: {data.predicted_question_prefix}")
        
        # Step 1: Download old question PDFs from S3
        print("📥 Downloading old question PDFs from S3...")
        old_question_pdfs = download_pdfs_from_s3(data.bucket_name, data.old_question_prefix)
        
        # Step 2: Download syllabus PDFs from S3
        print("📚 Downloading syllabus PDFs from S3...")
        syllabus_pdfs = download_pdfs_from_s3(data.bucket_name, data.syllabus_prefix)
        
       
        
        # Step 3: Extract text from syllabus PDFs
        print("🔍 Extracting text from syllabus PDFs...")
        syllabus_text = extract_text_from_pdfs(syllabus_pdfs)
        if not syllabus_text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from syllabus PDF files")

        # Step 4: Extract text from old question PDFs
        print("🔍 Extracting text from old question PDFs...")
        old_questions_text = extract_text_from_pdfs(old_question_pdfs)
        if not old_questions_text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from old question PDF files")
        
        # Step 5: Apply structured text processing pipeline  
        print("🔧 Applying structured text processing pipeline...")
        
        # Process old questions text with the same pipeline as Streamlit app
        processed_text = strip_headers_and_footers(old_questions_text)
        processed_text = clean_body_keep_all_marks(processed_text)
        
        # Step 6: Generate question paper
        print("🤖 Generating predicted question paper...")
        chunks = chunk_text(processed_text)
        vector_store_path = create_vector_store(chunks, session_id)
        
        past_db = load_vector_store(session_id)
        raw_paper, past_context = generate_predicted_paper(
            past_db, 
            syllabus_text
        )
        
        # Step 7: Apply structured formatting pipeline (same as Streamlit app)
        print("🎨 Applying structured formatting (POM1 style)...")
        
        # Apply the same processing pipeline as in the Streamlit app
        paper = strip_headers_and_footers(raw_paper)
        paper = clean_body_keep_all_marks(paper)
        paper = enforce_structure(paper)
        paper = split_group_instructions(paper)
        paper = enforce_pom1_style(paper)
        paper = center_sections(paper)
        
        # Step 8: Generate PDF with structured formatting
        if not REPORTLAB_OK:
            raise HTTPException(
                status_code=501, 
                detail="PDF generation not available. Install reportlab: pip install reportlab"
            )
        
        # Extract subject name from the first syllabus file key
        first_syllabus_key = None
        try:
            s3_client = get_s3_client()
            response = s3_client.list_objects_v2(Bucket=data.bucket_name, Prefix=data.syllabus_prefix)
            if 'Contents' in response and response['Contents']:
                first_syllabus_key = response['Contents'][0]['Key']
                subject_name = os.path.splitext(os.path.basename(first_syllabus_key))[0]
            else:
                subject_name = "Predicted Question Paper"
        except Exception:
            subject_name = "Predicted Question Paper"
        
        pdf_bytes = paper_to_pdf_bytes(paper, subject_name)
        
        # Step 9: Upload to S3 using the provided predicted question prefix
        print("📤 Uploading structured PDF to S3...")
        timestamp = session_id[:8]
        filename = f"structured_paper_{timestamp}.pdf"
        s3_key = upload_pdf_to_s3(pdf_bytes, data.bucket_name, data.predicted_question_prefix, filename)
        
        # Step 10: Clean up
        try:
            import shutil
            if os.path.exists(vector_store_path):
                shutil.rmtree(vector_store_path)
        except Exception:
            pass
        
        # Step 11: Return success response
        return JSONResponse(
            content={
                "success": True,
                "message": "Structured question paper generated and uploaded successfully!",
                "s3_bucket": data.bucket_name,
                "s3_key": s3_key,
                "s3_url": f"https://{data.bucket_name}.s3.amazonaws.com/{s3_key}",
                "file_size": len(pdf_bytes),
                "session_id": session_id,
                "predicted_question_prefix": data.predicted_question_prefix,
                "subject_name": subject_name,
                "formatting": "Old Questions with Preserved Marks"
            },
            status_code=200
        )
        
    except HTTPException:
        raise
    except Exception as e:
        # Clean up on error
        try:
            import shutil
            vector_store_path = f"faiss_index_{session_id}"
            if os.path.exists(vector_store_path):
                shutil.rmtree(vector_store_path)
        except Exception:
            pass
        
        raise HTTPException(status_code=500, detail=f"Error in S3-based generation: {str(e)}")

 
if __name__ == "__main__":
    import uvicorn
    
    # Check if required environment variables are set
    missing_vars = []
    
    if not os.getenv("OPENAI_API_KEY"):
        missing_vars.append("OPENAI_API_KEY")
    
    if not os.getenv("AWS_ACCESS_KEY_ID"):
        missing_vars.append("AWS_ACCESS_KEY_ID")
    
    if not os.getenv("AWS_SECRET_ACCESS_KEY"):
        missing_vars.append("AWS_SECRET_ACCESS_KEY")
    
    if missing_vars:
        print("⚠️  Warning: Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\n   Please set these in a .env file or environment variables")
        print("   Example .env file:")
        print("   OPENAI_API_KEY=your_openai_key")
        print("   AWS_ACCESS_KEY_ID=your_aws_key")
        print("   AWS_SECRET_ACCESS_KEY=your_aws_secret")
        print("   AWS_REGION=eu-north-1")
        exit(1)
    
    print("🚀 Starting Question Paper Generator API...")
    print("📚 Swagger UI: http://localhost:8000/swagger")
    print("🔗 API Root: http://localhost:8000/")
    print("=" * 50)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
 