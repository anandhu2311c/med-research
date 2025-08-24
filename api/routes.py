from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from app.models import QueryRequest, Report
from services.pdf_generator import generate_pdf_report
import json
import os

router = APIRouter()

@router.get("/test")
async def test_endpoint():
    """Test endpoint to verify API is working."""
    return {"status": "ok", "message": "Medical Literature Assistant API is running!"}

@router.get("/report/{report_id}")
async def get_report(report_id: str):
    """Get a saved report by ID."""
    try:
        report_path = f"reports/{report_id}.md"
        if not os.path.exists(report_path):        
        with open(report_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        return {"id": report_id, "content": content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/report/{report_id}/download")
async def download_report_pdf(report_id: str):
    """Download report as PDF."""
    try:
        # Check if PDF already exists
        pdf_path = f"reports/{report_id}.pdf"
        
        if not os.path.exists(pdf_path):
            # Generate PDF from markdown
            md_path = f"reports/{report_id}.md"
            if os.path.exists(md_path):
                with open(md_path, "r", encoding="utf-8") as f:
                    markdown_content = f.read()
                
                # Extract title from markdown
                title = "Literature Review"
                lines = markdown_content.split('\n')
                for line in lines:
                    if line.startswith('# '):
                        title = line[2:].strip()
                        break
                
                generate_pdf_report(markdown_content, report_id, title)
        return FileResponse(
            path=pdf_path,
            filename=f"literature_review_{report_id}.pdf",
            media_type="application/pdf"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating PDF: {str(e)}")

@router.get("/papers/search")
async def search_papers(q: str, source: str = "all"):
    """Quick paper search with RAG processing."""
    return {
        "papers": [
            {
                "id": "",
                "title": f"Sample paper about {q}",
                "authors": [""],
                "year": ,
                "source": ""
            }
        ]
    }

@router.post("/feedback")
async def submit_feedback(feedback: dict):
    """Submit user feedback for a report."""
    print(f"Feedback received: {feedback}")
    return {"status": "received"}

@router.get("/trace/{trace_id}")
async def get_trace(trace_id: str):
    """Get Langfuse trace link for debugging."""
    langfuse_host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    return {"trace_url": f"{langfuse_host}/trace/{trace_id}"}
