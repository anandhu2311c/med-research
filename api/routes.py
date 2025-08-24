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
            # Return a demo report for now
            demo_content = f"""# Literature Review: Demo Report

Generated: 2024-01-01T00:00:00

## Key Advances

Recent advances in AI for medical imaging have shown significant progress in automated diagnosis and treatment planning. Machine learning algorithms have demonstrated remarkable capabilities in detecting patterns in medical images that may be subtle or invisible to the human eye. These systems can process thousands of images in minutes, providing rapid screening capabilities that significantly reduce diagnostic delays.

The integration of deep learning with medical imaging has led to breakthrough applications in radiology, pathology, and ophthalmology. Convolutional neural networks (CNNs) have achieved diagnostic accuracy comparable to or exceeding that of experienced specialists in specific tasks such as diabetic retinopathy screening, skin cancer detection, and pneumonia diagnosis from chest X-rays.

## Methodological Trends

Deep learning approaches, particularly convolutional neural networks, have become the standard for medical image analysis. The field has witnessed a shift from traditional feature engineering to end-to-end learning systems that automatically extract relevant features from raw image data.

Transfer learning has emerged as a crucial technique, allowing models pre-trained on large natural image datasets to be fine-tuned for medical applications. This approach has proven particularly valuable given the limited availability of large, annotated medical datasets.

Multi-modal learning approaches are gaining traction, combining imaging data with electronic health records, genomic information, and clinical notes to provide more comprehensive diagnostic insights. Attention mechanisms and transformer architectures are being adapted from natural language processing to improve the interpretability and performance of medical image analysis systems.

## Clinical Outcomes

Studies show improved accuracy and reduced diagnosis time when AI tools are used as decision support systems. Clinical trials have demonstrated that AI-assisted diagnosis can reduce false positive rates by up to 30% while maintaining or improving sensitivity for disease detection.

The implementation of AI systems in clinical workflows has shown measurable improvements in patient outcomes, including earlier detection of critical conditions, reduced time to treatment, and more consistent diagnostic quality across different healthcare settings and practitioner experience levels.

Cost-effectiveness analyses indicate that AI-powered screening programs can significantly reduce healthcare costs by enabling earlier intervention and reducing the need for expensive follow-up procedures. Remote and underserved areas particularly benefit from AI-enabled diagnostic capabilities that can provide specialist-level analysis without requiring on-site expertise.

## Limitations

Current limitations include the need for large annotated datasets and concerns about model interpretability. The "black box" nature of deep learning models makes it challenging for clinicians to understand the reasoning behind AI recommendations, which can impact trust and adoption in clinical settings.

Data quality and standardization remain significant challenges, as models trained on data from one institution or imaging protocol may not generalize well to different settings. Bias in training data can lead to disparities in AI performance across different demographic groups, potentially exacerbating existing healthcare inequalities.

Regulatory approval processes for AI medical devices are still evolving, creating uncertainty about compliance requirements and approval timelines. Integration with existing hospital information systems and clinical workflows often requires significant technical and organizational changes that can be costly and time-consuming to implement.

## Future Directions

Future work should focus on federated learning approaches and explainable AI methods for clinical deployment. Federated learning will enable the development of more robust models by training on distributed datasets while preserving patient privacy and institutional data governance requirements.

The development of standardized evaluation frameworks and benchmarks will be crucial for comparing AI systems and ensuring consistent performance across different clinical environments. Real-world evidence generation through post-market surveillance and continuous learning systems will help validate AI performance in diverse clinical settings.

Integration with emerging technologies such as augmented reality, robotics, and precision medicine platforms will create new opportunities for AI-enhanced healthcare delivery. The focus will shift toward developing AI systems that not only diagnose but also recommend personalized treatment strategies and predict treatment outcomes.

---
*This is a comprehensive demo report showcasing the full capabilities of the Medical Literature Assistant. Report ID: {report_id}*
"""
            return {"id": report_id, "content": demo_content}
        
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
            else:
                # Generate demo PDF
                demo_content = """# Literature Review: Demo Report

Generated: 2024-01-01T00:00:00

## Key Advances

Recent advances in AI for medical imaging have shown significant progress in automated diagnosis and treatment planning.

## Methodological Trends

Deep learning approaches, particularly convolutional neural networks, have become the standard for medical image analysis.

## Clinical Outcomes

Studies show improved accuracy and reduced diagnosis time when AI tools are used as decision support systems.

## Limitations

Current limitations include the need for large annotated datasets and concerns about model interpretability.

## Future Directions

Future work should focus on federated learning approaches and explainable AI methods for clinical deployment.
"""
                generate_pdf_report(demo_content, report_id, "Demo Literature Review")
        
        # Return PDF file
        return FileResponse(
            path=pdf_path,
            filename=f"literature_review_{report_id}.pdf",
            media_type="application/pdf"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating PDF: {str(e)}")

@router.get("/papers/search")
async def search_papers(q: str, source: str = "all"):
    """Quick paper search without RAG processing."""
    # Mock response for now
    return {
        "papers": [
            {
                "id": "mock:1",
                "title": f"Sample paper about {q}",
                "authors": ["Author One"],
                "year": 2024,
                "source": "arxiv"
            }
        ]
    }

@router.post("/feedback")
async def submit_feedback(feedback: dict):
    """Submit user feedback for a report."""
    # In production, store in database and link to Langfuse trace
    print(f"Feedback received: {feedback}")
    return {"status": "received"}

@router.get("/trace/{trace_id}")
async def get_trace(trace_id: str):
    """Get Langfuse trace link for debugging."""
    langfuse_host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    return {"trace_url": f"{langfuse_host}/trace/{trace_id}"}