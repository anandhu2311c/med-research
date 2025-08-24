import httpx
from typing import List
from app.models import Paper

class ArxivClient:
    """Client for fetching papers from arXiv API."""
    
    def __init__(self):
        self.base_url = "https://export.arxiv.org/api/query"
    
    def search(self, query: str, max_results: int = 50) -> List[Paper]:
        """Search arXiv for papers matching the query."""
        params = {
            "search_query": query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending"
        }
        
        try:
            with httpx.Client() as client:
                response = client.get(self.base_url, params=params)
                response.raise_for_status()
                
                # Parse XML response (simplified)
                papers = self._parse_arxiv_response(response.text)
                return papers
        except Exception as e:
            print(f"Error fetching from arXiv: {e}")
            return []
    
    def _parse_arxiv_response(self, xml_content: str) -> List[Paper]:
        """Parse arXiv XML response into Paper objects."""
        # Simplified parsing - in production, use proper XML parser
        papers = []
        
        # Mock data for now
        papers.append(Paper(
            id="arxiv:2401.01234",
            title="AI in Medical Imaging: A Comprehensive Review",
            authors=["John Doe", "Jane Smith"],
            year=2024,
            source="arxiv",
            url="https://arxiv.org/abs/2401.01234",
            abstract="This paper reviews recent advances in AI for medical imaging...",
            modality=["CT", "MRI"],
            task=["detection", "segmentation"]
        ))
        
        return papers