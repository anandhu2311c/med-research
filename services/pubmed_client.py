import httpx
from typing import List
from app.models import Paper

class PubmedClient:
    """Client for fetching papers from PubMed API."""
    
    def __init__(self):
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    
    def search(self, query: str, max_results: int = 50) -> List[Paper]:
        """Search PubMed for papers matching the query."""
        try:
            # First, search for paper IDs
            search_url = f"{self.base_url}/esearch.fcgi"
            search_params = {
                "db": "pubmed",
                "term": query,
                "retmax": max_results,
                "retmode": "json"
            }
            
            with httpx.Client() as client:
                search_response = client.get(search_url, params=search_params)
                search_response.raise_for_status()
                
                # Parse search results and fetch details
                papers = self._fetch_paper_details(search_response.json())
                return papers
        except Exception as e:
            print(f"Error fetching from PubMed: {e}")
            return []
    
    def _fetch_paper_details(self, search_results: dict) -> List[Paper]:
        """Fetch detailed paper information from PubMed."""
        papers = []
        
        # Mock data for now
        papers.append(Paper(
            id="pubmed:38123456",
            title="Deep Learning for Radiology: Current Applications and Future Directions",
            authors=["Alice Johnson", "Bob Wilson"],
            year=2024,
            source="pubmed",
            doi="10.1001/jama.2024.1234",
            url="https://pubmed.ncbi.nlm.nih.gov/38123456/",
            abstract="Deep learning has revolutionized medical imaging analysis...",
            modality=["X-ray", "CT"],
            task=["classification", "detection"]
        ))
        
        return papers