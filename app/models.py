from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

class Paper(BaseModel):
    id: str
    title: str
    authors: List[str]
    year: int
    source: str  # "arxiv" | "pubmed" | "ctgov"
    doi: Optional[str] = None
    url: str
    abstract: str
    modality: List[str] = []  # ["CT", "MRI", "X-ray"]
    task: List[str] = []  # ["detection", "segmentation", "classification"]
    metrics: Dict[str, float] = {}  # {"auc": 0.93, "sens": 0.91}
    trial_phase: Optional[str] = None
    raw: Dict[str, Any] = {}

class Chunk(BaseModel):
    id: str
    paper_id: str
    text: str
    embedding: List[float]
    metadata: Dict[str, Any]

class Citation(BaseModel):
    paper_id: str
    snippet: str
    confidence: float = 1.0

class ReportSection(BaseModel):
    title: str
    content: str

class Report(BaseModel):
    id: str
    query: str
    created_at: datetime
    sections: List[ReportSection]
    citations: List[Citation]
    confidence: float
    files: Dict[str, str] = {}

class QueryRequest(BaseModel):
    query: str
    sources: Optional[List[str]] = ["arxiv", "pubmed", "ctgov"]
    from_date: Optional[str] = None
    to_date: Optional[str] = None
    modalities: Optional[List[str]] = None
    study_types: Optional[List[str]] = None
    k: int = 20

class GraphState(BaseModel):
    query: str
    filters: Dict[str, Any] = {}
    papers: List[Paper] = []
    chunks: List[Dict[str, str]] = []
    retrieved: List[Dict[str, Any]] = []
    evidence_table: List[Dict[str, Any]] = []
    draft: str = ""
    confidence: float = 0.0
    report_id: str = ""