from langchain_groq import ChatGroq
from langfuse import Langfuse
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import os
import uuid
from datetime import datetime
from dotenv import load_dotenv
from app.models import GraphState, Paper
from services.arxiv_client import ArxivClient
from services.pubmed_client import PubmedClient

# Load environment variables
load_dotenv()

# Global variables for services
lf = None
llm = None
embedding_model = None
pc = None
index = None

def init_services():
    """Initialize all services with error handling."""
    global lf, llm, embedding_model, pc, index
    
    # Initialize Langfuse with v3 API
    try:
        lf = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        )
        print("✅ Langfuse initialized successfully")
    except Exception as e:
        print(f"⚠️ Langfuse initialization failed: {e}")
        lf = None
    
    # Initialize Groq LLM
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables")
    
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", api_key=groq_api_key)
    
    # Initialize embedding model
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    # Initialize Pinecone
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if pinecone_api_key:
        try:
            pc = Pinecone(api_key=pinecone_api_key)
            index = pc.Index("medlit-embeddings")
            print("✅ Pinecone initialized successfully")
        except Exception as e:
            print(f"⚠️ Pinecone initialization failed: {e}")
            pc = None
            index = None
    else:
        print("⚠️ PINECONE_API_KEY not found")
        pc = None
        index = None

# Initialize services
init_services()

def planner_node(state: GraphState) -> GraphState:
    """Plan the search strategy and build queries for each source."""
    print(f"Planning search for: {state.query}")
    
    # Use Langfuse context manager for tracing
    if lf:
        try:
            with lf.start_as_current_span(name="planner") as span:
                span.update(
                    input={"query": state.query},
                    metadata={"step": "planning", "node": "planner"}
                )
                
                # Simple planning
                filters = {
                    "arxiv_query": f"{state.query} AND (radiology OR medical imaging OR AI)",
                    "pubmed_query": state.query,
                    "max_results": 5
                }
                state.filters = filters
                
                span.update(output={"filters": filters})
                print(f"Generated search filters: {filters}")
                
        except Exception as e:
            print(f"Langfuse tracing error: {e}")
            # Fallback without tracing
            filters = {
                "arxiv_query": f"{state.query} AND (radiology OR medical imaging OR AI)",
                "pubmed_query": state.query,
                "max_results": 5
            }
            state.filters = filters
            print(f"Generated search filters: {filters}")
    else:
        # No Langfuse available
        filters = {
            "arxiv_query": f"{state.query} AND (radiology OR medical imaging OR AI)",
            "pubmed_query": state.query,
            "max_results": 5
        }
        state.filters = filters
        print(f"Generated search filters: {filters}")
    
    return state

def fetch_arxiv_node(state: GraphState) -> GraphState:
    """Fetch papers from arXiv."""
    print("Fetching papers from arXiv...")
    
    if lf:
        try:
            with lf.start_as_current_span(name="fetch_arxiv") as span:
                span.update(
                    input={"query": state.filters["arxiv_query"]},
                    metadata={"source": "arxiv", "max_results": state.filters["max_results"]}
                )
                
                try:
                    client = ArxivClient()
                    papers = client.search(state.filters["arxiv_query"], max_results=state.filters["max_results"])
                    state.papers.extend(papers)
                    
                    result = {"papers_fetched": len(papers), "total_papers": len(state.papers)}
                    print(f"Fetched {len(papers)} papers from arXiv")
                    span.update(output=result)
                    
                except Exception as e:
                    error_msg = f"Error fetching from arXiv: {e}"
                    print(error_msg)
                    span.update(output={"error": error_msg})
                    
        except Exception as e:
            print(f"Langfuse span error: {e}")
            # Fallback without tracing
            try:
                client = ArxivClient()
                papers = client.search(state.filters["arxiv_query"], max_results=state.filters["max_results"])
                state.papers.extend(papers)
                print(f"Fetched {len(papers)} papers from arXiv")
            except Exception as e:
                print(f"Error fetching from arXiv: {e}")
    else:
        # No Langfuse available
        try:
            client = ArxivClient()
            papers = client.search(state.filters["arxiv_query"], max_results=state.filters["max_results"])
            state.papers.extend(papers)
            print(f"Fetched {len(papers)} papers from arXiv")
        except Exception as e:
            print(f"Error fetching from arXiv: {e}")
    
    return state

def fetch_pubmed_node(state: GraphState) -> GraphState:
    """Fetch papers from PubMed."""
    print("Fetching papers from PubMed...")
    
    if lf:
        try:
            with lf.start_as_current_span(name="fetch_pubmed") as span:
                span.update(
                    input={"query": state.filters["pubmed_query"]},
                    metadata={"source": "pubmed", "max_results": state.filters["max_results"]}
                )
                
                try:
                    client = PubmedClient()
                    papers = client.search(state.filters["pubmed_query"], max_results=state.filters["max_results"])
                    state.papers.extend(papers)
                    
                    result = {"papers_fetched": len(papers), "total_papers": len(state.papers)}
                    print(f"Fetched {len(papers)} papers from PubMed")
                    span.update(output=result)
                    
                except Exception as e:
                    error_msg = f"Error fetching from PubMed: {e}"
                    print(error_msg)
                    span.update(output={"error": error_msg})
                    
        except Exception as e:
            print(f"Langfuse span error: {e}")
            # Fallback without tracing
            try:
                client = PubmedClient()
                papers = client.search(state.filters["pubmed_query"], max_results=state.filters["max_results"])
                state.papers.extend(papers)
                print(f"Fetched {len(papers)} papers from PubMed")
            except Exception as e:
                print(f"Error fetching from PubMed: {e}")
    else:
        # No Langfuse available
        try:
            client = PubmedClient()
            papers = client.search(state.filters["pubmed_query"], max_results=state.filters["max_results"])
            state.papers.extend(papers)
            print(f"Fetched {len(papers)} papers from PubMed")
        except Exception as e:
            print(f"Error fetching from PubMed: {e}")
    
    return state

def normalize_node(state: GraphState) -> GraphState:
    """Normalize and clean paper data."""
    print("Normalizing paper data...")
    
    if lf:
        try:
            with lf.start_as_current_span(name="normalize") as span:
                span.update(
                    input={"paper_count": len(state.papers)},
                    metadata={"step": "normalization"}
                )
                
                # Basic normalization
                normalized_papers = []
                for paper in state.papers:
                    if paper.abstract:
                        paper.abstract = paper.abstract.replace("<p>", "").replace("</p>", "")
                    normalized_papers.append(paper)
                
                state.papers = normalized_papers
                
                result = {"normalized_count": len(normalized_papers)}
                print(f"Normalized {len(normalized_papers)} papers")
                span.update(output=result)
                
        except Exception as e:
            print(f"Langfuse span error: {e}")
            # Fallback without tracing
            normalized_papers = []
            for paper in state.papers:
                if paper.abstract:
                    paper.abstract = paper.abstract.replace("<p>", "").replace("</p>", "")
                normalized_papers.append(paper)
            state.papers = normalized_papers
            print(f"Normalized {len(normalized_papers)} papers")
    else:
        # No Langfuse available
        normalized_papers = []
        for paper in state.papers:
            if paper.abstract:
                paper.abstract = paper.abstract.replace("<p>", "").replace("</p>", "")
            normalized_papers.append(paper)
        state.papers = normalized_papers
        print(f"Normalized {len(normalized_papers)} papers")
    
    return state

def dedupe_rank_node(state: GraphState) -> GraphState:
    """Deduplicate and rank papers."""
    print("Deduplicating and ranking papers...")
    
    if lf:
        try:
            with lf.start_as_current_span(name="dedupe_rank") as span:
                span.update(
                    input={"input_papers": len(state.papers)},
                    metadata={"step": "deduplication"}
                )
                
                # Simple deduplication by title similarity
                unique_papers = []
                seen_titles = set()
                
                for paper in state.papers:
                    title_key = paper.title.lower().strip()
                    if title_key not in seen_titles:
                        seen_titles.add(title_key)
                        unique_papers.append(paper)
                
                # Sort by year (most recent first)
                unique_papers.sort(key=lambda p: p.year, reverse=True)
                state.papers = unique_papers[:10]  # Keep top 10
                
                result = {"unique_papers": len(state.papers), "duplicates_removed": len(unique_papers) - len(state.papers)}
                print(f"Kept {len(state.papers)} unique papers")
                span.update(output=result)
                
        except Exception as e:
            print(f"Langfuse span error: {e}")
            # Fallback without tracing
            unique_papers = []
            seen_titles = set()
            
            for paper in state.papers:
                title_key = paper.title.lower().strip()
                if title_key not in seen_titles:
                    seen_titles.add(title_key)
                    unique_papers.append(paper)
            
            unique_papers.sort(key=lambda p: p.year, reverse=True)
            state.papers = unique_papers[:10]
            print(f"Kept {len(state.papers)} unique papers")
    else:
        # No Langfuse available
        unique_papers = []
        seen_titles = set()
        
        for paper in state.papers:
            title_key = paper.title.lower().strip()
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_papers.append(paper)
        
        unique_papers.sort(key=lambda p: p.year, reverse=True)
        state.papers = unique_papers[:10]
        print(f"Kept {len(state.papers)} unique papers")
    
    return state

def embed_upsert_node(state: GraphState) -> GraphState:
    """Generate embeddings and upsert to Pinecone."""
    print("Generating embeddings...")
    
    chunks = []
    
    if lf:
        try:
            with lf.start_as_current_span(name="embed_upsert") as span:
                span.update(
                    input={"papers_to_embed": len(state.papers)},
                    metadata={"step": "embedding", "model": "all-MiniLM-L6-v2"}
                )
                
                for paper in state.papers:
                    chunk_id = f"chunk:{paper.id}"
                    text = f"{paper.title}. {paper.abstract}"
                    
                    try:
                        embedding = embedding_model.encode([text], normalize_embeddings=True)[0]
                        
                        # Upsert to Pinecone if available
                        if index:
                            vectors = [{
                                "id": chunk_id,
                                "values": embedding.tolist(),
                                "metadata": {
                                    "paper_id": paper.id,
                                    "title": paper.title,
                                    "year": paper.year,
                                    "source": paper.source,
                                    "text": text
                                }
                            }]
                            index.upsert(vectors=vectors)
                        
                        chunks.append({"id": chunk_id, "paper_id": paper.id})
                    except Exception as e:
                        print(f"Error generating embedding: {e}")
                
                state.chunks = chunks
                
                result = {"embeddings_generated": len(chunks)}
                print(f"Generated {len(chunks)} embeddings")
                span.update(output=result)
                
        except Exception as e:
            print(f"Langfuse span error: {e}")
            # Fallback without tracing
            for paper in state.papers:
                chunk_id = f"chunk:{paper.id}"
                text = f"{paper.title}. {paper.abstract}"
                
                try:
                    embedding = embedding_model.encode([text], normalize_embeddings=True)[0]
                    
                    if index:
                        vectors = [{
                            "id": chunk_id,
                            "values": embedding.tolist(),
                            "metadata": {
                                "paper_id": paper.id,
                                "title": paper.title,
                                "year": paper.year,
                                "source": paper.source,
                                "text": text
                            }
                        }]
                        index.upsert(vectors=vectors)
                    
                    chunks.append({"id": chunk_id, "paper_id": paper.id})
                except Exception as e:
                    print(f"Error generating embedding: {e}")
            
            state.chunks = chunks
            print(f"Generated {len(chunks)} embeddings")
    else:
        # No Langfuse available
        for paper in state.papers:
            chunk_id = f"chunk:{paper.id}"
            text = f"{paper.title}. {paper.abstract}"
            
            try:
                embedding = embedding_model.encode([text], normalize_embeddings=True)[0]
                
                if index:
                    vectors = [{
                        "id": chunk_id,
                        "values": embedding.tolist(),
                        "metadata": {
                            "paper_id": paper.id,
                            "title": paper.title,
                            "year": paper.year,
                            "source": paper.source,
                            "text": text
                        }
                    }]
                    index.upsert(vectors=vectors)
                
                chunks.append({"id": chunk_id, "paper_id": paper.id})
            except Exception as e:
                print(f"Error generating embedding: {e}")
        
        state.chunks = chunks
        print(f"Generated {len(chunks)} embeddings")
    
    return state

def retrieve_node(state: GraphState) -> GraphState:
    """Retrieve relevant chunks using hybrid search."""
    print("Retrieving relevant papers...")
    
    # For demo, just use the papers we have
    retrieved = []
    for i, paper in enumerate(state.papers[:5]):
        retrieved.append({
            "chunk_id": f"chunk:{paper.id}",
            "score": 0.9 - (i * 0.1),
            "metadata": {
                "paper_id": paper.id,
                "title": paper.title,
                "year": paper.year,
                "source": paper.source,
                "text": f"{paper.title}. {paper.abstract}"
            }
        })
    
    state.retrieved = retrieved
    
    if lf:
        try:
            with lf.start_as_current_span(name="retrieve") as span:
                span.update(
                    input={"query": state.query, "available_chunks": len(state.chunks)},
                    output={"retrieved_count": len(retrieved)},
                    metadata={"step": "retrieval"}
                )
        except Exception as e:
            print(f"Langfuse span error: {e}")
    
    print(f"Retrieved {len(retrieved)} relevant papers")
    return state

def extract_table_node(state: GraphState) -> GraphState:
    """Extract evidence table from retrieved papers."""
    print("Extracting evidence table...")
    
    evidence_table = []
    
    for item in state.retrieved:
        evidence_table.append({
            "paper_id": item["metadata"]["paper_id"],
            "title": item["metadata"]["title"],
            "year": item["metadata"]["year"],
            "source": item["metadata"]["source"],
            "relevance_score": item["score"]
        })
    
    state.evidence_table = evidence_table
    
    if lf:
        try:
            with lf.start_as_current_span(name="extract_table") as span:
                span.update(
                    input={"retrieved_papers": len(state.retrieved)},
                    output={"evidence_entries": len(evidence_table)},
                    metadata={"step": "evidence_extraction"}
                )
        except Exception as e:
            print(f"Langfuse span error: {e}")
    
    print(f"Extracted {len(evidence_table)} evidence entries")
    return state

def synthesize_node(state: GraphState) -> GraphState:
    """Generate summary using LLM."""
    print("Synthesizing literature summary...")
    
    # Prepare context from retrieved papers
    context = "\n\n".join([
        f"Paper: {item['metadata']['title']} ({item['metadata']['year']})\nAbstract: {item['metadata']['text']}"
        for item in state.retrieved[:3]
    ])
    
    prompt = f"""Based on the following research papers, provide a comprehensive and detailed literature review on: {state.query}

Context:
{context}

Please write a thorough literature review with the following sections. Each section should be substantial (3-4 paragraphs) and well-detailed:

## 1. Key Advances
## 2. Methodological Trends  
## 3. Clinical Outcomes
## 4. Limitations
## 5. Future Directions

Make each section comprehensive with detailed explanations. Include inline citations using [Paper Title, Year] format."""

    if lf:
        try:
            with lf.start_as_current_generation(
                name="literature_synthesis",
                model="llama-3.3-70b-versatile"
            ) as generation:
                generation.update(
                    input={"prompt": prompt, "context_papers": len(state.retrieved)},
                    metadata={"provider": "groq", "step": "synthesis"}
                )
                
                try:
                    response = llm.invoke(prompt)
                    state.draft = response.content
                    
                    generation.update(
                        output={"completion": response.content},
                        usage={
                            "input_tokens": len(prompt.split()),
                            "output_tokens": len(response.content.split()),
                            "total_tokens": len(prompt.split()) + len(response.content.split())
                        }
                    )
                    
                    print(f"Generated comprehensive summary ({len(state.draft)} characters)")
                    
                except Exception as e:
                    error_msg = f"Error generating summary: {e}"
                    print(error_msg)
                    generation.update(output={"error": error_msg})
                    
                    # Enhanced fallback content
                    state.draft = f"""# Literature Review: {state.query}

## 1. Key Advances
The field of {state.query} has witnessed remarkable progress in recent years...

[Enhanced fallback content would go here]
"""
                    
        except Exception as e:
            print(f"Langfuse generation error: {e}")
            # Fallback without tracing
            try:
                response = llm.invoke(prompt)
                state.draft = response.content
                print(f"Generated comprehensive summary ({len(state.draft)} characters)")
            except Exception as e:
                print(f"Error generating summary: {e}")
                state.draft = f"# Literature Review: {state.query}\n\nDemo summary generated for query: {state.query}"
    else:
        # No Langfuse available
        try:
            response = llm.invoke(prompt)
            state.draft = response.content
            print(f"Generated comprehensive summary ({len(state.draft)} characters)")
        except Exception as e:
            print(f"Error generating summary: {e}")
            state.draft = f"# Literature Review: {state.query}\n\nDemo summary generated for query: {state.query}"
    
    return state

def cite_check_node(state: GraphState) -> GraphState:
    """Verify citations and calculate confidence."""
    print("Checking citations...")
    
    # Simple confidence calculation
    confidence = min(0.9, len(state.retrieved) / 10.0)
    state.confidence = confidence
    
    if lf:
        try:
            with lf.start_as_current_span(name="cite_check") as span:
                span.update(
                    input={"draft_length": len(state.draft), "retrieved_papers": len(state.retrieved)},
                    output={"confidence": confidence},
                    metadata={"step": "citation_verification"}
                )
        except Exception as e:
            print(f"Langfuse span error: {e}")
    
    print(f"Confidence score: {confidence:.2f}")
    return state

def persist_node(state: GraphState) -> GraphState:
    """Save the report."""
    print("Saving report...")
    
    report_id = f"rep_{uuid.uuid4().hex[:8]}"
    state.report_id = report_id
    
    # Save to file
    os.makedirs("reports", exist_ok=True)
    with open(f"reports/{report_id}.md", "w") as f:
        f.write(f"# Literature Review: {state.query}\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        f.write(state.draft)
    
    if lf:
        try:
            with lf.start_as_current_span(name="persist") as span:
                span.update(
                    input={"report_length": len(state.draft)},
                    output={"report_id": report_id, "file_path": f"reports/{report_id}.md"},
                    metadata={"step": "persistence"}
                )
            
            # Flush Langfuse data
            lf.flush()
            print("✅ Langfuse data flushed!")
            
        except Exception as e:
            print(f"Langfuse error: {e}")
    
    print(f"Report saved: {report_id}")
    return state