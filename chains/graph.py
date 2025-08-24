from langgraph.graph import StateGraph
from langchain_groq import ChatGroq
from langfuse import Langfuse
import os
from app.models import GraphState, Paper
from .nodes_langfuse_v3 import (
    planner_node,
    fetch_arxiv_node,
    fetch_pubmed_node,
    normalize_node,
    dedupe_rank_node,
    embed_upsert_node,
    retrieve_node,
    extract_table_node,
    synthesize_node,
    cite_check_node,
    persist_node
)

def build_graph():
    """Build the LangGraph workflow for literature search and analysis."""
    
    # Initialize Langfuse for observability
    lf = Langfuse()
    
    # Create state graph
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("fetch_arxiv", fetch_arxiv_node)
    workflow.add_node("fetch_pubmed", fetch_pubmed_node)
    workflow.add_node("normalize", normalize_node)
    workflow.add_node("dedupe_rank", dedupe_rank_node)
    workflow.add_node("embed_upsert", embed_upsert_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("extract_table", extract_table_node)
    workflow.add_node("synthesize", synthesize_node)
    workflow.add_node("cite_check", cite_check_node)
    workflow.add_node("persist", persist_node)
    
    # Add edges
    workflow.add_edge("planner", "fetch_arxiv")
    workflow.add_edge("fetch_arxiv", "fetch_pubmed")
    workflow.add_edge("fetch_pubmed", "normalize")
    workflow.add_edge("normalize", "dedupe_rank")
    workflow.add_edge("dedupe_rank", "embed_upsert")
    workflow.add_edge("embed_upsert", "retrieve")
    workflow.add_edge("retrieve", "extract_table")
    workflow.add_edge("extract_table", "synthesize")
    workflow.add_edge("synthesize", "cite_check")
    workflow.add_edge("cite_check", "persist")
    
    # Set entry point
    workflow.set_entry_point("planner")
    workflow.set_finish_point("persist")
    
    return workflow.compile()