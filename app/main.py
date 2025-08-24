from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
from dotenv import load_dotenv
import os

# Load environment variables first
load_dotenv()

from chains.graph import build_graph
from api.routes import router

app = FastAPI(title="Medical Literature Assistant API")

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Build and add LangGraph workflow
workflow = build_graph()
add_routes(app, workflow, path="/api/query")

# Additional API routes
app.include_router(router, prefix="/api")

@app.get("/health")
async def health():
    return {"status": "ok", "service": "medical-literature-assistant"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)