from fastapi import FastAPI
from pydantic import BaseModel, Field
from app.agents import run_agent_workflow

app = FastAPI(
    title="Multi-Agent LangGraph API",
    description="Planner → specialist agent → evaluator workflow for the multi-agent capstone.",
    version="1.0.0",
)

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=5000, description="Question or task for the multi-agent workflow")

class QueryResponse(BaseModel):
    query: str
    answer: str
    route: str
    trace: list[str]

@app.get("/")
def root():
    return {
        "message": "Multi-Agent LangGraph API is running",
        "docs": "/docs",
        "health": "/health",
    }

@app.get("/health")
def health():
    return {"status": "ok", "graph_loaded": True}

@app.post("/query", response_model=QueryResponse)
def query_agent(request: QueryRequest):
    result = run_agent_workflow(request.query.strip())
    return QueryResponse(
        query=request.query,
        answer=result["final_answer"],
        route=result["route"],
        trace=result["trace"],
    )
