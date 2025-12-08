from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from rag_pipeline.api.routes_run import router as run_router
from rag_pipeline.api.routes_upload import router as upload_router
from rag_pipeline.api.routes_workspace_status import router as status_router
# from rag_pipeline.core.processing import bootstrap_data_store
app = FastAPI(title="RAG Pipeline API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload_router, prefix="/api/workspace", tags=["workspace"])
app.include_router(run_router, prefix="/api", tags=["rag"])
app.include_router(status_router, prefix="/api", tags=["workspace"])

# bootstrap_data_store()

@app.get("/api/health")
def health():
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("rag_pipeline.main:app", host="0.0.0.0", port=8000, reload=True)
