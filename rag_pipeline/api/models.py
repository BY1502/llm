from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel

class RunRequest(BaseModel):
    mode: Literal["folder", "uploaded"] = "uploaded"
    workspace_id: Optional[str] = None
    folder_path: Optional[str] = None
    query: Optional[str] = None
    k: int = 10 # K값
    final_k: int = 3 # 최종 K값
    retrieval: Literal["dense","sparse","hybrid"] = "hybrid" # RAG 방식 선택
    use_rerank: bool = True
    # LLM 모델 선택
    llm_choise: Optional[str] = None
    
    tags: Optional[List[str]] = None

# class RunResponse(BaseModel):
#     summary: str
#     sources: List[Dict[str, Any]]

class RunResponse(BaseModel):
    summary: str
    mode: Optional[str] = None          # ← pipeline에서 넣는 mode
    llm_model: Optional[str] = None     # ← pipeline에서 넣는 llm_model (llm_choise 로그용)
    json_data: Optional[Dict[str, Any]] = None  # ← 우리가 만든 JSON Auto-Schema 전체

    sources: List[Dict[str, Any]]       # 그대로 유지