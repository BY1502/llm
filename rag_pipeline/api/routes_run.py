import logging
import asyncio
from fastapi import APIRouter, HTTPException

from rag_pipeline.api.models import RunRequest, RunResponse
from rag_pipeline.core.pipeline import run_pipeline
from rag_pipeline.core.workspace import WORKSPACES

logger = logging.getLogger(__name__)

router = APIRouter()
@router.post("/run", response_model=RunResponse)
def run(req: RunRequest):
    logger.debug("workspace_id=%r (type=%s)", req.workspace_id, type(req.workspace_id))

    wid = req.workspace_id
    if isinstance(wid, str):
        normalized = wid.strip().lower()
        if normalized in ("", "null", "none", "undefined"):
            wid = None

    req.workspace_id = wid    
    if False and req.mode == "uploaded":
        if wid and wid not in WORKSPACES:
            if wid not in WORKSPACES:
                raise HTTPException(400, "유효하지 않은 workspace_id 입니다.")        
    try:
        result = run_pipeline(req)
        logger.debug("Pipeline result type: %s", type(result))
        logger.debug("Pipeline result payload: %s", result)
        return result
    except HTTPException as e:
        logger.warning("HTTPException in pipeline: %s - %s", e.status_code, e.detail)
        raise
    except Exception as e:
        logger.exception("Unexpected error in pipeline")
        raise HTTPException(500, f"Internal error: {str(e)}")
