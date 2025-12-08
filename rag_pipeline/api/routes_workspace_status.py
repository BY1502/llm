from fastapi import APIRouter, HTTPException
from ..core.workspace import WORKSPACES

router = APIRouter()

@router.get("/workspace/status/{wid}")
def status(wid: str):
    ws = WORKSPACES.get(wid)
    if not ws:
        raise HTTPException(404, "workspace not found")
    return {
        "workspace_id": wid,
        "ready": ws.ready,
        "error": ws.error,
        "last_updated": ws.last_updated,
        "files": ws.files,
    }