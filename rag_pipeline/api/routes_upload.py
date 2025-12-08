from fastapi import APIRouter, UploadFile, File, Query, HTTPException
from time import perf_counter, sleep
from rag_pipeline.core.workspace import create_workspace
from rag_pipeline.core.processing import start_processing
from rag_pipeline.core.workspace import WORKSPACES

router = APIRouter()


@router.post("/upload")
async def upload_files(files: list[UploadFile] = File(...),
                       wait: bool = Query(True),
                       timeout: int = Query(60)):
    ws = await create_workspace(files) # saving -> indexing
    start_processing(ws)
    if not wait:
        return {"workspace_id": ws.id, "status": ws.status, "message": "업로드 시작"}
    t0 = perf_counter()
    while (perf_counter() - t0) < timeout:
        if ws.status == "ready":
            return {"workspace_id": ws.id, "status": "ready","files": ws.files , "message": "업로드 및 인덱싱 완료"}
        if ws.status == "error":
            raise HTTPException(400, f"업로드/인덱싱 실패: {ws.error}")
        sleep(0.3)
    return {"workspace_id": ws.id, "status": ws.status, "message": "처리 중입니다..."}
    # workspace = await create_workspace(files)
    # start_processing(workspace)
    # return {"workspace_id": workspace.id, "message": "처리 중입니다..."}

