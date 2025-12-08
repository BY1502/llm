from pathlib import Path
from uuid import uuid4
import time
from typing import Dict, List, Optional
from fastapi import UploadFile
from pathlib import Path
from rag_pipeline.config import ModelCfg

try:
    import chromadb  # type: ignore
    from langchain_chroma import Chroma  # type: ignore
    from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
except Exception:
    chromadb = None
    Chroma = None
    HuggingFaceEmbeddings = None

class Workspace:
    def __init__(self, wid: str, folder: Path):
        self.id = wid
        self.folder = folder
        self.ready = False
        self.error: str | None = None # 실패 사유 저장
        self.files: list[str] = [] # 업로드된 파일명 리스트
        self.status: str = "created"
        self.last_updated = time.time()
        

WORKSPACES: Dict[str, Workspace] = {}
DATA_STORE = Path("data_store")
DATA_STORE.mkdir(exist_ok=True)

async def create_workspace(files: List[UploadFile]) -> Workspace:
    wid = str(uuid4())[:8]
    folder = DATA_STORE / wid
    folder.mkdir(parents=True, exist_ok=True)
    ws = Workspace(wid, folder)
    WORKSPACES[wid] = ws
    
    ws.status = "saving"
    for f in files:
        dest = folder / f.filename
        # 파일 저장
        with open(dest, "wb") as out:
            out.write(await f.read())

        # ✅ 저장 직후 확인 루프 (Windows I/O 딜레이 방지)
        for _ in range(6):  # 약 0.3초 동안 6회 확인
            if dest.exists() and dest.stat().st_size > 0:
                break
            time.sleep(0.05)

        ws.files.append(dest.name)
    ws.status = "indexing"
    return ws