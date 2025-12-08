
from dataclasses import dataclass
from pathlib import Path

BM25_INDEX_NAME = "storage/bm25.pkl"

@dataclass(frozen=True)
class ModelCfg:
    embed_model: str = "BAAI/bge-m3"
    rerank_model: str = "BAAI/bge-reranker-base"
    ollama_model: str = "gemma3:27b"
    
    # llm_choise: str = "gemma"

@dataclass(frozen=True)
class ChunkCfg:
    size: int = 2000
    overlap: int = 200

@dataclass(frozen=True)
class IndexCfg:
    chroma_dir: Path = Path(".chroma_rag")
    collection: str = "accidents_hybrid"

@dataclass(frozen=True)
class PipelineCfg:
    k_dense: int = 30
    k_sparse: int = 30
    k_final: int = 10
    use_rerank: bool = True
    schema_report_dir: Path | None = None
