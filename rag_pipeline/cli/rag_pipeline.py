# from __future__ import annotations

# import argparse
# import logging
# from pathlib import Path

# from rag_pipeline.config import ChunkCfg, IndexCfg, ModelCfg, PipelineCfg
# from rag_pipeline.data_io.loaders import load_documents_from_path
# from rag_pipeline.indexing.vectorstore import build_vectorstore
# from rag_pipeline.indexing.sparse import build_sparse_retriever
# from rag_pipeline.retrieval.hybrid import hybrid_retrieve
# from rag_pipeline.retrieval.rerankers import CrossEncoderReranker
# from rag_pipeline.llm.summarize import summarize_with_llm


# def main():
#     logging.basicConfig(level=logging.INFO, format="%(message)s")
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--data", type=str, required=True, help="file or folder: csv/txt/pdf")
#     ap.add_argument("--query", type=str, required=True)
#     ap.add_argument("--k", type=int, default=8, help="dense/sparse top-k")
#     ap.add_argument("--final", type=int, default=5, help="final results after rerank")
#     ap.add_argument("--no_rerank", action="store_true")
#     ap.add_argument("--schema_report_dir", type=str, default=None)
#     args = ap.parse_args()

#     model_cfg = ModelCfg()
#     chunk_cfg = ChunkCfg()
#     index_cfg = IndexCfg()
#     pipe_cfg = PipelineCfg(
#         k_dense=args.k,
#         k_sparse=args.k,
#         k_final=args.final,
#         use_rerank=not args.no_rerank,
#         schema_report_dir=Path(args.schema_report_dir) if args.schema_report_dir else None,
#     )

#     path = Path(args.data)
#     docs = load_documents_from_path(path, chunk_cfg, pipe_cfg)
#     if not docs:
#         print("요약할 정보가 없습니다.")
#         return

#     vs = build_vectorstore(docs, model_cfg, index_cfg)
#     bm25 = build_sparse_retriever(docs)
#     reranker = CrossEncoderReranker(model_cfg.rerank_model) if pipe_cfg.use_rerank else None

#     matched = hybrid_retrieve(
#         args.query,
#         vs,
#         bm25,
#         k_dense=pipe_cfg.k_dense,
#         k_sparse=pipe_cfg.k_sparse,
#         k_final=pipe_cfg.k_final,
#         reranker=reranker,
#     )

#     summary = summarize_with_llm(matched, model_cfg.ollama_model)
#     print("\n=== 검색 결과 요약 ===\n")
#     print(summary)
#     print("\n[참고 문서]")
#     for i, d in enumerate(matched, 1):
#         when = d.metadata.get('사고일시', '')
#         print(f"{i}. {when} | {d.metadata.get('source','')}")


# if __name__ == "__main__":
#     main()
