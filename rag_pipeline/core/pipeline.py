from rag_pipeline.config import ModelCfg
from rag_pipeline.retrieval.hybrid import hybrid_retrieve
from rag_pipeline.retrieval.rerankers import CrossEncoderReranker
from rag_pipeline.indexing.sparse import build_sparse_retriever
from rag_pipeline.llm.json_answer import answer_with_json_autoschema
from rag_pipeline.llm.extract_filters import extract_filters
from rag_pipeline.core.global_index import get_global_index
from rag_pipeline.core.workspace import WORKSPACES
from rag_pipeline.core.processing import VS_CACHE, BM25_CACHE, DOCS_CACHE
from fastapi import HTTPException
from langchain_core.documents import Document
import logging
from typing import Any
logger = logging.getLogger(__name__)


# ---------------------------
#  Helper: tags parser
# ---------------------------
def _ensure_tags_list(raw):
    """
    metadata['tags'] 값이
    - string: "가설공사, 비계, 추락 위험"
    - list: ["가설공사","비계"]
    어느 형태든 list[str] 로 변환.
    """
    if isinstance(raw, str):
        return [t.strip() for t in raw.split(",") if t.strip()]
    if isinstance(raw, list):
        return raw
    return []

CASE_QUERY_STOPWORDS = [
    "사고", "사례", "사고사례", "사고 사례",
    "알려줘", "알려 줘", "알려줘요",
    "에 대한", "에대한", "에 관해", "에관해",
    "을", "를", "이", "가", "은", "는"
]

def extract_query_keywords(query: str | None) -> list[str]:
    """
    질문에서 '순영종합건설', '스타필드안성', '지식산업센타' 같은
    고유명사/핵심 키워드만 대략 뽑는다.
    '우선순위 힌트' 로 사용
    """
    if not query:
        return []
    q = query.strip()
    if not q:
        return []
    
    parts = q.split()
    keywords: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        if any(sw in part for sw in CASE_QUERY_STOPWORDS):
            continue
        if len(part) < 2:
            continue
        
        keywords.append(part)
        
    return keywords

def prioritize_docs_by_keywords(docs: list[Document], keywords: list[str]) -> list[Document]:
    """
    문서를 버리지 않고, 키워드가 많이 포함된 문서일수록 리스트의 앞쪽으로 정렬합니다.
    (단순 유/무가 아니라, 매칭된 개수(Count)를 기준으로 내림차순 정렬)
    """
    if not keywords or not docs:
        return docs
    
    # 문서별 매칭 점수 계산
    scored_docs = []
    for d in docs:
        md = d.metadata or {}
        # 검색 대상 텍스트 생성 (메타데이터 + 본문)
        # None 값 필터링 및 문자열 변환
        meta_values = [str(v) for v in md.values() if v is not None]
        text_pieces = meta_values + [
            str(md.get("source", "") or ""),
            str(d.page_content or ""),
        ]
        big_text = " ".join(text_pieces)
        
        # 🔥 [핵심 수정] 키워드가 '몇 개'나 포함되었는지 카운트 (점수화)
        match_count = sum(1 for kw in keywords if kw in big_text)
        
        # (매칭 개수, 원래 순서 보존을 위한 문서 객체)
        scored_docs.append((match_count, d))
        
    # 매칭 개수 기준 내림차순 정렬 (많은 게 위로)
    # 파이썬의 sort는 stable하므로, 점수가 같으면 원래(Vector/BM25) 순위가 유지됨
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    
    # 정렬된 문서 리스트 반환
    return [d for count, d in scored_docs]
def run_pipeline(req):

    model_cfg = ModelCfg()

    # 🔹 LLM 모델 선택
    llm_choise = getattr(req, "llm_model", None) or "gemma3:27b"
    print(f"[LLM MODEL] : {llm_choise}")

    # 질의 정보 로그
    print(f"[QUERY] query={req.query} | workspace={req.workspace_id} | tags={req.tags}")

    reranker = CrossEncoderReranker(model_cfg.rerank_model) if req.use_rerank else None

    vs = bm25 = docs = None
    ws_id = getattr(req, "workspace_id", None)

    # -----------------------------------------------------
    # 1) workspace 캐시 기반 로드
    # -----------------------------------------------------
    if ws_id:
        ws = WORKSPACES.get(ws_id)
        if not ws:
            raise HTTPException(400, "유효하지 않은 workspace_id 입니다.")

        vs = VS_CACHE.get(ws_id)
        bm25 = BM25_CACHE.get(ws_id)
        docs = DOCS_CACHE.get(ws_id)

        print(
            f"[DEBUG] ws_id={ws_id}, "
            f"vs_in_cache={ws_id in VS_CACHE}, "
            f"bm25_in_cache={ws_id in BM25_CACHE}, "
            f"docs_in_cache={ws_id in DOCS_CACHE}, "
            f"docs_len={len(docs) if docs else 0}"
        )

    # -----------------------------------------------------
    # 2) 전역 인덱스 fallback
    # -----------------------------------------------------
    if vs is None and bm25 is None and docs is None:
        try:
            g_vs, g_bm25, g_docs = get_global_index()
            vs = g_vs or vs
            bm25 = g_bm25 or bm25
            docs = g_docs or docs
            print(
                f"[DEBUG] global_index: vs={bool(vs)}, "
                f"bm25={bool(bm25)}, docs_len={len(docs) if docs else 0}"
            )
        except Exception:
            print("[DEBUG] get_global_index() 실패, 전역 인덱스 사용 안 함")

    # -----------------------------------------------------
    # 3) BM25 lazy build
    # -----------------------------------------------------
    if bm25 is None and docs:
        bm25 = build_sparse_retriever(docs)
        if ws_id:
            BM25_CACHE[ws_id] = bm25
        print(f"[DEBUG] lazy build bm25, docs={len(docs)}")
        
    # -----------------------------------------------------
    # 3.5) Pre-filtering 조건 추출
    # -----------------------------------------------------
    
    search_filter = None
    # 질문이 너무 짧지 않을 때만 필터 추출 시도 (비용/속도 고려)
    if req.query and len(req.query) > 5:
        try:
            search_filter = extract_filters(req.query, model_cfg)
        except Exception as e:
            print(f"[FILTER] 필터 추출 중 오류 발생: {e}")

    # -----------------------------------------------------
    # 4) retrieval (dense / sparse / hybrid)
    # -----------------------------------------------------
    used_vs = False
    used_bm25 = False

    mode = (req.retrieval or "hybrid").lower()
    
    CANDIDATE_K = max(req.final_k * 3, 10)

    if mode == "dense":
        # Dense 모드에도 필터 적용
        dense_kwargs = {"k": req.k}
        if search_filter:
            dense_kwargs["filter"] = search_filter
        matched = vs.as_retriever(search_kwargs={"k": req.k}).invoke(req.query) if vs else []
        used_vs = True

    elif mode == "sparse":
        if bm25:
            bm25.k = req.k
            matched = bm25.invoke(req.query)
            used_bm25 = True
        else:
            matched = []

    else:  # hybrid
        if vs and bm25:
            matched = hybrid_retrieve(
                req.query,
                vs,
                bm25,
                k_dense=req.k,
                k_sparse=req.k,
                # k_final=req.final_k,
                k_final=CANDIDATE_K,
                reranker=reranker,
                filter=search_filter, # 추출한 필터 전달
            )
            used_vs = True
            used_bm25 = True
        elif vs:
            matched = vs.as_retriever(search_kwargs={"k": req.k}).invoke(req.query)
            used_vs = True
        elif bm25:
            bm25.k = req.k
            matched = bm25.invoke(req.query)
            used_bm25 = True
        else:
            matched = []

    print(f"[DEBUG] mode={mode}, vs={used_vs}, bm25={used_bm25}, matched_len={len(matched)}")
    print(
        f"[DEBUG] query={req.query!r}, mode={mode}, "
        f"k={req.k}, final_k={req.final_k}, use_rerank={req.use_rerank}, "
    )

    # -----------------------------------------------------
    # 5) 매칭된 문서들의 태그 분포 출력
    # -----------------------------------------------------
    all_tags = []
    for d in matched:
        all_tags.extend(_ensure_tags_list(d.metadata.get("tags")))

    # 🔥 [주석 처리] 매칭된 데이터 출력 ( 너무 길어서 출력하지 않음 )
    # unique_tags = sorted(set(all_tags))
    # print(f"[MATCH] matched_docs={len(matched)} | unique_tags={unique_tags}")

    # if matched:
    #     print("[DEBUG] first matched metadata:", matched[0].metadata)

    # -----------------------------------------------------
    # 6) 요청에서 tags 필터링
    # -----------------------------------------------------
    req_tags = getattr(req, "tags", None)
    if req_tags:
        required_tags = set(req_tags)
        before = len(matched)

        matched = [
            d for d in matched
            if required_tags.intersection(_ensure_tags_list(d.metadata.get("tags")))
        ]

        print(f"[TAG-FILTER] required={list(required_tags)} | before={before} -> after={len(matched)}")

    # 문서가 없으면 종료 (기존 구조 유지)
    if not matched:
        return {
            "summary": "질문과 일치하는 문서를 찾지 못했습니다.",
            "sources": [],
        }

    # ----------------------------------------------------
    # 6.5) 키워드 기반 '순서만' 조정 (문서 버리지 않음)
    # ----------------------------------------------------
    keywords = extract_query_keywords(getattr(req, "query", None))
    ordered = prioritize_docs_by_keywords(matched, keywords)

    print(
        # f"[LLM MODEL] : {llm_choise}"
        f"[ORDER] keywords={keywords} | before={len(matched)} | "
        f"first_changed={matched[0] is not ordered[0] if matched and ordered else False}"
    )

    # 🔹 LLM에 넘길 최종 문서 리스트 (정렬 적용, 실패 시 fallback)
    # final_docs = ordered or matched
    final_docs = ordered[:req.final_k] if ordered else matched[:req.final_k]

    print(
        f"[ORDER] keywords={keywords} | candidates={len(ordered)} -> final_k={len(final_docs)} | "
        f"first_changed={matched[0] is not ordered[0] if matched and ordered else False}"
    )

    # -----------------------------------------------------
    # 7) LLM JSON Auto-Schema 응답 생성
    # -----------------------------------------------------

    # ✅ JSON Mode + Auto-Schema 호출
    json_answer = answer_with_json_autoschema(
        query=req.query,
        docs=final_docs,
        model_cfg=model_cfg,
        llm_model=llm_choise,   # 요청에서 받은 LLM 선택 반영
    )

    # -----------------------------------------------------
    # 8) 반환 (기존 응답 스키마 유지 + JSON Auto-Schema 추가)
    # -----------------------------------------------------

    # summary 필드는 response 모델이 요구하니까, 간단히 schema.description을 사용
    summary_text = json_answer.get("schema", {}).get("description", "")

    return {
        "summary": summary_text,          # 🔹 FastAPI 응답 스키마용 (string)
        "mode": mode,
        "llm_model": llm_choise,
        "json_data": json_answer,              # 🔹 새 JSON Auto-Schema 전체
        "sources": [
            {
                "source": d.metadata.get("source", ""),
                "doc_id": d.metadata.get("doc_id") or d.metadata.get("source") or "",
                "chunk_id": d.metadata.get("chunk_id") or d.metadata.get("id") or "",
            }
            for d in final_docs
        ],
    }

