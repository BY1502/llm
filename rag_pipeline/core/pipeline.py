from rag_pipeline.config import ModelCfg
from rag_pipeline.retrieval.hybrid import hybrid_retrieve
from rag_pipeline.retrieval.rerankers import CrossEncoderReranker
from rag_pipeline.indexing.sparse import build_sparse_retriever
from rag_pipeline.llm.summarize import summarize_with_llm
from rag_pipeline.llm.json_answer import answer_with_json_autoschema
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

def _answer_to_text(answer: Any) -> str:
    """
    JSON Auto-Schema의 answer 객체를 사람이 읽을 수 있는 텍스트로 풀어주는 유틸.
    구조가 매번 달라도 최대한 예쁘게 펼쳐서 한글 요약처럼 보여주기 위함.
    """
    if answer is None:
        return ""

    # 1) 사고 예시처럼 단순 dict 인 경우
    if isinstance(answer, dict):
        lines = []
        for k, v in answer.items():
            # 기본 타입은 "키: 값" 형태로
            if isinstance(v, (str, int, float, bool)):
                lines.append(f"{k}: {v}")
            # 리스트인 경우 (예: cases 리스트 등)
            elif isinstance(v, list):
                # 리스트 안에 dict들이 들어있는 경우 첫 번째만 간단 요약
                if v and isinstance(v[0], dict):
                    lines.append(f"{k}:")
                    first = v[0]
                    for kk, vv in first.items():
                        if isinstance(vv, (str, int, float, bool)):
                            lines.append(f"  - {kk}: {vv}")
                else:
                    # 단순 문자열 리스트 등
                    joined = ", ".join(map(str, v))
                    lines.append(f"{k}: {joined}")
            else:
                # 그 밖의 타입들은 문자열로 그냥 던짐
                lines.append(f"{k}: {str(v)}")
        return "\n".join(lines)

    # 2) 리스트 전체가 answer인 경우
    if isinstance(answer, list):
        parts = []
        for idx, item in enumerate(answer, start=1):
            if isinstance(item, dict):
                parts.append(f"[{idx}번 항목]")
                for k, v in item.items():
                    parts.append(f"- {k}: {v}")
            else:
                parts.append(f"- {item}")
        return "\n".join(parts)

    # 3) 그 외는 그냥 문자열로 캐스팅
    return str(answer)


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
    문서를 버리지 않고, 키워드가 들어있는 문서만 앞으로 정렬
    - 매칭된 문서들 먼저
    - 나머지 문서들 그 뒤에 그대로
    """
    if not keywords or not docs:
        return docs
    
    hits: list[Document] = []
    others: list[Document] = []
    
    for d in docs:
        md = d.metadata or {}
        meta_values = [str(v) for v in md.values() if v is not None]
        text_pieces = meta_values + [
            str(md.get("source", "") or ""),
            str(d.page_content or ""),
        ]
        big_text = " ".join(text_pieces)
        
        if any(kw in big_text for kw in keywords):
            hits.append(d)
        else:
            others.append(d)
            
    # 문서 하나도 매칭 안 되면, 순서 안 건드림 
    if not hits:
        return docs
    
    # 키워드가 들어있는 애들을 앞으로 나머지는 그대로 뒤에
    return hits + others

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
    # 4) retrieval (dense / sparse / hybrid)
    # -----------------------------------------------------
    used_vs = False
    used_bm25 = False

    mode = (req.retrieval or "hybrid").lower()

    if mode == "dense":
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
                k_final=req.final_k,
                reranker=reranker,
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

    unique_tags = sorted(set(all_tags))
    print(f"[MATCH] matched_docs={len(matched)} | unique_tags={unique_tags}")

    if matched:
        print("[DEBUG] first matched metadata:", matched[0].metadata)

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
        f"[LLM MODEL] : {llm_choise}"
        f"[ORDER] keywords={keywords} | before={len(matched)} | "
        f"first_changed={matched[0] is not ordered[0] if matched and ordered else False}"
    )

    # 🔹 LLM에 넘길 최종 문서 리스트 (정렬 적용, 실패 시 fallback)
    final_docs = ordered or matched

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
        "json": json_answer,              # 🔹 새 JSON Auto-Schema 전체
        "sources": [
            {
                "source": d.metadata.get("source", ""),
                "doc_id": d.metadata.get("doc_id") or d.metadata.get("source") or "",
                "chunk_id": d.metadata.get("chunk_id") or d.metadata.get("id") or "",
            }
            for d in final_docs
        ],
    }

