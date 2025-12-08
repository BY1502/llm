# # rag_pipeline/llm/json_answer.py

# from __future__ import annotations

# import json
# from dataclasses import dataclass
# from typing import Any, Dict, List

# from langchain_core.documents import Document
# from langchain_ollama import ChatOllama

# from rag_pipeline.config import ModelCfg  # 네가 쓰고 있는 위치 기준으로 맞춰줘


# # 🔹 JSON + Auto-Schema 용 시스템 프롬프트
# AUTO_SCHEMA_SYSTEM_PROMPT = """
# 당신은 JSON 구조 설계와 데이터 매핑을 수행하는 어시스턴트입니다.

# 규칙:
# 1. 반드시 유효한 JSON만 출력하세요. 설명 문장, 마크다운, 주석을 절대 포함하지 마세요.
# 2. 최상위 키는 반드시 query, schema, answer, source_chunks 네 개만 사용하세요.
# 3. schema.fields 안의 필드 목록은 "사용자 질문"과 "컨텍스트"를 보고 당신이 스스로 설계하세요.
# 4. answer 객체는 schema.fields 정의에 맞게만 값을 채우세요.
# 5. source_chunks에는 실제로 사용한 근거 청크만 최대 5개까지 넣으세요.
# 6. 자료가 없는 경우에는 "자료 없음"이라고 명시하세요.

# 출력 JSON 스키마:
# {
#   "query": string,                     
#   "schema": {
#     "description": string,             
#     "fields": [
#       {
#         "name": string,                
#         "type": "string | number | boolean | array | object",
#         "description": string
#       }
#     ]
#   },
#   "answer": object,                    
#   "source_chunks": [
#     {
#       "doc_id": string,
#       "chunk_id": string,
#       "snippet": string
#     }
#   ]
# }
# """


# def _build_context_block(docs: List[Document]) -> str:
#     """LLM에 넘길 컨텍스트 문자열 생성 (기존 디버그 스타일 유지 느낌으로)."""
#     lines: List[str] = []
#     for idx, d in enumerate(docs):
#         meta = d.metadata or {}
#         doc_id = (
#             meta.get("doc_id")
#             or meta.get("source")
#             or meta.get("doc")
#             or ""
#         )
#         chunk_id = meta.get("chunk_id") or meta.get("id") or f"chunk_{idx}"

#         header = f"[doc_id={doc_id} chunk_id={chunk_id}]"
#         content = d.page_content.strip().replace("\n", " ")
#         lines.append(f"{header} {content}")
#     return "\n".join(lines)


# def _parse_json_safely(raw: str) -> Dict[str, Any]:
#     """
#     Ollama가 앞뒤에 약간의 텍스트를 붙이는 경우를 대비해서
#     JSON 블록만 잘라내서 파싱하는 유틸.
#     """
#     raw = raw.strip()

#     # 이미 깨끗한 JSON일 가능성 우선 시도
#     try:
#         return json.loads(raw)
#     except Exception:
#         pass

#     # 첫 번째 '{'부터 마지막 '}'까지 잘라서 재시도
#     try:
#         start = raw.index("{")
#         end = raw.rindex("}") + 1
#         return json.loads(raw[start:end])
#     except Exception:
#         raise ValueError(f"LLM JSON 파싱 실패: {raw[:200]}...")


# def answer_with_json_autoschema(
#     query: str,
#     docs: List[Document],
#     model_cfg: ModelCfg,
#     llm_model: str | None = None,
# ) -> Dict[str, Any]:
#     """
#     JSON Mode + Auto-Schema 방식으로 답변하는 LLM 호출 함수.
#     - query: 사용자 질문
#     - docs: RAG로 찾은 top_k Document 리스트
#     - model_cfg: 기존에 쓰는 ModelCfg (ollama_model 사용)
#     """
#     context_block = _build_context_block(docs)

#     user_prompt = f"""
# 사용자 질문: "{query}"

# 검색된 컨텍스트(청크들):
# \"\"\"
# {context_block}
# \"\"\"

# 위 정보를 참고해서 다음 규칙을 지키면서 JSON만 출력하세요.
# """

#     llm = ChatOllama(
#         model=model_cfg.ollama_model or model_cfg.ollama_model,
#         temperature=0.1,
#     )

#     # 🔹 Chat 형식 메시지 구성 (네 스타일에 맞게 단순하게)
#     messages = [
#         {"role": "system", "content": AUTO_SCHEMA_SYSTEM_PROMPT},
#         {"role": "user", "content": user_prompt},
#     ]

#     resp = llm.invoke(messages)
#     content = getattr(resp, "content", resp)

#     data = _parse_json_safely(content)
    
#     if "answer" not in data and "response" in data:
#         data["answer"] = data["response"]

#     # 최소 검증
#     for key in ("query", "schema", "answer"):
#         if key not in data:
#             raise ValueError(f"LLM JSON 응답에 '{key}' 키가 없습니다: {data}")

#     return data

# rag_pipeline/llm/json_answer.py

from __future__ import annotations

import json
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_ollama import ChatOllama

from rag_pipeline.config import ModelCfg


# 🔹 RAG 단순 답변용 시스템 프롬프트
AUTO_SCHEMA_SYSTEM_PROMPT = """
당신은 건설 안전 사고 데이터를 기반으로 답변하는 assistant입니다.

역할:
- 사용자의 질문과 함께 주어지는 "검색된 컨텍스트"를 보고,
  질문에 대한 답변을 한국어로 정확하고 간결하게 작성합니다.
- 반드시 컨텍스트 안에 있는 정보만 사용해야 하며,
  파일 밖의 지식은 사용하지 마십시오.

규칙:
1. 사용자가 특정 항목(예: 구체적 사고원인, 재발방지 대책 등)을 물어보면,
   그 항목에 해당하는 내용만 중심으로 답변하세요.
2. 여러 사고가 섞여 있더라도, 질문과 가장 관련도가 높은 한 건의 사고를 기준으로 답변하세요.
3. 답변은 자연어 문장 또는 짧은 bullet 형태로 작성해도 됩니다.
4. 컨텍스트에 없는 정보는 절대 지어내지 말고,
   "컨텍스트에 해당 정보가 없습니다."라고 명시하세요.
"""


def _build_context_block(docs: List[Document]) -> str:
    """
    LLM에 넘길 컨텍스트 문자열 생성.
    각 Document의 메타데이터(doc_id, chunk_id)와 내용을 한 줄로 정리한다.
    """
    lines: List[str] = []
    for idx, d in enumerate(docs):
        meta = d.metadata or {}
        doc_id = (
            meta.get("doc_id")
            or meta.get("source")
            or meta.get("doc")
            or ""
        )
        chunk_id = meta.get("chunk_id") or meta.get("id") or f"chunk_{idx}"

        header = f"[doc_id={doc_id} chunk_id={chunk_id}]"
        content = d.page_content.strip().replace("\n", " ")
        lines.append(f"{header} {content}")
    return "\n".join(lines)


def _parse_json_safely(raw: str) -> Dict[str, Any]:
    """
    (현재는 사용하지 않지만, 필요 시 재사용 가능한 JSON 파서)
    Ollama가 앞뒤에 텍스트를 붙이는 경우를 대비해서
    JSON 블록만 잘라내서 파싱하는 유틸.
    """
    raw = raw.strip()

    # 이미 깨끗한 JSON일 가능성 우선 시도
    try:
        return json.loads(raw)
    except Exception:
        pass

    # 첫 번째 '{'부터 마지막 '}'까지 잘라서 재시도
    try:
        start = raw.index("{")
        end = raw.rindex("}") + 1
        return json.loads(raw[start:end])
    except Exception:
        raise ValueError(f"LLM JSON 파싱 실패: {raw[:200]}...")


def answer_with_json_autoschema(
    query: str,
    docs: List[Document],
    model_cfg: ModelCfg,
    llm_model: str | None = None,
) -> Dict[str, Any]:
    """
    단순 RAG 답변 함수.

    Flow:
      1) 검색된 문서들을 컨텍스트 문자열로 합쳐서
      2) 시스템 프롬프트 + 질문 + 컨텍스트를 LLM에 전달
      3) LLM이 생성한 한국어 답변을 그대로 받아서
      4) 파이프라인 호환을 위해 JSON 형태로 래핑해서 반환

    ※ 더 이상 Auto-Schema로 필드를 설계하지 않고,
       LLM이 질문과 컨텍스트를 보고 자유롭게 답변하도록 한다.
    """

    # 1) 컨텍스트 블록 생성
    context_block = _build_context_block(docs)

    # 2) 사용자 프롬프트
    user_prompt = f"""
[사용자 질문]
{query}

[검색된 컨텍스트]
\"\"\" 
{context_block}
\"\"\" 

위 컨텍스트 범위 안에서만 정보를 사용하여,
사용자의 질문에 대해 한국어로 간결하게 답변하세요.
질문에서 특정 항목(예: 구체적 사고원인, 재발방지 대책 등)을 요구하면,
그 항목 위주로만 정리해서 답변하세요.
컨텍스트에 정보가 없으면, 정보가없다고만 출력하세요.
"""

    llm = ChatOllama(
        model=llm_model or model_cfg.ollama_model,
        temperature=0.1,
    )

    messages = [
        {"role": "system", "content": AUTO_SCHEMA_SYSTEM_PROMPT.strip()},
        {"role": "user", "content": user_prompt.strip()},
    ]

    resp = llm.invoke(messages)
    content = getattr(resp, "content", resp)
    answer_text = str(content).strip()

    # 3) 사용한 근거 청크 (상위 몇 개만)
    source_chunks = []
    for i, d in enumerate(docs[:3]):
        md = d.metadata or {}
        source_chunks.append(
            {
                "doc_id": md.get("source", ""),
                "chunk_id": md.get("chunk_id") or md.get("id", f"chunk_{i}"),
                "snippet": d.page_content[:300],
            }
        )

    # 4) 파이프라인/프론트 호환용 JSON 래핑
    return {
        "query": query,
        "schema": {
            "description": "자연어 RAG 답변",
            "fields": [
                {
                    "name": "answer",
                    "type": "string",
                    "description": "사용자 질문에 대한 한국어 답변",
                }
            ],
        },
        "answer": {
            "answer": answer_text,
        },
        "source_chunks": source_chunks,
    }
