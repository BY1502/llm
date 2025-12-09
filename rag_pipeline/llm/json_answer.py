from __future__ import annotations

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
1. 사용자가 특정 항목(예: 구체적 사고원인, 재발방지 대책 등)을 물어보면, 그 항목에 해당하는 내용만 중심으로 답변하세요.
2. 여러 사고가 섞여 있더라도, 질문과 가장 관련도가 높은 한 건의 사고를 기준으로 답변하세요.
3. 컨텍스트에는 CSV의 모든 원본 정보가 포함되어 있으므로, 불필요한 정보(예: '미입력', '0', '1,000억원 미만' 등)는 무시하고, 핵심 정보에 집중하십시오.
4. 컨텍스트에 없는 정보는 절대 지어내지 말고, "컨텍스트에 해당 정보가 없습니다."라고 명시하세요.
5. 만약 질문이 '순서', '절차' 등을 요구한다면, 단순히 번호가 매겨진 목록을 발견했다고 해서 바로 추출하지 마십시오.
   **반드시 그 목록의 소제목이나 각 항목의 내용이 질문의 의도(예: '해체')와 일치하는지 확인하십시오.**
   * 주의: 문서 제목에 '설치/해체'가 같이 있더라도, **하위 목록의 내용이 '설치(자재반입, 조립 등)'라면 해체 작업의 답변으로 사용해서는 안 됩니다.**
   * '해체'를 물었다면 내용에 '해체', '분리', '제거' 등의 단어가 포함된 목록만을 선택하여 추출하십시오.
6. [🔥 핵심 규칙: 조건부 출력] **만약 질문이 특정 사고의 상세 내용 (예: '스타필드 안성 사고사례')을 묻는 경우**, 답변은 **주요 항목을 빠짐없이 포함한 마크다운 리스트 형식** (예: `* 사고명: ..., * 사고원인: ...`)으로 작성하여 **사실을 명확하게 구분**하십시오. **이때 포함할 항목은 컨텍스트에서 찾을 수 있는 '사고명', '사고일시', '사고경위', '구체적 사고원인', '재발방지대책' 등**을 중심으로 모델이 **자율적으로 판단**하여 포함합니다.
7. 그 외의 일반적인 질문(예: '공사 종류는 무엇인가요?', '안전 수칙은?')에 대해서는 규칙 5의 구조를 따르지 않고, 간결한 서술형 문장 또는 일반적인 마크다운 리스트로 자유롭게 답변하세요.
8. [🔥 종료 규칙] 질문에 대한 답변 작성이 완료되면, 컨텍스트에 남은 다른 문서(특히 질문과 관련 없는 '설치' 등의 내용)가 있더라도 **절대 추가 내용을 덧붙이지 말고 즉시 답변을 종료**하십시오.
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
만약 질문이 **'순서', '절차', '단계', '방법' 등**의 **복잡한 단계별 목록**을 요구한다면, 컨텍스트에 있는 **가장 상세하고 계층적인 목록**을 **요약하지 말고** 번호나 불렛을 사용하여 **원문 그대로 충실히 재구성하여** 답변하십시오.
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
