
"""
Summarisation helpers (currently unused).

이 모듈은 예전에는 RAG로 검색된 문서를
- 단일 단계 요약
- 사고사례용 2단계(구조화 → 보고서) 요약
같은 방식으로 처리하던 코드가 들어 있었다.

지금 버전의 파이프라인에서는
`rag_pipeline.llm.json_answer.answer_with_json_autoschema` 만을 사용하고 있고,
이 모듈은 어디에서도 import 되지 않는다(단, 타입 힌트/레거시 용도로만 남겨둠).

나중에 다시 요약 파이프라인을 붙이고 싶다면
여기에서 새로운 구현을 추가하면 된다.
"""

from __future__ import annotations

from typing import List
from langchain_core.documents import Document
from rag_pipeline.config import ModelCfg


def summarize_with_llm(
    docs: List[Document],
    query: str | None,
    llm_model: str,
    cfg: ModelCfg,
) -> str:
    """
    레거시 API 서명을 유지하기 위한 stub 함수.

    현재는 실제로 호출되지 않으며, 호출될 경우 명시적으로 예외를 발생시킨다.
    """
    raise NotImplementedError(
        "summarize_with_llm는 더 이상 사용하지 않습니다. "
        "JSON Auto-Schema 기반 응답(`answer_with_json_autoschema`)을 사용하세요."
    )
