from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.language_models import BaseLanguageModel

logger = logging.getLogger(__name__)


# 1) 태그 + 메타데이터 추출 프롬프트 템플릿
TAG_METADATA_PROMPT = """너는 건설현장 안전관리 문서를 분석해서 메타데이터와 태그를 추출하는 도우미야.

아래 문서는 "안전관리계획서 / 위험요인 / 안전대책"과 관련된 PDF 또는 TXT 문서의 일부야.
문서 내용을 읽고 다음 정보를 JSON 형식으로 추출해라.

반드시 아래 스키마를 지켜라.
값을 알 수 없으면 null 또는 빈 배열([])로 넣어라.

스키마:
{{
  "document_type": "safety_plan | risk_guide | checklist | other 중 하나를 선택",
  "project_name": "공사명 또는 프로젝트명 (없으면 null)",
  "location": "시/군/구 또는 현장 위치 (없으면 null)",
  "company": "시공사 또는 발주처 (없으면 null)",
  "facility_type": "지식산업센터, 아파트, 물류센터, 상가 등 (없으면 null)",

  "tags": [
    "이 문서에서 중요한 공종, 작업명, 위험요인, 안전대책 주제를 5~15개 정도 태그 형태로 추출한다.",
    "예: 가설공사, 동바리 설치, 비계 작업, 거푸집 해체, 추락 위험, 낙하물 위험, 개구부, 작업발판, 안전난간, 방호망"
  ],

  "source_filename": "이 문서의 파일 이름 (모르면 null)"
}}

출력 규칙:
- 반드시 유효한 JSON만 출력해라.
- 설명, 주석, 자연어 문장은 쓰지 말고 JSON만 출력해라.

--- 문서 내용 시작 ---
{content}
--- 문서 내용 끝 ---
"""


def _read_text_for_tagging(path: Path, max_chars: int = 4000) -> str:
    """
    태그/메타데이터 추출용으로 문서 앞부분 텍스트를 읽는다.
    - PDF: 앞 페이지들 합쳐서 max_chars까지
    - TXT: 앞부분 max_chars까지
    """
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        loader = PyPDFLoader(str(path))
        docs = loader.load()
        buf: list[str] = []
        total_len = 0
        for d in docs:
            if total_len >= max_chars:
                break
            remaining = max_chars - total_len
            chunk = d.page_content[:remaining]
            buf.append(chunk)
            total_len += len(chunk)
        text = "\n\n".join(buf)
    else:
        # 기본 TXT (필요하면 여기서 확장자 추가)
        loader = TextLoader(str(path), encoding="utf-8")
        docs = loader.load()
        text = docs[0].page_content[:max_chars]

    return text.strip()


def _clean_json_str(s: str) -> str:
    """
    LLM이 앞뒤에 ```json 같은 걸 붙였을 때를 대비해서 최소 정리.
    """
    s = s.strip()
    if s.startswith("```"):
        # ```json ... ``` 제거
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:]
    return s.strip()


def extract_metadata_from_text(
    text: str,
    filename: str | None,
    llm: BaseLanguageModel,
) -> Dict[str, Any]:
    """
    텍스트 + 파일명 + LLM → 메타데이터/태그 dict 반환.
    """
    prompt = TAG_METADATA_PROMPT.format(content=text)

    logger.debug("[TAG] LLM에 메타데이터/태그 추출 요청: file=%s", filename or "<unknown>")
    raw = llm.invoke(prompt)  # Chat/LLM 타입에 따라 .predict/.invoke 선택
    if isinstance(raw, str):
        output = raw
    else:
        # ChatModel인 경우 content만 꺼내기
        try:
            output = raw.content  # type: ignore[attr-defined]
        except Exception:
            output = str(raw)

    cleaned = _clean_json_str(output)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error("[TAG] JSON 파싱 실패: %s | output=%r", e, cleaned[:300])
        raise

    # 기본 필드 보정
    data.setdefault("document_type", None)
    data.setdefault("project_name", None)
    data.setdefault("location", None)
    data.setdefault("company", None)
    data.setdefault("facility_type", None)
    data.setdefault("tags", [])
    data.setdefault("source_filename", filename)

    # tags가 문자열로 왔을 수도 있으니 배열로 정규화
    if isinstance(data["tags"], str):
        data["tags"] = [t.strip() for t in data["tags"].split(",") if t.strip()]
    elif not isinstance(data["tags"], list):
        data["tags"] = []

    # file_path는 여기서 추가
    if filename:
        data.setdefault("file_path", str(filename))

    return data


def extract_metadata_from_file(
    path: str | Path,
    llm: BaseLanguageModel,
    max_chars: int = 4000,
) -> Dict[str, Any]:
    """
    파일 경로 + LLM → 문서 레벨 메타데이터/태그 dict.
    """
    p = Path(path)
    text = _read_text_for_tagging(p, max_chars=max_chars)
    if not text:
        raise ValueError(f"태깅할 텍스트가 비어 있습니다: {p}")

    md = extract_metadata_from_text(
        text=text,
        filename=p.name,
        llm=llm,
    )
    md["file_path"] = str(p)
    return md
