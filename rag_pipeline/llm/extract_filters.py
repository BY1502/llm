from __future__ import annotations
import json
from typing import Any, Dict

from langchain_ollama import ChatOllama
from rag_pipeline.config import ModelCfg
from rag_pipeline.data_io.csv_schema import FILTERABLE_FIELDS

# ✅ 허용된 필드 목록
ALLOWED_FIELDS = FILTERABLE_FIELDS

FILTER_SYSTEM_PROMPT = f"""
당신은 사용자의 질문에서 '검색 필터'를 추출하는 AI입니다.
질문을 분석하여 데이터베이스 검색에 필요한 '조건(WHERE 절)'만 JSON으로 추출하세요.

허용된 필드: {", ".join(ALLOWED_FIELDS)}

[중요 규칙]
1. **검색 조건**만 추출하세요. 사용자가 **결과로 알고 싶어하는 항목(Target)**은 절대 필터로 넣지 마세요.
   - 나쁜 예: "스타필드 사고의 날씨 알려줘" -> {{ "사고명": "스타필드", "날씨": "알려줘" }} (X) -> 날씨는 조건이 아님!
   - 좋은 예: "스타필드 사고의 날씨 알려줘" -> {{ "사고명": "스타필드" }} (O)
2. 값이 명확한 고유명사, 숫자, 상태인 경우에만 추출하세요. ("미상", "모름", "알려줘", "무엇" 등의 값은 제외)
3. 조건이 없으면 빈 JSON {{}}을 반환하세요.
4. 출력은 오직 JSON 형식이어야 합니다.

예시 1:
질문: "날씨가 강우인 사고 알려줘"
출력: {{"날씨": "강우"}}

예시 2:
질문: "스타필드 안성 사고의 재발방지대책은?"
출력: {{"사고명": "스타필드 안성"}}
(설명: 재발방지대책은 사용자가 묻는 것이지, 검색 조건이 아님)
"""

def extract_filters(query: str, model_cfg: ModelCfg) -> Dict[str, Any]:
    """
    사용자 질문 -> 메타데이터 필터(dict) 변환
    (ChromaDB $and 문법 지원)
    """
    llm = ChatOllama(model=model_cfg.ollama_model, temperature=0)
    
    messages = [
        {"role": "system", "content": FILTER_SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    
    try:
        resp = llm.invoke(messages)
        content = getattr(resp, "content", str(resp))
        
        # JSON 파싱
        content = content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
            
        filters = json.loads(content)
        
        # 안전장치 1: 허용된 필드만 남기기
        # 안전장치 2: 값이 '미상', 'None' 등이면 제거 (LLM 할루시네이션 방지)
        safe_filters = {}
        for k, v in filters.items():
            if k in ALLOWED_FIELDS and v and str(v).strip() not in ["미상", "모름", "unknown", "None"]:
                safe_filters[k] = v
        
        # 🔥 [핵심 수정] ChromaDB 문법 호환 처리 ($and)
        if len(safe_filters) > 1:
            # 조건이 2개 이상이면 {"$and": [{"k1": "v1"}, {"k2": "v2"}]} 형태로 변환
            final_filter = {"$and": [{k: v} for k, v in safe_filters.items()]}
        else:
            # 조건이 0개 또는 1개면 그대로 반환
            final_filter = safe_filters

        if final_filter:
            print(f"[FILTER] Extracted filters (Raw): {filters}")
            print(f"[FILTER] 🎯 적용된 필터 (Chroma): {final_filter}")
            
        return final_filter

    except Exception as e:
        print(f"[FILTER] Extraction failed: {e}")
        return {}