from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


FIELD_ALIASES: Dict[str, list[str]] = {
    "사고명": ["사고명", "사고 이름", "사고 제목"],
    "사고일시": ["사고일시", "사고 일시", "발생일시"],
    "구체적사고원인": ["구체적사고원인", "구체적 사고원인", "사고원인"],
    "날씨": ["날씨", "기상상태", "기상 상태"],
    "공사종류": ["공사종류", "공사 종류", "시설물분류", "시설물 분류"],
    "인적사고종류": ["인적사고종류(대분류)", "인적사고종류", "사고 중분류"],
    "공종(대)": ["공종(대분류)", "사고종류(소분류)", "사고 소분류"],
    "사고경위": ["사고경위", "사고 경위", "사고 개요", "사고 개황"],
    "사망자수": ["사망자수", "사망", "사망자"],
    "부상자수": ["부상자수", "부상자", "부상"],
    "사고발생후 조치사항": [
        "사고발생후 조치사항",
        "사고 발생 후 조치사항",
        "사고발생 후 조치",
        "사고 발생 후 조치",
    ],
    "피해내용": ["피해내용", "피해 내용", "피해상황"],
    "재발방지대책": ["재발방지대책","재발 방지 대책", "재발방지 대책", "재발 방지대책"],
    "향후조치계획": ["향후조치계획","향후 조치 계획", "향후조치 계획", "향후 조치계획"],
}


def meta_get(md: Dict[str, Any], logical_key: str) -> str:
    for real_key in FIELD_ALIASES.get(logical_key, [logical_key]):
        if real_key in md and str(md[real_key]).strip():
            return str(md[real_key]).strip()
    return "정보 없음"


def _resolve_column(logical_key: str, columns: List[str]) -> str | None:
    for real_key in FIELD_ALIASES.get(logical_key, [logical_key]):
        for col in columns:
            if col.strip() == real_key:
                return col
    return None


def make_csv_schema_report(df: pd.DataFrame, path: Path) -> Dict[str, Any]:
    cols = [str(c).strip() for c in df.columns]
    alias_mapping = {lk: _resolve_column(lk, cols) for lk in FIELD_ALIASES.keys()}
    used = {v for v in alias_mapping.values() if v}
    unused = [c for c in cols if c not in used]
    null_ratio = {c: float(df[c].isna().mean()) for c in df.columns}
    sample = {c: (None if df[c].isna().all() else str(df[c].dropna().iloc[0])[:120]) for c in df.columns}
    return {
        "file": str(path),
        "rows": int(len(df)),
        "columns": cols,
        "alias_mapping": alias_mapping,
        "unused_columns": unused,
        "null_ratio": null_ratio,
        "sample_values": sample,
    }


def print_csv_schema_report(report: Dict[str, Any]) -> None:
    logger.info("=== CSV 스키마 자동감지 리포트 ===")
    logger.info("파일: %s", report["file"])
    logger.info("행 수: %s | 컬럼 수: %s", report["rows"], len(report["columns"]))
    logger.info("[별칭 매핑 결과]")
    for lk, rk in report["alias_mapping"].items():
        logger.info("- %s -> %s", lk, rk if rk else "매핑 없음")
    if report["unused_columns"]:
        logger.info("[사용되지 않은 컬럼]")
        for c in report["unused_columns"]:
            logger.info("- %s", c)
    logger.info("[결측 비율 상위 5개]")
    top_null = sorted(report["null_ratio"].items(), key=lambda x: x[1], reverse=True)[:5]
    for c, r in top_null:
        logger.info("- %s: %.2f%%", c, r * 100)
    logger.info("=== 리포트 끝 ===")

def save_csv_schema_report(report: Dict[str, Any], outdir: str | None) -> None:
    if not outdir:
        return
    os.makedirs(outdir, exist_ok=True)
    name = Path(report["file"]).stem + "_schema_report.json"
    out = Path(outdir) / name
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info("[저장됨] 스키마 리포트: %s", out)

