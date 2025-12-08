import re
from typing import Dict


def extract_kv_metadata(text: str) -> Dict[str, str]:
    """Best-effort 'key: value' extraction for Korean docs."""
    meta: Dict[str, str] = {}
    pattern = re.compile(r"^\s*([\w가-힣/()\- ]+?)\s*[:\-]\s*(.+)$")
    for line in text.splitlines():
        m = pattern.match(line.strip())
        if m:
            k = m.group(1).strip()
            v = m.group(2).strip()
            if k and k not in meta:
                meta[k] = v
    return meta

