from __future__ import annotations
import json
from typing import Iterable, List, Dict, Any, Optional

REQUIRED_FIELDS = [
    "is_attack_contained",
    "is_knowledge_sufficient",
    "are_forensics_preserved",
    "is_eradicated",
    "is_hardened",
    "is_recovered",
]

def _extract_json_objects(text: str) -> List[str]:
    objects: List[str] = []
    depth = 0
    start_idx = None
    in_string = False
    escape = False

    for i, ch in enumerate(text):
        if in_string:   # text 中""结束
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':  # text 中""开始
            in_string = True
            continue

        if ch == "{":
            if depth == 0:
                start_idx = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start_idx is not None:
                    objects.append(text[start_idx : i + 1])  # 再list中添加一个完整的JSON对象
                    start_idx = None
    return objects

def _coerce_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lower = value.strip().lower()
        if lower == "true":
            return True
        if lower == "false":
            return False
    return None

def _parse_state_json(text: str) -> Optional[Dict[str, bool]]:
    candidates = _extract_json_objects(text)  # 提取 [{JSON},{JSON},...,{JSON}] 列表 
    for raw in reversed(candidates):
        try:
            data = json.loads(raw)     
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        if not all(key in data for key in REQUIRED_FIELDS):
            continue
        
        parsed: Dict[str, bool] = {}  # 运行效果等价于parsed = {}, 类型提示：parsed 是 str→bool 的字典
        ok = True
        for key in REQUIRED_FIELDS:
            coerced = _coerce_bool(data.get(key))
            if coerced is None:
                ok = False
                break
            parsed[key] = coerced
        if ok:
            return parsed
    return None

def exact_match_score(prediction_text: str, label_text: str) -> int:
    pred = _parse_state_json(prediction_text)
    label = _parse_state_json(label_text)
    if pred is None or label is None:
        return 0
    return 1 if all(pred[k] == label[k] for k in REQUIRED_FIELDS) else 0


def exact_match_accuracy(predictions: Iterable[str], labels: Iterable[str]) -> float:
    scores = [exact_match_score(p, l) for p, l in zip(predictions, labels)]
    if not scores:
        return 0.0
    return sum(scores) / len(scores)
