import time
import copy
import re
from datetime import datetime
from typing import Any


_TOOL_TRACE = []
_SESSION_CUSTOMER = None
_AGENT_HISTORY = []
_RESET_CALLBACKS = []
_LAST_AGENT_MESSAGE = None
_USER_MESSAGES = []
_HANDLE_DATASET_INCONSISTENCIES = False

# Dialog state for robust continuity
_DEFAULT_DIALOG_STATE = {
    "active_route": None,  # PUBLIC_INVENTORY | PRIVATE_INVENTORY | RAG | AUTH | None
    "active_entity_type": None,  # product | order | item | policy_topic | None
    "active_entity_id": None,  # id or None
    "candidate_entities": [],  # [{'id':..., 'name':...}, ...]
    "last_list_context": None,  # 'products' | 'orders' | ...
    "pending_clarification": False,
    "last_user_intent": None,  # QUERY | FOLLOW_QUERY | AUTH_ATTEMPT | SMALL_TALK
    "turn_index": 0,
    "timestamp": None,
}
_DIALOG_STATE = copy.deepcopy(_DEFAULT_DIALOG_STATE)

 
#tool trace management
def add_tool_trace(tool_name: str, input_data: Any, output_data: Any) -> dict:
    entry = {
        "tool_name": tool_name,
        "input_data": input_data,
        "output_data": output_data,
        "timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3],
    }
    _TOOL_TRACE.append(entry)
    return entry

def get_tool_trace() -> list:
    return list(_TOOL_TRACE)

def get_tool_trace_length() -> int:
    return len(_TOOL_TRACE)

def get_tool_trace_since(index: int) -> list:
    if (index >= len(_TOOL_TRACE)):
        return []
    if (index < 0):
        index = 0
    return _TOOL_TRACE[index:]

#customer session management

def set_session_customer(customer_id, display_name):
    global _SESSION_CUSTOMER
    _SESSION_CUSTOMER = {
        "customer_id": customer_id,
        "display_name": display_name
    }
    return dict(_SESSION_CUSTOMER)

def get_session_customer() -> dict | None:
    return _SESSION_CUSTOMER if _SESSION_CUSTOMER is not None else None

def set_handle_dataset_inconsistencies(enabled: bool):
    global _HANDLE_DATASET_INCONSISTENCIES
    _HANDLE_DATASET_INCONSISTENCIES = bool(enabled)
    return _HANDLE_DATASET_INCONSISTENCIES

def get_handle_dataset_inconsistencies() -> bool:
    return _HANDLE_DATASET_INCONSISTENCIES

def reset_session():
    global _TOOL_TRACE, _SESSION_CUSTOMER, _LAST_AGENT_MESSAGE
    _TOOL_TRACE.clear()
    _SESSION_CUSTOMER = None
    


#conversation context management
def set_last_agent_message(message: str | None):
    global _LAST_AGENT_MESSAGE
    _LAST_AGENT_MESSAGE = message
    return _LAST_AGENT_MESSAGE

def get_last_agent_message() -> str | None:
    return _LAST_AGENT_MESSAGE

def add_user_message(message: str):
    _USER_MESSAGES.append(message)
    return len(_USER_MESSAGES)

def get_user_messages() -> list[str]:
    return list(_USER_MESSAGES)

# Dialog state management
def get_dialog_state() -> dict:
    return copy.deepcopy(_DIALOG_STATE)

def update_dialog_state(**kwargs):
    global _DIALOG_STATE
    for k, v in kwargs.items():
        _DIALOG_STATE[k] = v
    return get_dialog_state()

def clear_dialog_memory():
    global _DIALOG_STATE
    _DIALOG_STATE = copy.deepcopy(_DEFAULT_DIALOG_STATE)
    return get_dialog_state()


def _extract_explicit_entity_id(message: str) -> str | None:
    msg = (message or "").strip().lower()
    patterns = [
        r"\b(?:pedido|orden)\s*(?:#|n[o°]\.?\s*)?(\d{2,})\b",
        r"\b(?:producto|sku|product_id|item_id)\s*(?:#|:)?\s*([a-z0-9\-_]{2,})\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, msg, flags=re.IGNORECASE)
        if match:
            return str(match.group(1)).strip()
    return None


def _token_overlap_score(left: str, right: str) -> float:
    left_tokens = {t for t in re.split(r"\s+", left) if t}
    right_tokens = {t for t in re.split(r"\s+", right) if t}
    if not left_tokens or not right_tokens:
        return 0.0
    inter = len(left_tokens.intersection(right_tokens))
    union = len(left_tokens.union(right_tokens))
    return inter / union if union else 0.0

# Entity resolver (deterministic)
def resolve_active_entity(message: str, state: dict | None = None, min_confidence: float = 0.65) -> dict:
    """
    Returns: {
        'resolved': bool,
        'entity_id': str|None,
        'entity_type': str|None,
        'reason': str,
        'confidence': float,
        'candidate': dict|None,
        'clarification_options': list|None
    }
    """
    if state is None:
        state = get_dialog_state()
    if not isinstance(state, dict):
        state = get_dialog_state()

    candidates = state.get("candidate_entities", [])
    msg = (message or "").lower().strip()
    if not candidates:
        return {
            "resolved": False,
            "entity_id": None,
            "entity_type": state.get("active_entity_type"),
            "reason": "sin candidatos en memoria",
            "confidence": 0.0,
            "candidate": None,
            "clarification_options": None,
        }

    explicit_id = _extract_explicit_entity_id(msg)
    ordinal_map = {
        "primero": 0, "primer": 0, "1": 0, "uno": 0,
        "segundo": 1, "2": 1, "dos": 1,
        "tercero": 2, "3": 2, "tres": 2,
        "cuarto": 3, "4": 3, "cuatro": 3,
        "quinto": 4, "5": 4, "cinco": 4,
        "último": -1, "ultimo": -1
    }
    ordinal_idx = None
    for word, idx in ordinal_map.items():
        if re.search(rf"\b{word}\b", msg):
            ordinal_idx = idx if idx >= 0 else len(candidates) - 1
            break

    has_anaphora = bool(re.search(r"\b(ese|esa|eso|el de antes|el anterior|la anterior)\b", msg))
    has_other_ref = bool(re.search(r"\b(el otro|la otra|no ese no, el otro)\b", msg))

    active_entity_id = state.get("active_entity_id")
    scored_candidates: list[tuple[float, dict, str]] = []

    for idx, candidate in enumerate(candidates):
        cand_id = str(candidate.get("id", "")).strip()
        cand_name = str(candidate.get("name", "")).strip().lower()
        score = 0.0
        reasons: list[str] = []

        if explicit_id and cand_id and explicit_id == cand_id:
            score += 1.0
            reasons.append("id_explicito")

        if ordinal_idx is not None and idx == ordinal_idx:
            score += 0.9
            reasons.append("ordinal")

        if has_anaphora and active_entity_id and cand_id == str(active_entity_id):
            score += 0.75
            reasons.append("anafora_activa")

        if has_other_ref and active_entity_id and cand_id != str(active_entity_id):
            score += 0.7
            reasons.append("el_otro")

        token_score = _token_overlap_score(msg, cand_name)
        if token_score > 0:
            score += min(0.6, token_score)
            reasons.append("similitud")

        if idx == len(candidates) - 1:
            score += 0.1
            reasons.append("recencia")

        scored_candidates.append((score, candidate, "+".join(reasons) or "sin_match"))

    scored_candidates.sort(key=lambda row: row[0], reverse=True)
    best_score, best_candidate, best_reason = scored_candidates[0]
    second_score = scored_candidates[1][0] if len(scored_candidates) > 1 else -1.0
    score_margin = best_score - second_score

    if best_score >= min_confidence and (score_margin >= 0.15 or best_score >= 0.9):
        return {
            "resolved": True,
            "entity_id": best_candidate.get("id"),
            "entity_type": state.get("active_entity_type"),
            "reason": f"{best_reason} (margin={score_margin:.2f})",
            "confidence": round(best_score, 3),
            "candidate": best_candidate,
            "clarification_options": None
        }

    options = [f"{i+1}. {c['name']}" for i, c in enumerate(candidates)] if candidates else None
    return {
        "resolved": False,
        "entity_id": None,
        "entity_type": state.get("active_entity_type"),
        "reason": f"ambigua: best={best_score:.2f} margin={score_margin:.2f}",
        "confidence": round(best_score, 3),
        "candidate": None,
        "clarification_options": options
    }

#agent history management

def add_agent_to_history(agent_name: str):
    _AGENT_HISTORY.append(agent_name)

def get_agent_history() -> list:
    return list(_AGENT_HISTORY)

def get_last_used_agent() -> str | None:
    return _AGENT_HISTORY[-1] if _AGENT_HISTORY else None

#reset callback management

def register_reset_callback(callback):
    _RESET_CALLBACKS.append(callback)

def reset_agent_memory():
    global _LAST_AGENT_MESSAGE
    _LAST_AGENT_MESSAGE = None
    _USER_MESSAGES.clear()
    _AGENT_HISTORY.clear()
    clear_dialog_memory()
    for cb in _RESET_CALLBACKS:
        cb()



#test

if __name__ == "__main__":
    add_tool_trace("test_tool", {"input": "test"}, {"output": "test"})
    time.sleep(0.33)
    add_tool_trace("test_tool_2", {"input": "test2"}, {"output": "test2"})

    used_tools = get_tool_trace()
    print(used_tools)




