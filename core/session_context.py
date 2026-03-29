import time
from datetime import datetime
from typing import Any

_TOOL_TRACE = []
_SESSION_CUSTOMER = None

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

def reset_session():
    global _TOOL_TRACE, _SESSION_CUSTOMER
    _TOOL_TRACE.clear()
    _SESSION_CUSTOMER = None


#test

if __name__ == "__main__":
    add_tool_trace("test_tool", {"input": "test"}, {"output": "test"})
    time.sleep(0.33)
    add_tool_trace("test_tool_2", {"input": "test2"}, {"output": "test2"})

    used_tools = get_tool_trace()
    print(used_tools)




