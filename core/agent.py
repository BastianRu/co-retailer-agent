from core.session_context import (
    get_tool_trace,
    set_session_customer,
    reset_session,
    get_session_customer,
    add_agent_to_history,
    get_agent_history,
    reset_agent_memory,
)

from core.routers.query_router import classify_query_route
from core.agents.input_agent import classify_input
from core.agents.rag_agent import solve_rag_query
from core.agents.inventory_agent import solve_inventory_query
from core.agents.auth_agent import auth_agent_loop
from core.data_store import load_all_s3_data

import time


_WARMED_UP = False


def warmup_routers():
    global _WARMED_UP
    if _WARMED_UP:
        return

    try:
        classify_query_route("ok")
    except Exception:
        pass

    try:
        classify_input("ok")
    except Exception:
        pass

    try:
        load_all_s3_data()
    except Exception:
        pass

    try:
        solve_rag_query("ok")
    except Exception:
        pass

    try:
        solve_inventory_query("ok")
    except Exception:
        pass

    _WARMED_UP = True


class AgentResponse:
    def __init__(self, content: str):
        self.content = content

    def __str__(self):
        return self.content


def create_agent(streaming: bool = False):
    warmup_routers()

    reset_agent_memory()
    reset_session()

    def agent(message: str):
        try:
            routed = classify_input(message)

            route = routed.get("route")
            user_message = routed.get("message", message)

            if route == "QUERY":
                query_route = classify_query_route(user_message)

                auth_route = query_route.get("auth_route")
                query_type = query_route.get("query_route")

                if auth_route == "PUBLIC":
                    if query_type in ("FAQ", "POLICY"):
                        response = solve_rag_query(user_message)
                        add_agent_to_history("RAG")
                        return AgentResponse(response["message"])

                    if query_type == "INVENTORY":
                        response = solve_inventory_query(user_message)
                        add_agent_to_history("PUBLIC_INVENTORY")
                        return AgentResponse(response["message"])

                    return AgentResponse(
                        "No pude clasificar con suficiente certeza tu consulta pública."
                    )

                if auth_route == "PRIVATE":
                    if get_session_customer() is None:                        
                        response = auth_agent_loop(user_message)
                        add_agent_to_history("AUTH")
                        return AgentResponse(response["message"])

                    response = solve_inventory_query(user_message, query_type="PRIVATE")
                    add_agent_to_history("PRIVATE_INVENTORY")
                    return AgentResponse(response["message"])

                return AgentResponse(
                    "No pude determinar si la consulta es pública o privada."
                )

            if route == "FOLLOW_QUERY":
                follow_route = routed.get("follow_query_route")

                if follow_route == "PRIVATE_INVENTORY":
                    if get_session_customer() is None:
                        response = auth_agent_loop(user_message)
                        add_agent_to_history("AUTH")
                        return AgentResponse(response["message"])
                    
                    response = solve_inventory_query(message, query_type="PRIVATE")
                    add_agent_to_history("PRIVATE_INVENTORY")
                    return AgentResponse(response["message"])

                if follow_route in ("PUBLIC_INVENTORY", "INVENTORY"):
                    response = solve_inventory_query(message)
                    add_agent_to_history("PUBLIC_INVENTORY")
                    return AgentResponse(response["message"])

                if follow_route == "RAG":
                    response = solve_rag_query(message)
                    add_agent_to_history("RAG")
                    return AgentResponse(response["message"])

                if follow_route == "AUTH":
                    response = auth_agent_loop(user_message)
                    add_agent_to_history("AUTH")
                    return AgentResponse(response["message"])

                return AgentResponse(
                    "No pude determinar a qué se refiere tu consulta de seguimiento."
                )

            if route == "AUTH_ATTEMPT":
                response = auth_agent_loop(user_message)
                add_agent_to_history("AUTH")

                if response.get("authenticated"):
                    print("auth ok")

                return AgentResponse(response["message"])

            if route == "DIRECT_ANSWER":
                return AgentResponse(routed["message"])

            if route == "BLOCK":
                return AgentResponse(routed["message"])

            print(get_agent_history())
            return AgentResponse("No pude procesar la solicitud en este momento.")
        
        except Exception:
            return AgentResponse("Ocurrió un error procesando tu solicitud.")

        

    def reset_memory():
        reset_agent_memory()

    agent.reset_memory = reset_memory
    return agent