from core.session_context import (
    get_tool_trace,
    get_tool_trace_length,
    get_tool_trace_since,
    set_session_customer,
    reset_session,
    get_session_customer,
    set_handle_dataset_inconsistencies,
    add_agent_to_history,
    get_agent_history,
    reset_agent_memory,
    add_user_message,
    get_user_messages,
    set_last_agent_message,
    get_last_agent_message,
    get_dialog_state,
    update_dialog_state,
)

from core.routers.query_router import classify_query_route
from core.agents.input_agent import classify_input
from core.agents.faq_agent import solve_faq_query
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
        solve_faq_query("ok")
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



def create_agent(streaming: bool = False, handle_dataset_inconsistencies: bool = False):
    warmup_routers()

    reset_agent_memory()
    reset_session()
    set_handle_dataset_inconsistencies(handle_dataset_inconsistencies)

    def _invalidate_entity_for_topic_switch(new_route: str):
        dialog_state = get_dialog_state()
        previous_route = dialog_state.get("active_route")
        if previous_route and previous_route != new_route:
            update_dialog_state(
                active_entity_id=None,
                active_entity_type=None,
                candidate_entities=[],
                last_list_context=None,
                pending_clarification=False,
            )

    def _extract_candidates_from_traces(trace_start_index: int):
        traces = get_tool_trace_since(trace_start_index)
        product_candidates = []
        order_candidates = []

        for trace in traces:
            tool_name = str(trace.get("tool_name", "")).strip()
            output_data = trace.get("output_data", {}) or {}

            if tool_name == "search_product":
                results = output_data.get("results", []) if isinstance(output_data, dict) else []
                for item in results:
                    if not isinstance(item, dict):
                        continue
                    product_id = item.get("product_id")
                    if product_id is None:
                        product_ids = item.get("product_ids")
                        if isinstance(product_ids, list) and product_ids:
                            product_id = product_ids[0]
                    if product_id is None:
                        continue
                    product_candidates.append(
                        {
                            "id": str(product_id),
                            "name": str(item.get("name", f"Producto {product_id}")),
                        }
                    )

            if tool_name == "get_customer_orders":
                orders = output_data.get("orders", []) if isinstance(output_data, dict) else []
                for order in orders:
                    if not isinstance(order, dict):
                        continue
                    order_id = order.get("order_id")
                    if order_id is None:
                        continue
                    label = f"Pedido {order_id}"
                    status = order.get("status")
                    if status:
                        label = f"Pedido {order_id} ({status})"
                    order_candidates.append({"id": str(order_id), "name": label})

            if tool_name == "get_order_details":
                if isinstance(output_data, dict):
                    order_id = output_data.get("order_id")
                    if order_id is not None:
                        order_candidates.append({"id": str(order_id), "name": f"Pedido {order_id}"})

        # dedupe preserving order
        def _dedupe(candidates: list[dict]) -> list[dict]:
            seen = set()
            deduped = []
            for candidate in candidates:
                key = candidate.get("id")
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(candidate)
            return deduped

        return _dedupe(product_candidates), _dedupe(order_candidates)

    def _update_dialog_after_tools(trace_start_index: int, active_route: str, intent: str):
        _invalidate_entity_for_topic_switch(active_route)
        product_candidates, order_candidates = _extract_candidates_from_traces(trace_start_index)

        if active_route == "PUBLIC_INVENTORY":
            update_dialog_state(
                active_route=active_route,
                last_user_intent=intent,
                pending_clarification=False,
                candidate_entities=product_candidates,
                active_entity_type="product" if product_candidates else None,
                active_entity_id=(product_candidates[-1]["id"] if product_candidates else None),
                last_list_context=("products" if product_candidates else None),
            )
            return

        if active_route == "PRIVATE_INVENTORY":
            update_dialog_state(
                active_route=active_route,
                last_user_intent=intent,
                pending_clarification=False,
                candidate_entities=order_candidates,
                active_entity_type="order" if order_candidates else None,
                active_entity_id=(order_candidates[-1]["id"] if order_candidates else None),
                last_list_context=("orders" if order_candidates else None),
            )
            return

        update_dialog_state(
            active_route=active_route,
            last_user_intent=intent,
            pending_clarification=False,
        )

    def agent(message: str):
        try:
            add_user_message(message)

            routed = classify_input(
                message,
                last_agent_message=get_last_agent_message(),
                agent_history=get_agent_history(),
                user_messages=get_user_messages(),
            )

            def _respond(text: str):
                set_last_agent_message(text)
                return AgentResponse(text)

            route = routed.get("route")
            user_message = routed.get("message", message)

            if route == "QUERY":
                trace_start_idx = get_tool_trace_length()
                query_route = classify_query_route(user_message)

                auth_route = query_route.get("auth_route")
                query_type = query_route.get("query_route")

                if auth_route == "PUBLIC":
                    if query_type == "FAQ":
                        response = solve_faq_query(user_message)

                        if response["route"] == "DIRECT_ANSWER":
                            add_agent_to_history("FAQ")
                            return _respond(response["message"])

                        if response["route"] == "BLOCK":
                            return _respond(response["message"])

                        if response["route"] == "QUERY":
                            rerouted_message = response.get("message", user_message)
                            response = solve_rag_query(rerouted_message)
                            add_agent_to_history("RAG")
                            return _respond(response["message"])

                        return _respond(
                            "No pude procesar la solicitud en este momento."
                        )

                    if query_type == "POLICY":
                        response = solve_rag_query(user_message)
                        add_agent_to_history("RAG")
                        _update_dialog_after_tools(trace_start_idx, "RAG", "QUERY")
                        return _respond(response["message"])

                    if query_type == "INVENTORY":
                        response = solve_inventory_query(user_message)
                        add_agent_to_history("PUBLIC_INVENTORY")
                        _update_dialog_after_tools(trace_start_idx, "PUBLIC_INVENTORY", "QUERY")
                        return _respond(response["message"])

                    return _respond(
                        "No pude clasificar con suficiente certeza tu consulta pública."
                    )

                if auth_route == "PRIVATE":
                    if get_session_customer() is None:                        
                        response = auth_agent_loop(user_message)
                        add_agent_to_history("AUTH")
                        _update_dialog_after_tools(trace_start_idx, "AUTH", "QUERY")
                        return _respond(response["message"])

                    response = solve_inventory_query(user_message, query_type="PRIVATE")
                    add_agent_to_history("PRIVATE_INVENTORY")
                    _update_dialog_after_tools(trace_start_idx, "PRIVATE_INVENTORY", "QUERY")
                    return _respond(response["message"])

                return _respond(
                    "No pude determinar si la consulta es pública o privada."
                )

            if route == "FOLLOW_QUERY":
                trace_start_idx = get_tool_trace_length()
                follow_route = routed.get("follow_query_route")
                if not follow_route:
                    dialog_state = get_dialog_state()
                    follow_route = dialog_state.get("active_route")
                if follow_route == "PRIVATE_INVENTORY":
                    if get_session_customer() is None:
                        response = auth_agent_loop(user_message)
                        add_agent_to_history("AUTH")
                        _update_dialog_after_tools(trace_start_idx, "AUTH", "FOLLOW_QUERY")
                        return _respond(response["message"])
                    response = solve_inventory_query(message, query_type="PRIVATE")
                    add_agent_to_history("PRIVATE_INVENTORY")
                    update_dialog_state(
                        pending_clarification=False,
                        active_route="PRIVATE_INVENTORY",
                        last_user_intent="FOLLOW_QUERY",
                    )
                    _update_dialog_after_tools(trace_start_idx, "PRIVATE_INVENTORY", "FOLLOW_QUERY")
                    return _respond(response["message"])

                if follow_route in ("PUBLIC_INVENTORY", "INVENTORY"):
                    response = solve_inventory_query(message)
                    add_agent_to_history("PUBLIC_INVENTORY")
                    update_dialog_state(
                        pending_clarification=False,
                        active_route="PUBLIC_INVENTORY",
                        last_user_intent="FOLLOW_QUERY",
                    )
                    _update_dialog_after_tools(trace_start_idx, "PUBLIC_INVENTORY", "FOLLOW_QUERY")
                    return _respond(response["message"])

                if follow_route == "RAG":
                    response = solve_rag_query(message)
                    add_agent_to_history("RAG")
                    _update_dialog_after_tools(trace_start_idx, "RAG", "FOLLOW_QUERY")
                    return _respond(response["message"])

                if follow_route == "AUTH":
                    response = auth_agent_loop(user_message)
                    add_agent_to_history("AUTH")
                    _update_dialog_after_tools(trace_start_idx, "AUTH", "FOLLOW_QUERY")
                    return _respond(response["message"])

                return _respond(
                    "No pude determinar a qué se refiere tu consulta de seguimiento."
                )

            if route == "AUTH_ATTEMPT":
                response = auth_agent_loop(user_message)
                add_agent_to_history("AUTH")
                update_dialog_state(active_route="AUTH", last_user_intent="AUTH_ATTEMPT")

                if response.get("authenticated"):
                    print("auth ok")

                return _respond(response["message"])

            if route == "DIRECT_ANSWER":
                update_dialog_state(last_user_intent="SMALL_TALK")
                return _respond(routed["message"])

            if route == "BLOCK":
                return _respond(routed["message"])

           # print(get_agent_history())
            return _respond("No pude procesar la solicitud en este momento.")
        
        except Exception as e:
            error_text = f"Ocurrió un error procesando tu solicitud. ({type(e).__name__}: {e})"
            set_last_agent_message(error_text)
            return AgentResponse(error_text)

        

    def reset_memory():
        reset_agent_memory()

    agent.reset_memory = reset_memory
    return agent