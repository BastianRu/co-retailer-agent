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
    get_conversation_window,
    set_last_agent_message,
    get_last_agent_message,
    get_dialog_state,
    update_dialog_state,
    resolve_active_entity,
)

from core.routers.query_router import classify_query_route
from core.agents.input_agent import classify_input
from core.agents.faq_agent import solve_faq_query
from core.agents.rag_agent import solve_rag_query
from core.agents.inventory_agent import solve_inventory_query
from core.agents.auth_agent import auth_agent_loop
from core.data_store import load_all_s3_data
import time
import re


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


def _build_ambiguous_query_clarification(message: str) -> str:
    msg = str(message or "").strip().lower()

    if "inform" in msg:
        return "¿A qué tipo de información te refieres? ¿Catálogo de productos, tus pedidos, o políticas de envío/devolución?"

    if "compra" in msg:
        return "¿En qué parte de tu compra necesitas ayuda? ¿Buscas productos, consultas sobre tu pedido o información sobre políticas?"

    if "producto" in msg:
        return "¿Necesitas ayuda con un producto del catálogo, con una compra ya realizada o con políticas como garantía y devoluciones?"

    return "¿Te refieres a productos, pedidos o políticas? Dame un poco más de contexto para ayudarte mejor."


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
                            "price": item.get("price"),
                            "availability_units": item.get("availability_units"),
                        }
                    )

            if tool_name == "get_product_details":
                if isinstance(output_data, dict):
                    product_id = output_data.get("product_id")
                    if product_id is not None:
                        product_candidates.append(
                            {
                                "id": str(product_id),
                                "name": str(output_data.get("name", f"Producto {product_id}")),
                                "price": output_data.get("price"),
                                "availability_units": output_data.get("availability_units"),
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

    def _update_dialog_after_tools(trace_start_index: int, active_route: str, intent: str, source_message: str | None = None):
        _invalidate_entity_for_topic_switch(active_route)
        previous_state = get_dialog_state()
        product_candidates, order_candidates = _extract_candidates_from_traces(trace_start_index)

        if active_route == "PUBLIC_INVENTORY":
            previous_candidates = previous_state.get("candidate_entities", []) if previous_state.get("active_entity_type") == "product" else []
            resolved_candidates = product_candidates or previous_candidates
            update_dialog_state(
                active_route=active_route,
                last_user_intent=intent,
                pending_clarification=False,
                candidate_entities=resolved_candidates,
                active_entity_type="product" if resolved_candidates else previous_state.get("active_entity_type"),
                active_entity_id=(resolved_candidates[-1]["id"] if resolved_candidates else previous_state.get("active_entity_id")),
                last_list_context=("products" if resolved_candidates else previous_state.get("last_list_context")),
                last_resolved_query=source_message,
                pending_auth_route=None,
                pending_auth_message=None,
            )
            return

        if active_route == "PRIVATE_INVENTORY":
            previous_candidates = previous_state.get("candidate_entities", []) if previous_state.get("active_entity_type") == "order" else []
            resolved_candidates = order_candidates or previous_candidates
            update_dialog_state(
                active_route=active_route,
                last_user_intent=intent,
                pending_clarification=False,
                candidate_entities=resolved_candidates,
                active_entity_type="order" if resolved_candidates else previous_state.get("active_entity_type"),
                active_entity_id=(resolved_candidates[-1]["id"] if resolved_candidates else previous_state.get("active_entity_id")),
                last_list_context=("orders" if resolved_candidates else previous_state.get("last_list_context")),
                last_resolved_query=source_message,
                pending_auth_route=None,
                pending_auth_message=None,
            )
            return

        update_dialog_state(
            active_route=active_route,
            last_user_intent=intent,
            pending_clarification=False,
            last_resolved_query=source_message,
            pending_auth_route=None if active_route != "AUTH" else get_dialog_state().get("pending_auth_route"),
            pending_auth_message=None if active_route != "AUTH" else get_dialog_state().get("pending_auth_message"),
        )

    def _candidate_label(entity_id: str | None, candidates: list[dict]) -> str | None:
        if entity_id is None:
            return None
        for candidate in candidates:
            if str(candidate.get("id")) == str(entity_id):
                return str(candidate.get("name", "")).strip() or None
        return None

    def _format_currency(value: object) -> str | None:
        try:
            amount = float(str(value))
        except (TypeError, ValueError):
            return None
        return f"${int(round(amount)):,}".replace(",", ".")

    def _find_candidate_from_last_agent_message(candidates: list[dict], last_agent_message: str | None) -> dict | None:
        if not candidates or not last_agent_message:
            return None

        normalized_reply = str(last_agent_message).strip().lower()
        best_candidate = None
        best_score = 0.0

        for candidate in candidates:
            score = 0.0
            candidate_name = str(candidate.get("name", "")).strip().lower()
            if candidate_name and candidate_name in normalized_reply:
                score += 2.0

            price_text = _format_currency(candidate.get("price"))
            if price_text and price_text in str(last_agent_message):
                score += 2.0

            availability_units = candidate.get("availability_units")
            if availability_units is not None and f"{availability_units}" in str(last_agent_message):
                score += 0.5

            if score > best_score:
                best_score = score
                best_candidate = candidate

        return best_candidate if best_score > 0 else None

    def _is_price_follow_up(message: str) -> bool:
        return bool(re.search(r"\b(cuanto|cuánto)\s+(cuesta|vale)\b|\bprecio\b", str(message).lower()))

    def _is_stock_follow_up(message: str) -> bool:
        return bool(re.search(r"\b(stock|disponible|disponibilidad|unidades)\b", str(message).lower()))

    def _is_free_shipping_follow_up(message: str) -> bool:
        return bool(re.search(r"env[ií]o gratis", str(message).lower()))

    def _is_shipping_time_follow_up(message: str) -> bool:
        return bool(re.search(r"\b(cuanto|cuánto)\s+(tarda|demora)\b|\bllega\b|\btiempo de entrega\b", str(message).lower()))

    def _is_warranty_follow_up(message: str) -> bool:
        return "garant" in str(message).lower()

    def _is_return_follow_up(message: str) -> bool:
        return "devol" in str(message).lower() or "devolver" in str(message).lower()

    def _is_promotions_follow_up(message: str) -> bool:
        return bool(re.search(r"\b(promoci[oó]n|promociones|descuento|oferta)\b", str(message).lower()))

    def _rewrite_public_inventory_follow_up(message: str) -> str | None:
        dialog_state = get_dialog_state()
        candidates = dialog_state.get("candidate_entities", []) or []
        resolved = resolve_active_entity(message, dialog_state)
        candidate = resolved.get("candidate") if resolved.get("resolved") else None
        if candidate is None:
            candidate = _find_candidate_from_last_agent_message(candidates, get_last_agent_message())
        if candidate is None:
            return None

        product_id = candidate.get("id")
        if product_id is None:
            return None

        if _is_price_follow_up(message):
            return f"¿Cuánto cuesta el producto {product_id}?"
        if _is_stock_follow_up(message):
            return f"¿Hay stock del producto {product_id}?"
        if _is_free_shipping_follow_up(message):
            return f"¿El envío es gratis para el producto {product_id}?"
        if _is_shipping_time_follow_up(message):
            return f"¿Cuánto tarda en llegar el producto {product_id}?"
        if _is_warranty_follow_up(message):
            return f"¿El producto {product_id} tiene garantía?"
        if _is_return_follow_up(message):
            return f"¿Puedo devolver el producto {product_id}?"
        if _is_promotions_follow_up(message):
            return f"¿Qué promociones tiene el producto {product_id}?"

        return None

    def _build_follow_up_context(message: str, follow_route: str) -> str:
        dialog_state = get_dialog_state()
        candidates = dialog_state.get("candidate_entities", []) or []
        resolved = resolve_active_entity(message, dialog_state)
        active_entity_id = dialog_state.get("active_entity_id")
        active_entity_label = _candidate_label(active_entity_id, candidates)
        last_resolved_query = dialog_state.get("last_resolved_query")
        pending_auth_message = dialog_state.get("pending_auth_message")
        last_agent_message = get_last_agent_message()

        lines = ["Contexto de seguimiento:"]
        if last_resolved_query:
            lines.append(f"- Consulta previa relevante: {last_resolved_query}")
        if last_agent_message:
            lines.append(f"- Última respuesta del asistente: {last_agent_message}")
        if pending_auth_message and follow_route == "PRIVATE_INVENTORY":
            lines.append(f"- Consulta privada pendiente/original: {pending_auth_message}")

        if resolved.get("resolved") and resolved.get("candidate"):
            candidate = resolved["candidate"]
            entity_type = dialog_state.get("active_entity_type") or "entidad"
            lines.append(f"- Entidad referida: {entity_type} {candidate.get('name')} (id: {candidate.get('id')})")
        elif active_entity_id is not None:
            entity_type = dialog_state.get("active_entity_type") or "entidad"
            if active_entity_label:
                lines.append(f"- Entidad activa: {entity_type} {active_entity_label} (id: {active_entity_id})")
            else:
                lines.append(f"- Entidad activa: {entity_type} id {active_entity_id}")

        if candidates:
            candidate_preview = "; ".join(
                f"{str(candidate.get('name', '')).strip()} (id: {candidate.get('id')})"
                for candidate in candidates[:5]
                if candidate.get("id") is not None
            )
            if candidate_preview:
                lines.append(f"- Candidatos recientes: {candidate_preview}")

        lines.append("- Usa este contexto solo para resolver referencias anafóricas del mensaje actual.")

        if follow_route == "PRIVATE_INVENTORY":
            lines.append("- Si el usuario pregunta por su pedido sin order_id y la tool permite omitirlo, puedes usar el pedido más reciente autenticado antes de pedir aclaración.")

        lines.append(f"Mensaje actual del usuario: {message}")
        return "\n".join(lines)

    def agent(message: str):
        try:
            add_user_message(message)

            routed = classify_input(
                message,
                last_agent_message=get_last_agent_message(),
                agent_history=get_agent_history(),
                user_messages=get_user_messages(),
                dialog_state=get_dialog_state(),
                session_customer=get_session_customer(),
                conversation_window=get_conversation_window(limit=12),
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
                        _update_dialog_after_tools(trace_start_idx, "RAG", "QUERY", user_message)
                        return _respond(response["message"])

                    if query_type == "INVENTORY":
                        response = solve_inventory_query(user_message)
                        add_agent_to_history("PUBLIC_INVENTORY")
                        _update_dialog_after_tools(trace_start_idx, "PUBLIC_INVENTORY", "QUERY", user_message)
                        return _respond(response["message"])

                    return _respond(
                        "No pude clasificar con suficiente certeza tu consulta pública."
                    )

                if auth_route == "PRIVATE":
                    if get_session_customer() is None:                        
                        update_dialog_state(
                            active_route="AUTH",
                            pending_auth_route="PRIVATE_INVENTORY",
                            pending_auth_message=user_message,
                            last_resolved_query=user_message,
                            last_user_intent="QUERY",
                            pending_clarification=False,
                        )
                        add_agent_to_history("AUTH")
                        return _respond("No puedo ayudarte con eso. Comparte tu DNI o teléfono para autenticarte.")

                    response = solve_inventory_query(user_message, query_type="PRIVATE")
                    add_agent_to_history("PRIVATE_INVENTORY")
                    _update_dialog_after_tools(trace_start_idx, "PRIVATE_INVENTORY", "QUERY", user_message)
                    return _respond(response["message"])

                update_dialog_state(pending_clarification=True, last_user_intent="QUERY")
                return _respond(_build_ambiguous_query_clarification(user_message))

            if route == "FOLLOW_QUERY":
                trace_start_idx = get_tool_trace_length()
                follow_route = routed.get("follow_query_route")
                dialog_state = get_dialog_state()
                if not follow_route:
                    follow_route = dialog_state.get("active_route")

                if follow_route == "AUTH" and get_session_customer() is not None:
                    restored_route = dialog_state.get("pending_auth_route") or dialog_state.get("active_route")
                    if restored_route in {"PRIVATE_INVENTORY", "PUBLIC_INVENTORY", "RAG"}:
                        follow_route = restored_route

                if follow_route == "PRIVATE_INVENTORY":
                    if get_session_customer() is None:
                        response = auth_agent_loop(user_message)
                        add_agent_to_history("AUTH")
                        update_dialog_state(
                            active_route="AUTH",
                            pending_auth_route="PRIVATE_INVENTORY",
                            pending_auth_message=dialog_state.get("pending_auth_message") or dialog_state.get("last_resolved_query"),
                            last_user_intent="FOLLOW_QUERY",
                            pending_clarification=False,
                        )
                        return _respond(response["message"])
                    follow_message = _build_follow_up_context(message, "PRIVATE_INVENTORY")
                    response = solve_inventory_query(follow_message, query_type="PRIVATE")
                    add_agent_to_history("PRIVATE_INVENTORY")
                    update_dialog_state(
                        pending_clarification=False,
                        active_route="PRIVATE_INVENTORY",
                        last_user_intent="FOLLOW_QUERY",
                    )
                    _update_dialog_after_tools(trace_start_idx, "PRIVATE_INVENTORY", "FOLLOW_QUERY", message)
                    return _respond(response["message"])

                if follow_route in ("PUBLIC_INVENTORY", "INVENTORY"):
                    follow_message = _rewrite_public_inventory_follow_up(message) or _build_follow_up_context(message, "PUBLIC_INVENTORY")
                    response = solve_inventory_query(follow_message)
                    add_agent_to_history("PUBLIC_INVENTORY")
                    update_dialog_state(
                        pending_clarification=False,
                        active_route="PUBLIC_INVENTORY",
                        last_user_intent="FOLLOW_QUERY",
                    )
                    _update_dialog_after_tools(trace_start_idx, "PUBLIC_INVENTORY", "FOLLOW_QUERY", message)
                    return _respond(response["message"])

                if follow_route == "RAG":
                    follow_message = _build_follow_up_context(message, "RAG")
                    response = solve_rag_query(follow_message)
                    add_agent_to_history("RAG")
                    _update_dialog_after_tools(trace_start_idx, "RAG", "FOLLOW_QUERY", message)
                    return _respond(response["message"])

                if follow_route == "AUTH":
                    response = auth_agent_loop(user_message)
                    add_agent_to_history("AUTH")
                    update_dialog_state(active_route="AUTH", last_user_intent="FOLLOW_QUERY")
                    return _respond(response["message"])

                return _respond(
                    "No pude determinar a qué se refiere tu consulta de seguimiento."
                )

            if route == "AUTH_ATTEMPT":
                response = auth_agent_loop(user_message)
                add_agent_to_history("AUTH")
                dialog_state = get_dialog_state()
                pending_auth_route = dialog_state.get("pending_auth_route")
                pending_auth_message = dialog_state.get("pending_auth_message")
                if response.get("authenticated") and pending_auth_route in {"PRIVATE_INVENTORY", "PUBLIC_INVENTORY", "RAG"}:
                    update_dialog_state(
                        active_route=pending_auth_route,
                        last_user_intent="AUTH_ATTEMPT",
                        pending_clarification=False,
                        last_resolved_query=pending_auth_message or dialog_state.get("last_resolved_query"),
                    )
                else:
                    update_dialog_state(active_route="AUTH", last_user_intent="AUTH_ATTEMPT")

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