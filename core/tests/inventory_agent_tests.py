import time

from core.agents.inventory_agent import solve_inventory_query
from core.session_context import reset_session, set_session_customer, get_tool_trace

CATALOG_CASES = False
ANSWER_ORDER_CASES = False
ANSWER_LOGISTICS_CASES = False
ANSWER_RETURN_WARRANTY_CASES = False
NO_DATA_CASES = False

#catalog cases
answer_catalog_cases = [
    # discovery básico
    "que iphones tienen?",
    "busca iphones",
    "que celulares apple tienen disponibles?",
    "que productos samsung tienen?",
    
    # con intención más específica
    "tienen iphone 15 pro max?",
    "busca iphone 14",
    "hay celulares baratos?",
    "que laptops tienen disponibles?",
    
    # precio
    "cuanto vale el iphone 15 pro max?",
    "precio del iphone 14",
    "cuanto cuesta ese producto?",
    
    # stock
    "hay stock del iphone 15 pro max?",
    "esta disponible el iphone 14?",
    "tienen unidades disponibles?",
    
    # promociones
    "hay promociones en iphones?",
    "que productos tienen descuento?",
    "este producto tiene oferta?",
    
    # detalle de producto
    "dame detalles del iphone 15 pro max",
    "que incluye ese producto?",
    "ese producto tiene garantia?",
    "ese producto requiere instalacion?",
    
    # ambigüedad (importantes)
    "cuanto vale?",
    "hay disponibles?",
    "que tal ese?",
    
    # comparación implícita
    "que iphone es mejor?",
    "que celular recomiendas?",
    
    # edge cases
    "tienen iphone 99 ultra?",
    "busca algo que no exista xyz123",
]


# -----------------------------
# ANSWER: ORDERS
# Customer recomendado: 1208
# -----------------------------
answer_orders_cases = [
    "muestrame mis pedidos",
    "muestrame los detalles de mi orden 28",
    "muestrame los detalles de mi orden 618",
    "muestrame los detalles de mi orden 481",
    "que productos vienen en mi orden 28?",
    "que productos vienen en mi orden 481?",
    "cual es el total de mi orden 481?",
    "cual fue el metodo de pago de mi orden 618?",
    "mi orden 618 fue cancelada?",
]


# -----------------------------
# ANSWER: LOGISTICS
# Customer recomendado: 1208
# orden 28 -> shipped
# orden 481 -> delivered
# orden 618 -> cancelled
# -----------------------------
answer_logistics_cases = [
    "donde esta mi pedido 28?",
    "cual es el estado de mi orden 28?",
    "mi pedido 28 ya fue enviado?",
    "cual es el tracking de mi orden 28?",
    "cuando llega mi pedido 28?",
    "cual es el estado logistico de mi orden 481?",
    "mi pedido 481 ya fue entregado?",
    "cual es el tracking de mi orden 481?",
    "la orden 618 tiene envio o fue cancelada?",
]


# -----------------------------
# ANSWER: RETURN / WARRANTY
# Customer recomendado: 1292
# orden 520, item 1526 -> devolucion vigente hasta 2026-04-11
# orden 520, item 1526 -> garantia vigente hasta 2029-02-10
# orden 196, item 578 -> garantia vencida en 2026-04-05
# -----------------------------
answer_return_warranty_cases = [
    "puedo devolver el item 1526 de la orden 520?",
    "la orden 520 item 1526 es elegible para devolucion?",
    "hasta cuando puedo devolver el item 1526 de la orden 520?",
    "el item 1526 de la orden 520 aun tiene garantia?",
    "hasta cuando tiene garantia el item 1526 de la orden 520?",
    "el item 578 de la orden 196 aun tiene garantia?",
]


# -----------------------------
# NO_DATA CASES
# Customer recomendado: 1292
# -----------------------------
no_data_cases = [
    "dame el estado de la orden 999999",
    "quiero garantia del item 999999 de la orden 520",
    "puedo devolver el item 999999 de la orden 520?",
    "muestrame los detalles de mi orden 999999",
]

CUSTOM_CASES = True
custom_cases = [
    "que productos LG hay?",
    "dame los detalles del monitor LG ultra wide",
]

if CUSTOM_CASES:
    reset_session()
    set_session_customer(1208, "test_user")

    for i, case in enumerate(custom_cases):
        start = time.perf_counter()
        response = solve_inventory_query(case)
        elapsed = time.perf_counter() - start

        print(str(case))
        print(f"Case {i}: expected=ANSWER route={response['route']}")
        print(f"Message: {response['message']}")
        #print(f"Reason: {response['reason']}")
        print(f"Request time (s): {elapsed:.6f}")
        if response["response_data"] is not None:
            summary = response["response_data"].metrics.get_summary()
            last_usage = summary["agent_invocations"][-1]["usage"]
            print(f"Per-call usage: {last_usage}")
    print("-------------------------------------")
    print(get_tool_trace())

if CATALOG_CASES:
    reset_session()

    for i, case in enumerate(answer_catalog_cases):
        start = time.perf_counter()
        response = solve_inventory_query(case, query_type="PUBLIC")
        elapsed = time.perf_counter() - start

        print(str(case))
        print(f"Case {i}: expected=ANSWER route={response['route']}")
        print(f"Message: {response['message']}")
        #print(f"Reason: {response['reason']}")
        print(f"Request time (s): {elapsed:.6f}")
        if response["response_data"] is not None:
            summary = response["response_data"].metrics.get_summary()
            last_usage = summary["agent_invocations"][-1]["usage"]
            print(f"Per-call usage: {last_usage}")
        print("-------------------------------------")
    print(get_tool_trace())

if ANSWER_ORDER_CASES:
    reset_session()
    set_session_customer(1208, "test_user")

    for i, case in enumerate(answer_orders_cases):
        start = time.perf_counter()
        response = solve_inventory_query(case, query_type="PRIVATE")
        elapsed = time.perf_counter() - start

        print(str(case))
        print(f"Case {i}: expected=ANSWER route={response['route']}")
        print(f"Message: {response['message']}")
        #print(f"Reason: {response['reason']}")
        print(f"Request time (s): {elapsed:.6f}")
        if response["response_data"] is not None:
            summary = response["response_data"].metrics.get_summary()
            last_usage = summary["agent_invocations"][-1]["usage"]
            print(f"Per-call usage: {last_usage}")
        print("-------------------------------------")
    #print(get_tool_trace())


if ANSWER_LOGISTICS_CASES:
    reset_session()
    set_session_customer(1208, "test_user")

    for i, case in enumerate(answer_logistics_cases):
        start = time.perf_counter()
        response = solve_inventory_query(case, query_type="PRIVATE")
        elapsed = time.perf_counter() - start

        print(str(case))
        print(f"Case {i}: expected=ANSWER route={response['route']}")
        print(f"Message: {response['message']}")
        #print(f"Reason: {response['reason']}")
        print(f"Request time (s): {elapsed:.6f}")
        if response["response_data"] is not None:
            summary = response["response_data"].metrics.get_summary()
            last_usage = summary["agent_invocations"][-1]["usage"]
            print(f"Per-call usage: {last_usage}")
        print("-------------------------------------")


if ANSWER_RETURN_WARRANTY_CASES:
    reset_session()
    set_session_customer(1292, "test_user_1292")

    for i, case in enumerate(answer_return_warranty_cases):
        start = time.perf_counter()
        response = solve_inventory_query(case, query_type="PRIVATE")
        elapsed = time.perf_counter() - start

        print(str(case))
        print(f"Case {i}: expected=ANSWER route={response['route']}")
        print(f"Message: {response['message']}")
        #print(f"Reason: {response['reason']}")
        print(f"Request time (s): {elapsed:.6f}")
        if response["response_data"] is not None:
            summary = response["response_data"].metrics.get_summary()
            last_usage = summary["agent_invocations"][-1]["usage"]
            print(f"Per-call usage: {last_usage}")
        print("-------------------------------------")
    print("Tool trace snapshot:")
    #print(get_tool_trace())



if NO_DATA_CASES:
    reset_session()
    set_session_customer(1292, "test_user_1292")

    for i, case in enumerate(no_data_cases):
        start = time.perf_counter()
        response = solve_inventory_query(case, query_type="PRIVATE")
        elapsed = time.perf_counter() - start

        print(str(case))
        print(f"Case {i}: expected=NO_DATA route={response['route']}")
        print(f"Message: {response['message']}")
        #print(f"Reason: {response['reason']}")
        print(f"Request time (s): {elapsed:.6f}")
        if response["response_data"] is not None:
            summary = response["response_data"].metrics.get_summary()
            last_usage = summary["agent_invocations"][-1]["usage"]
            print(f"Per-call usage: {last_usage}")
        print("-------------------------------------")


