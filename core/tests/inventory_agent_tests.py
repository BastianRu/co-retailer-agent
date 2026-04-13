import time

from core.agents.inventory_agent import solve_inventory_query
from core.session_context import reset_session, set_session_customer, get_tool_trace

CATALOG_CASES = True
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

CUSTOM_CASES = False
custom_cases = [
    #"dame los detalles del iphone 15 pro max",
    #"dame los detalles del monitor LG ultra wide",
    #"que productos samsung tienen?",
    #"que celular recomiendas?",
    "si"
]

# Test para ambigüedad excesiva
AMBIGUITY_CASES = False
ambiguity_test_cases = [
    # Casos sin contexto (AGENT_HISTORY vacío)
    ("Quiero información", []),
    ("Necesito ayuda", []),
    ("Tengo una pregunta", []),
    
    # Casos con múltiples agentes sin suficiente semántica
    ("Eso", ["PUBLIC_INVENTORY", "RAG", "PRIVATE_INVENTORY"]),
    ("Lo otro", ["PUBLIC_INVENTORY", "PRIVATE_INVENTORY"]),
    
    # Casos que SÍ deben ser FOLLOW_QUERY (tienen semántica)
    ("Sí", ["PUBLIC_INVENTORY"]),
    ("¿Y cuánto cuesta?", ["PUBLIC_INVENTORY"]),
    ("¿Mi pedido?", ["RAG", "PRIVATE_INVENTORY"]),
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


# Test INPUT AGENT - AMBIGÜEDAD
if AMBIGUITY_CASES:
    from core.agents.input_agent import classify_input
    from core.session_context import get_agent_history
    
    print("\n\n=== PRUEBAS DE AMBIGÜEDAD EN INPUT_AGENT ===\n")
    
    reset_session()
    
    for i, (message, history) in enumerate(ambiguity_test_cases):
        # Simulamos inyectar la historia en el contexto
        # (En la práctica, classify_input usa get_agent_history(), así que solo resetamos)
        
        start = time.perf_counter()
        result = classify_input(message)
        elapsed = time.perf_counter() - start
        
        print(f"Caso {i}: '{message}'")
        print(f"  Route: {result['route']}")
        print(f"  Follow Route: {result['follow_query_route']}")
        print(f"  Message: {result['message']}")
        print(f"  Reason: {result['reason']}")

ORCHESTRATOR_TEST = False
if ORCHESTRATOR_TEST:
    from core.agent import create_agent
    
    print("\n\n=== PRUEBAS DE ORCHESTRATOR - CONVERSACIÓN COMPLETA ===\n")
    
    # Test 1: Ambigüedad en consulta inicial
    print("--- Test 1: Ambigüedad sin contexto ---")
    agent = create_agent()
    response = agent("Quiero información")
    print(f"User: 'Quiero información'")
    print(f"Agent: {response}")
    print()
    
    # Test 2: Consulta clara → FOLLOW_QUERY con continuidad
    print("--- Test 2: Continuidad después de producto ---")  
    agent = create_agent()
    response = agent("¿Hay stock del Samsung Galaxy S24 Ultra?")
    print(f"User: '¿Hay stock del Samsung Galaxy S24 Ultra?'")
    print(f"Agent: {response}")
    print()
    
    response = agent("¿Y cuánto cuesta?")
    print(f"User: '¿Y cuánto cuesta?'")
    print(f"Agent: {response}")
    print()
    
    # Test 3: Consulta PRIVATE sin autenticación
    print("--- Test 3: PRIVATE_INVENTORY sin autenticación (debe pedir AUTH) ---")
    agent = create_agent()
    response = agent("¿Cuál es el estado de mi pedido?")
    print(f"User: '¿Cuál es el estado de mi pedido?'")
    print(f"Agent: {response}")
    print()
    
    # Test 4: FOLLOW_QUERY a PRIVATE_INVENTORY sin auth
    print("--- Test 4: Continuidad a PRIVATE_INVENTORY sin autenticación ---")
    agent = create_agent()
    response = agent("Dame detalles del iphone 14")
    print(f"User: '¿Tienes iphone 14?'")
    print(f"Agent: {response}")
    print()
    
    response = agent("Quiero los detalles de mi orden")
    print(f"User: 'Quiero los detalles de mi orden'")
    print(f"Agent: {response}")
    print()




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


