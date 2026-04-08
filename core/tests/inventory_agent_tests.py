from core.agents.inventory_agent import solve_inventory_query
from core.session_context import reset_session, set_session_customer, get_tool_trace

RUN_AUTH_REQUIRED = False
RUN_ANSWER_PRODUCTS = False
RUN_ANSWER_ORDERS = False
RUN_ANSWER_LOGISTICS = False
RUN_ANSWER_RETURN_WARRANTY = False
RUN_NO_DATA = True


# -----------------------------
# AUTH_REQUIRED CASES
# -----------------------------
auth_required_cases = [
    "donde esta mi pedido?",
    "cual es el estado de mi orden 74?",
    "que compre en mi ultima orden?",
    "el item 221 tiene garantia?",
    "puedo devolver el item 221 de la orden 74?",
]


# -----------------------------
# ANSWER: PRODUCTS / INVENTORY
# -----------------------------
answer_products_cases = [
    "tienen stock de samsung?",
    "cuanto cuesta el producto 5056?",
    "muestrame detalle del producto 5056",
]


# -----------------------------
# ANSWER: ORDERS
# -----------------------------
answer_orders_cases = [
    "muestrame mis pedidos",
    "muestrame los detalles de mi orden 74",
    "que productos vienen en mi orden 74?",
]


# -----------------------------
# ANSWER: LOGISTICS
# -----------------------------
answer_logistics_cases = [
    "donde esta mi pedido?",
    "cual es el estado de mi orden 74?",
    "mi pedido ya fue enviado?",
    "cual es el tracking de mi orden 74?",
    "cuando llega mi pedido 74?",
]


# -----------------------------
# ANSWER: RETURN / WARRANTY
# -----------------------------
answer_return_warranty_cases = [
    "puedo devolver el item 221 de la orden 74?",
    "el item 221 de la orden 74 aun tiene garantia?",
    "la orden 74 item 221 es elegible para devolucion?",
]


# -----------------------------
# NO_DATA CASES
# -----------------------------
no_data_cases = [
    "dame el estado de la orden 999999",
    "quiero garantia del item 999999 de la orden 74",
]


if RUN_AUTH_REQUIRED:
    reset_session()
    for i, case in enumerate(auth_required_cases):
        response = solve_inventory_query(case)

        print(str(case))
        print(f"Case {i}: expected=AUTH_REQUIRED route={response['route']}")
        print(f"Message: {response['message']}")
        print(f"Reason: {response['reason']}")
        print("-------------------------------------")


if RUN_ANSWER_PRODUCTS:
    reset_session()
    set_session_customer(1234, "test_user")

    for i, case in enumerate(answer_products_cases):
        response = solve_inventory_query(case)

        print(str(case))
        print(f"Case {i}: expected=ANSWER route={response['route']}")
        print(f"Message: {response['message']}")
        print(f"Reason: {response['reason']}")
        if response["response_data"] is not None:
            summary = response["response_data"].metrics.get_summary()
            last_usage = summary["agent_invocations"][-1]["usage"]
            print(f"Avg cycle (s): {summary['average_cycle_time']}")
            print(f"Per-call usage: {last_usage}")
        print("-------------------------------------")


if RUN_ANSWER_ORDERS:
    reset_session()
    set_session_customer(1234, "test_user")

    for i, case in enumerate(answer_orders_cases):
        response = solve_inventory_query(case)

        print(str(case))
        print(f"Case {i}: expected=ANSWER route={response['route']}")
        print(f"Message: {response['message']}")
        print(f"Reason: {response['reason']}")
        if response["response_data"] is not None:
            summary = response["response_data"].metrics.get_summary()
            last_usage = summary["agent_invocations"][-1]["usage"]
            print(f"Avg cycle (s): {summary['average_cycle_time']}")
            print(f"Per-call usage: {last_usage}")
        print("-------------------------------------")


if RUN_ANSWER_LOGISTICS:
    reset_session()
    set_session_customer(1234, "test_user")

    for i, case in enumerate(answer_logistics_cases):
        response = solve_inventory_query(case)

        print(str(case))
        print(f"Case {i}: expected=ANSWER route={response['route']}")
        print(f"Message: {response['message']}")
        print(f"Reason: {response['reason']}")
        if response["response_data"] is not None:
            summary = response["response_data"].metrics.get_summary()
            last_usage = summary["agent_invocations"][-1]["usage"]
            print(f"Avg cycle (s): {summary['average_cycle_time']}")
            print(f"Per-call usage: {last_usage}")
        print("-------------------------------------")


if RUN_ANSWER_RETURN_WARRANTY:
    reset_session()
    set_session_customer(1234, "test_user")

    for i, case in enumerate(answer_return_warranty_cases):
        response = solve_inventory_query(case)

        print(str(case))
        print(f"Case {i}: expected=ANSWER route={response['route']}")
        print(f"Message: {response['message']}")
        print(f"Reason: {response['reason']}")
        if response["response_data"] is not None:
            summary = response["response_data"].metrics.get_summary()
            last_usage = summary["agent_invocations"][-1]["usage"]
            print(f"Avg cycle (s): {summary['average_cycle_time']}")
            print(f"Per-call usage: {last_usage}")
        print("-------------------------------------")
        print("Tool trace snapshot:")
        print(get_tool_trace())



if RUN_NO_DATA:
    reset_session()
    set_session_customer(1234, "test_user")

    for i, case in enumerate(no_data_cases):
        response = solve_inventory_query(case)

        print(str(case))
        print(f"Case {i}: expected=NO_DATA route={response['route']}")
        print(f"Message: {response['message']}")
        print(f"Reason: {response['reason']}")
        if response["response_data"] is not None:
            summary = response["response_data"].metrics.get_summary()
            last_usage = summary["agent_invocations"][-1]["usage"]
            print(f"Avg cycle (s): {summary['average_cycle_time']}")
            print(f"Per-call usage: {last_usage}")
        print("-------------------------------------")


