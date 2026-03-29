import json
from core.routers.auth_routing import classify_intent
import re

#Public cases
public_cases = [
    "¿Qué métodos de pago manejan?",
    "¿Hacen envíos a todo Colombia?",
    "¿Cuánto tarda un envío a una ciudad principal?",
    "¿Cuál es la política de devoluciones?",
    "¿Cómo funciona la garantía?",
    "¿Puedo cambiar un producto en general?",
    "¿Cuánto cuesta el producto Samsung Galaxy A55?",
    "¿Tienen stock del producto Air Fryer Oster?",
    "¿Cuáles son sus canales de atención?",
    "¿Cómo hago seguimiento a un pedido?",
    "¿Se puede cambiar la dirección antes del despacho?",
    "¿Qué cubre la garantía de electrónica?"
]

#Private cases 
private_cases = [
    "¿Cuál es el total de mi pedido?",
    "¿Cuánto pagué de IVA en mi compra?",
    "¿Dónde está mi pedido?",
    "Muéstrame el historial de envío del pedido 1234.",
    "Quiero saber si mi compra ya fue entregada.",
    "¿Mi producto todavía tiene garantía?",
    "¿Mi pedido ya fue despachado?",
    "¿Cuándo llegará mi pedido?",
    "Quiero devolver el pedido 8472.",
    "¿Cuál fue el subtotal de mi orden?",
    "¿Cuántos intentos de entrega tuvo mi pedido?",
    "Quiero saber el estado de mi reembolso.",
    "Quiero revisar una devolución.",
    "Necesito ayuda con una garantía.",
    "Quiero saber sobre mi envío.",
    "Necesito el estado del pedido.",
    "Quiero consultar un reembolso.",
    "Ayúdame con una compra que hice.",
    #With ids
    "Consulta el pedido 100245.",
    "Revisa el tracking de la orden 8472.",
    "¿Cuál fue el total del pedido 9321?",
    "Quiero devolver el producto del pedido 4567.",
    "¿El pedido 2211 ya fue entregado?"
]

#ambiguous cases
ambiguous_cases = [
    "¿Puedo devolver este producto?",
    "Quiero saber sobre la garantía de mi producto.",
    "¿Cómo es el proceso de reembolso?",
    "¿Qué pasa si mi pedido se demora?",
    "Quiero saber sobre cambios de productos.",
    "¿Qué cubre la garantía en mi caso?",
    "Necesito información sobre una devolución.",
    "¿Puedo cancelar una compra?",
    "¿Cómo funciona el seguimiento del pedido?",
    "¿Qué pasa si el producto llega dañado?"
]

#Conversational-like cases
conversational_cases = [
    "Hola, quiero saber el estado de las políticas de devolución actuales.",
    "Necesito información sobre el precio y las políticas de garantía.",
    "Quiero consultar las condiciones actuales de envío y devoluciones.",
    "¿Cuál es la política actual sobre productos en promoción y devoluciones?",
    "Envío este mensaje para preguntar por las políticas de precios.",
    "¿Cómo están manejando actualmente las garantías y los cambios?",
    "Hola, buenas, quería saber si hacen envíos a Pasto.",
    "Buenas tardes, me gustaría saber qué medios de pago aceptan.",
    "Hola, quisiera preguntar cuánto tarda normalmente un envío rural.",
    "Estoy viendo un producto y quería saber si tienen envío gratis.",
    "Antes de comprar, ¿puedo saber si manejan cambios?"
]

#Special cases
special_cases = [
    "Quiero entender mejor mis opciones de devolución en general.",
    "Quiero conocer mis derechos de garantía como cliente.",
    "¿Cómo puedo saber si una compra tiene envío gratis?",
    "Quiero revisar mis posibilidades de cambio antes de comprar."
]

#Custom cases
custom_cases = [" tienen este producto a la venta? [imagen de producto x] "]

for i, case in enumerate(custom_cases):
    #response handling
    response = classify_intent(case)
    raw = str(response).strip()

    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.replace("json", "", 1).strip()

    try:
        data = json.loads(raw)
        classification = data.get("classification", "UNKNOWN")
    except json.JSONDecodeError:
        m = re.search(r"\b(PUBLICO|PRIVADO|AMBIGUO)\b", raw.upper())
        classification = m.group(1) if m else "UNKNOWN"

 
    #Results and metrics for each cycle
    summary = response.metrics.get_summary()
    last_usage = summary["agent_invocations"][-1]["usage"]
    print(f"Case {i}: {classification}")
    print(f"Reason: {data["reasoning"]}")
    print(f"Avg cycle (s): {summary['average_cycle_time']}")
    print(f"Per-call usage: {last_usage}")              
    print("-------------------------------------")



