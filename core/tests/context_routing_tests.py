from core.routers.context_routing import classify_context_route
import json
import re

auto_cases = [
    "¿Qué métodos de pago manejan?",
    "¿Cuánto cuesta el iPhone 13?",
    "¿Tienen stock del Samsung S24?",
    "¿Cuál es la política de devoluciones?",
    "¿Cuánto tarda un envío a Bogotá?",
    "¿Cómo descargo la factura?",
    "¿Qué cubre la garantía?",
    "Quiero saber el estado de mi pedido",
    "¿Cuántos intentos de entrega hacen?",
    "¿Puedo cancelar una compra antes del despacho?",
    "¿Qué productos tienen descuento actualmente?",
    "¿Cómo funciona el seguimiento del pedido?",
    "¿Cuál es el precio de la air fryer Oster?",
    
    # ruido realista
    "Oe, ¿cuánto vale ese celular Samsung?",
    "Buenas, ¿ustedes hacen envíos a Pasto?",
    "Parcero, ¿qué medios de pago aceptan?",
    "Hola, necesito saber si tienen stock de este producto"
]


follow_cases = [
    "Disculpa, ¿cuánto cuesta ese producto?",
    "¿Hay disponibilidad de este producto?",
    #"¿Este producto tiene envío gratis?",
    "¿Lo tienen?",
    "¿Y cuánto vale?",
    "¿Todavía hay?",
    "¿Y aplica garantía?",
    "¿Y si ya fue despachado?",
    "¿Eso también tiene envío gratis?",
    "¿Y ese?",
    "¿Ese cuánto cuesta?",
    "¿Y hay más unidades?",
    "¿Y cuánto demora?",
    "¿Eso se puede devolver?",
    "¿Y si está en promoción?",
    "¿También aplica?",
    "¿Y el otro?",
    "¿Y ese modelo?",
    
    # combinaciones más tricky
    "¿Y cuánto vale ese?",
    "¿Todavía lo venden?",
    "¿Eso sí tiene stock?",
    "¿Y si lo compro hoy?",
    "¿Y eso incluye envío?"
]

talk_cases = [
    "hola",
    "buenas",
    "holaaa",
    "hey",
    "gracias",
    "muchas gracias",
    "ok",
    "dale",
    "listo",
    "perfecto",
    "super",
    "genial",
    "vale",
    "de una",
    "jajaja",
    "jeje",
    
    # más realistas
    "ok gracias",
    "listo parcero",
    "gracias bro",
    "todo bien",
    "bien bien",
    "perfecto gracias",
    "ah bueno",
    "vale vale",
    "oki",
    "todo claro"
    "eyyyy como estamooo"
]

boundary_cases = [
    # pueden parecer FOLLOW pero son AUTO
    "¿Este producto tiene garantía?",
    "¿Ese producto tiene descuento?",
    
    # pueden parecer AUTO pero son FOLLOW
    "¿Y ese producto?",
    "¿Y el envío?",
    
    # pueden parecer TALK pero son FOLLOW
    "ok, ¿y cuánto vale?",
    "listo, ¿y hay stock?",
    
    # ambiguos reales
    "¿y entonces?",
    "¿y eso?",
    "¿y ahí qué?",
]

for i, case in enumerate(boundary_cases):
    #response handling
    response = classify_context_route(case)
    #Results and metrics for each cycle
    summary = response["response_data"].metrics.get_summary()
    last_usage = summary["agent_invocations"][-1]["usage"]
    print(str(case))
    print(f"Case {i}: {response["route"]}")
   # print(f"Reason: {data["reasoning"]}")
    print(f"Avg cycle (s): {summary['average_cycle_time']}")
    print(f"Per-call usage: {last_usage}")              
    print("-------------------------------------")

