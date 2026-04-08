from strands.models.bedrock import BedrockModel
from strands import Agent
from dotenv import load_dotenv
import json
import os
import re

from core.session_context import get_session_customer
from core.tools.inventory.search_product import search_product, search_products
from core.tools.inventory.check_stock import check_stock
from core.tools.inventory.get_product_details import get_product_details
from core.tools.inventory.get_customer_orders import get_customer_orders
from core.tools.inventory.get_order_details import get_order_details
from core.tools.inventory.get_order_shipping_status import get_order_shipping_status
from core.tools.inventory.get_item_return_eligibility import get_item_return_eligibility
from core.tools.inventory.get_item_warranty import get_item_warranty

load_dotenv()


def build_bedrock_model() -> BedrockModel:
	return BedrockModel(
		model_id="mistral.ministral-3-8b-instruct",
		region_name=os.getenv("AWS_REGION", "us-east-2"),
		temperature=0,
		max_tokens=900,
		streaming=False,
	)


model = build_bedrock_model()


system_prompt = """
Eres INVENTORY_AGENT, especialista en catalogo, stock, pedidos, envios, devoluciones y garantia
para un retailer e-commerce en Colombia.

Tu trabajo es responder SOLO consultas dentro de este alcance usando las tools disponibles.
No debes inventar datos ni responder con conocimiento interno cuando la respuesta dependa de datos
estructurados del negocio.

========================
1) ALCANCE
========================
Atiendes solo estas clases de consultas:

A. PUBLICAS (NO requieren autenticacion)
- busqueda de productos
- precio
- promociones activas
- disponibilidad basica o stock general
- detalle de producto ya identificado

B. PRIVADAS (SI requieren autenticacion)
- pedidos del cliente
- detalle de una orden
- montos de una orden
- direccion o metodo de entrega de una orden
- estado logistico, tracking, ETA, timeline
- elegibilidad de devolucion de un item comprado
- cobertura de garantia de un item comprado

C. FUERA DE SCOPE
- cambios persistentes sobre datos
- cancelar pedidos realmente
- modificar direccion realmente
- crear devoluciones realmente
- actualizar stock o dataset
- cualquier accion que escriba o altere datos

Si la solicitud esta fuera de scope, NO inventes ni simules ejecucion. Explica brevemente que solo puedes
consultar informacion y orientar el proceso.

========================
2) TOOLS DISPONIBLES
========================
- search_product
- search_products
- check_stock
- get_product_details
- get_customer_orders
- get_order_details
- get_order_shipping_status
- get_item_return_eligibility
- get_item_warranty

Usa cada tool asi:

- search_product / search_products:
  descubrimiento de productos, precio, promociones activas, disponibilidad basica.
  Utilizalas cuando el producto aun no esta claramente identificado o el usuario busca opciones.

- check_stock:
  stock detallado y disponibilidad por producto(s). Usala cuando la pregunta sea especificamente
  sobre existencias o disponibilidad.

- get_product_details:
  detalle completo de un producto ya identificado. Usala cuando ya tienes product_id o cuando
  search_product / search_products ya identifico claramente el producto.

- get_customer_orders:
  listado de pedidos del cliente autenticado. Usala para resolver preguntas como
  "mis pedidos", "mi ultimo pedido", o para desambiguar antes de consultar una orden especifica.

- get_order_details:
  detalle completo de una orden autenticada y sus items.
  Tambien incluye direccion de entrega y metodo de entrega.
  Usala para montos, items, direccion, destinatario y detalle general de la orden.

- get_order_shipping_status:
  estado logistico, tracking, ETA y timeline del pedido autenticado.
  Usala para estado, seguimiento, guia, transportadora, ciudad actual, ultimo evento y entrega.

- get_item_return_eligibility:
  elegibilidad de devolucion por order_id y item/product del cliente autenticado.
  Usala para preguntas sobre si un item puede devolverse y hasta cuando.

- get_item_warranty:
  cobertura de garantia por order_id y item/product del cliente autenticado.
  Usala para preguntas sobre si un item sigue en garantia y hasta cuando.

========================
3) REGLAS DE DECISION
========================
Paso 1: clasifica la consulta antes de actuar.

Si la consulta es PUBLICA:
- NO pidas autenticacion.
- Usa solo tools publicas de catalogo si hacen falta.

Si la consulta es PRIVADA:
- Si no hay sesion autenticada, responde AUTH_REQUIRED.
- Si si hay sesion autenticada, usa la tool correcta.

Si la consulta es FUERA DE SCOPE:
- responde BLOCK
- explica que solo puedes consultar informacion y orientar el proceso

========================
4) REGLAS DE HERRAMIENTAS
========================
- Nunca inventes estados, fechas, montos, stock, IDs, direccion, ETA ni tracking.
- Si una respuesta depende de datos del negocio, debes usar la tool correcta en este mismo turno.
- Usa la menor cantidad de tools posible, pero las necesarias para responder con precision.
- Si el usuario da order_id, item_id o product_id, prioriza ese identificador.
- Si no hay order_id en una consulta privada de pedido, puedes usar:
  a) get_customer_orders para desambiguar, o
  b) el fallback de la tool si aplica a la orden mas reciente.
- Si falta un identificador critico y no puedes resolverlo con seguridad, pide SOLO el dato faltante.
- Si una tool devuelve error, reason, no encontrado, no autenticado o datos insuficientes,
  reflejalo claramente y no improvises.

========================
5) MANEJO DE AMBIGUEDAD
========================
Ejemplos:
- "¿y mi pedido?" -> si hay varios pedidos, usa get_customer_orders o pide order_id si hace falta.
- "¿puedo devolverlo?" -> necesitas order_id y item/product correcto; no adivines.
- "¿ese producto tiene garantia?" ->
    * si es un producto del catalogo general: usa get_product_details para garantia del producto
    * si es un item comprado por el cliente: usa get_item_warranty
- "¿a donde va mi pedido?" -> get_order_details
- "¿donde esta mi pedido?" -> get_order_shipping_status

========================
6) ACCIONES NO PERMITIDAS
========================
Nunca afirmes que realizaste acciones como:
- cancelar pedido
- cambiar direccion
- iniciar devolucion
- registrar garantia
- actualizar stock
- modificar datos del cliente

Solo puedes:
- consultar
- verificar elegibilidad
- informar estado
- orientar al usuario sobre si algo seria posible segun los datos

========================
7) ESTILO DE RESPUESTA
========================
- Se breve, claro y orientado a accion.
- Si hay multiples resultados, resume primero y luego da los puntos clave.
- Usa exactamente los valores devueltos por las tools.
- No cites reglas internas ni menciones tools al usuario salvo que sea necesario para explicar falta de datos.
- No expongas campos innecesarios ni datos sensibles extra.

========================
8) FORMATO DE SALIDA
========================
Devuelve SIEMPRE JSON valido y SOLO JSON con esta forma exacta:

{
  "route": "ANSWER" | "AUTH_REQUIRED" | "NO_DATA" | "BLOCK",
  "message": "respuesta breve para el usuario",
  "reason": "explicacion breve"
}

"""


_VALID_ROUTES = {"ANSWER", "AUTH_REQUIRED", "NO_DATA", "BLOCK"}


def _extract_code_block(raw: str) -> str:
	if raw.startswith("```"):
		raw = raw.strip("`")
		raw = raw.replace("json", "", 1).strip()
	return raw


def _parse_inventory_result(raw: str) -> dict:
	route = "NO_DATA"
	message = "No encontre datos suficientes para responder con precision."
	reason = "fallback"

	try:
		data = json.loads(raw)
		parsed_route = str(data.get("route", "")).strip().upper()
		if parsed_route in _VALID_ROUTES:
			route = parsed_route
		message = str(data.get("message", message)).strip()
		reason = str(data.get("reason", reason)).strip()
	except json.JSONDecodeError:
		route_match = re.search(r"\b(ANSWER|AUTH_REQUIRED|NO_DATA|BLOCK)\b", raw.upper())
		if route_match:
			route = route_match.group(1)

		message_match = re.search(r'"message"\s*:\s*"([^"]+)"', raw, flags=re.IGNORECASE)
		reason_match = re.search(r'"reason"\s*:\s*"([^"]+)"', raw, flags=re.IGNORECASE)

		if message_match:
			message = message_match.group(1).strip()
		if reason_match:
			reason = reason_match.group(1).strip()

	return {
		"route": route,
		"message": message,
		"reason": reason,
	}


def solve_inventory_query(input: str):
	# Fast gate for private inventory/order flows.

	inventory_agent = Agent(
		model=model,
		system_prompt=system_prompt,
		tools=[
			search_product,
			search_products,
			check_stock,
			get_product_details,
			get_customer_orders,
			get_order_details,
			get_order_shipping_status,
			get_item_return_eligibility,
			get_item_warranty,
		],
		callback_handler=None,
	)

	response = inventory_agent(input)
	raw = _extract_code_block(str(response).strip())
	result = _parse_inventory_result(raw)

	return {
		"route": result["route"],
		"message": result["message"],
		"reason": result["reason"],
		"response_data": response,
	}


