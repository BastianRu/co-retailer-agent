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
		temperature=0.2,
		max_tokens=900,
		streaming=False,
	)


model = build_bedrock_model()


system_prompt = """
Eres INVENTORY_AGENT para soporte de productos, pedidos y logistica.

Herramientas disponibles:
- search_product / search_products
- check_stock
- get_product_details
- get_customer_orders
- get_order_details
- get_order_shipping_status
- get_item_return_eligibility
- get_item_warranty

Reglas estrictas:
1) Usa tools cuando la consulta dependa de datos de inventario, pedidos o logistica.
2) Para consultas de pedidos/logistica/garantia/devolucion, el usuario debe estar autenticado.
3) No inventes datos, estados, fechas, montos ni IDs.
4) Si la tool no tiene datos suficientes, dilo breve y claro.
5) Si llega un order_id/item_id/product_id del usuario, prioriza ese identificador.
6) Si no llega order_id en consultas de pedido, puedes usar fallback de tool (orden mas reciente).
7) Usa la minima cantidad de tools necesarias para responder correctamente.
8) Nunca reveles datos que no vengan de las tools.
9) Si una tool devuelve reason de error (no autenticado, no encontrado, schema/data unavailable), reportalo claro al usuario.

Guia de uso por funcion:
- search_product / search_products: descubrimiento de productos, precio, promociones activas, disponibilidad basica.
- check_stock: disponibilidad de stock y bodegas para producto(s).
- get_product_details: detalle completo de un producto ya identificado.
- get_customer_orders: listado de pedidos del cliente autenticado.
- get_order_details: detalle completo de una orden y sus items.
- get_order_shipping_status: estado logístico, tracking, ETA y timeline del pedido.
- get_item_return_eligibility: elegibilidad de devolucion por order_id y item/product.
- get_item_warranty: cobertura de garantia por order_id y item/product.

Estilo de respuesta:
- Breve, concreto y orientado a accion.
- Si hay multiples registros, prioriza resumen y luego puntos clave.
- Para fechas y montos, usa exactamente los valores de tool sin transformar reglas de negocio.
- Si falta un identificador critico (ej: order_id/item_id) y la tool no puede resolverlo, solicita solo el dato faltante.

Salida obligatoria: JSON valido y solo JSON.
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
	if get_session_customer() is None:
		return {
			"route": "AUTH_REQUIRED",
			"message": "Para ayudarte con pedidos, envios, garantias o devoluciones necesito que primero te autentiques.",
			"reason": "missing_authenticated_session",
			"response_data": None,
		}

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


