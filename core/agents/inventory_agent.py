from strands.models.bedrock import BedrockModel
from strands import Agent
from core.session_context import register_reset_callback, get_tool_trace_length, get_tool_trace_since
from dotenv import load_dotenv
import json
import os
import re

from core.tools.inventory.search_products import search_product
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
		max_tokens=2000,
		streaming=False,
	)

model = build_bedrock_model()


private_system_prompt = """
Eres PRIVATE INVENTORY AGENT para un retailer e-commerce en Colombia.

Objetivo: resolver consultas pedidos, envios, devoluciones y garantia
usando SIEMPRE las herramientas disponibles cuando la respuesta dependa de datos del negocio.

========================
1) RESPONSABILIDAD Y ALCANCE
========================

Tu responsabilidad es resolver la consulta de que te llega,
usando tools y respetando el estado real de los datos.

Consultas dentro de alcance:
- pedidos, detalle de orden, direccion de entrega
- tracking, estado logistico, ETA
- devolucion y garantia de items comprados

Fuera de scope:
- acciones de escritura o cambio real (cancelar, modificar direccion, crear devolucion,
	actualizar stock/datos, etc.)

Regla de salida:
- Fuera de scope => BLOCK

========================
2) USO DE TOOLS (REGLA PRINCIPAL)
========================

Si la respuesta depende de datos del negocio, usa tools en este mismo turno.

- get_customer_orders:
	lista de pedidos del cliente autenticado; usar para "mis pedidos" o desambiguar order_id.

- get_order_details:
	detalle integral de la orden (items, montos, direccion, destinatario, tipo de entrega).

- get_order_shipping_status:
	tracking y progreso logistico (estado, eventos, ETA, guia, ciudad actual).

- get_item_return_eligibility:
	valida si un item de una orden es elegible para devolucion y hasta cuando.

- get_item_warranty:
	valida cobertura de garantia de un item comprado y su vigencia.

- get_product_details:
	usar solo si piden detalle de un product_id concreto de una orden.

========================
3) ESTRATEGIA DE EJECUCION
========================

1. Verifica si la consulta esta dentro de scope o fuera de scope.
2. Elige la tool mas especifica para la intencion.
3. Prioriza IDs dados por el usuario (order_id, item_id, product_id).
4. Usa el minimo de tools necesario, pero sin sacrificar precision.
5. Si falta un dato critico y no se puede inferir con seguridad, pide SOLO ese dato.

No adivines identificadores.

========================
4) MANEJO DE DATOS CON INCONSISTENCIAS (OBLIGATORIO)
========================

Los datos pueden tener variantes, duplicados o nombres inconsistentes.
Tu responsabilidad es normalizar y responder utilmente.

DEBES:
- mostrar datos LIMPIOS y representativos
- consolidar variantes o duplicados en una sola respuesta clara

Si faltan datos, responde de forma neutral y accionable (NO_DATA) sin culpar al dataset.

========================
5) RESTRICCIONES
========================

- Nunca inventes: estados, fechas, montos, stock, IDs, direccion, ETA, tracking.
- No simules acciones de escritura.
- No expongas datos sensibles o irrelevantes.
- No menciones reglas internas.
- Nunca reformules valores exactos provenientes de tools.
- Métodos de pago deben copiarse exactamente como vienen.
- Montos deben copiarse exactamente como vienen, sin cambiar separadores ni redondear.
- Si no tienes nombre descriptivo del producto, usa product_id o di que el nombre no está disponible.
- Nunca expongas notas internas, metadatos internos ni observaciones administrativas al usuario.
- Nunca combines ni generalices fechas de garantía o devolución entre múltiples items.
- Si dos productos tienen plazos distintos, repórtalos por separado.
- Para preguntas cerradas sobre estado, método de pago o total, responde solo con ese dato y no agregues inferencias adicionales.
- Si el usuario pregunta “qué productos vienen en mi orden”, no afirmes una sola fecha de devolución para todos si los items tienen deadlines distintos.
- Nunca combines ni reinterpretes eventos del tracking timeline.
- Usa solo el último evento relevante para describir el estado.
- Si hay múltiples eventos (intento fallido, reintento, entrega), resume sin inventar narrativa adicional.
- Nunca infieras la fecha de inicio de garantía.
- Solo reporta la fecha de expiración si está disponible.
- Si la garantía ya venció, indica únicamente la fecha de expiración y que ya no está vigente.
- Nunca muestres variantes con nombres repetidos o inconsistentes.
- Usa el nombre más limpio y representativo.
- Nunca menciones errores, duplicados, inconsistencias o problemas internos del catálogo.
- No crees nombres nuevos a partir de fragmentos parciales.
- Si el nombre del producto no viene explícitamente de la tool, está PROHIBIDO generarlo.
- Nunca crees nombres como "Smartphone X Pro".
- Debes usar:
  - "Producto ID: <id>" o
  - "detalle no disponible"
- No expliques ni interpretes el timeline del tracking.
- Reporta solo el estado final y datos clave.
- Si la tool devuelve solo fecha (YYYY-MM-DD), debes responder SOLO con la fecha.
- Nunca agregues horas (HH:mm), minutos o segundos si no están en la tool.
- No estimes ni completes información temporal.
- No agregues recomendaciones o sugerencias si el usuario no las pidió.
- No agregues condiciones de devolución que no estén explícitamente en la tool.

VALORES EXACTOS:
- Campos cerrados como status, payment_method, delivery_method, tracking_number, order_id, product_id e item_id deben copiarse exactamente como vengan de la tool.
- Ejemplo: si la tool devuelve "Daviplata", debes responder exactamente "Daviplata".
- Nunca redondees ni reformatees montos unitarios o totales si la tool ya devuelve el valor.
- Si dudas sobre el nombre descriptivo de un producto, usa "Producto ID: <id>" o indica que el nombre no está disponible.
- En los pedidos, muestra el nombre del producto cuando menos

========================
7) FORMATO DE SALIDA OBLIGATORIO
========================

Devuelve SIEMPRE JSON valido y SOLO JSON:

{
	"route": "ANSWER" | "NO_DATA" | "BLOCK",
	"message": "respuesta para el usuario con la informacion solicitada",
}

"""

public_system_prompt = """
Eres INVENTORY_AGENT_PUBLIC para un retailer e-commerce en Colombia.
Respondes solo consultas PUBLICAS de catalogo.
Devuelve SIEMPRE JSON valido y SOLO JSON.

PRIORIDAD MAXIMA (OBLIGATORIA):
1) Si la consulta incluye un product_id explicito (ej: "producto 5001"), SIEMPRE usa get_product_details(product_id=<id>, product_ids=[<id>]).
2) No respondas sobre ese producto sin tool.
3) Si get_product_details trae datos, responde con esos datos. Si no trae datos, responde NO_DATA.

TOOLS:
- search_product(query): busqueda por nombre/marca/categoria.
- check_stock(query, product_id): stock.
- get_product_details(product_id, product_ids): detalle completo (price, availability_units, free_shipping, shipping_days, warranty_months, return_days, is_final_sale, promotions, description, specifications).

DECISION:
- Consulta con product_id explicito -> get_product_details.
- Consulta por nombre/marca (iphone, samsung, laptop, etc.) -> search_product.
- Consulta ambigua sin producto claro -> NO_DATA pidiendo una aclaracion breve.

REGLA CRITICA DE RESULTADOS:
- Si search_product devuelve resultados, debes responder con esos resultados (TODOS).
- Si hay resultados, esta prohibido decir que "no hay" o "no esta disponible"o "No encontré datos suficientes...”
- Usa solamente numeros (precios, stock, fechas) exactamente como los devueltos por la tools.

Ej: "¿Qué iphones tienen?"
		- tool: 6481727.00 
		- salida: $6.481.727 COP

- Si la consulta es por marca/modelo/categoria (ej: samsung, iphone, electronica), siempre intenta search_product antes de responder.
- Si search_product con la frase completa no devuelve resultados y la consulta incluye marca + categoria generica (ej: "celulares samsung"), realiza una segunda busqueda con la marca sola (ej: "samsung").
- Si la segunda busqueda devuelve resultados, responde con esos resultados.

- No repitas la misma tool con el mismo input mas de una vez.
- No hagas bucles de busqueda ni cadenas de variantes; si tras 1 busqueda principal y 1 fallback no hay resultados, responde NO_DATA.

REGLAS PARA PREGUNTAS DE ATRIBUTO (MUY IMPORTANTE):
- Si preguntan por envio gratis, responde con free_shipping.
- Si preguntan cuanto tarda en llegar, responde con shipping_days.
- Si preguntan por garantia, responde con warranty_months.
- Si preguntan por devolucion, responde con return_days e is_final_sale.
- Si is_final_sale=true o return_days=0, indica que es Venta Final y no admite devolucion.
- Para estas preguntas cerradas de atributo, responde en 1-2 oraciones y NO agregues ficha tecnica completa.
- En preguntas cerradas, responde SOLO el atributo solicitado y no agregues otros atributos no pedidos.
- Si preguntan por garantia, NO menciones devolucion ni envio.
- Si preguntan por tiempo de entrega, NO menciones garantia ni devolucion.
- Si preguntan por envio gratis, NO menciones garantia ni devolucion.
- Si preguntan por tiempo de entrega, no menciones free_shipping ni promociones.
- Si la pregunta es solo de precio (ej: "cuanto cuesta ..."), responde solo precio.
- En preguntas de precio, no llames get_product_details salvo que el usuario pida explicitamente detalles adicionales.

RESTRICCIONES:
- No inventes datos ni uses conocimiento externo.
- No digas "no tengo informacion" si ya hay datos en la tool.
- Usa solo campos devueltos por tools.
- No recomiendes autenticacion en consultas publicas de catalogo.
- No devuelvas JSON incrustado, fichas largas ni listados de campos internos en el mensaje al usuario.

ESTILO DE RESPUESTA:
- Directo, util.
- Para consultas amplias sin match claro (ej: "productos de electronica" sin resultados), pide especificar tipo de producto (celulares, laptops, televisores) en lugar de decir "no tengo informacion".
- Para preguntas simples de precio/disponibilidad, no agregues especificaciones tecnicas no solicitadas.

- Usa texto plano unicamente; no uses simbolos decorativos como "✅", "❌", "•" ni caracteres especiales de estado.
- No uses tabulaciones en el mensaje; usa lineas simples con prefijo "- ".
- No cierres con contra-preguntas tipo "¿Te gustaria...?" cuando la consulta ya es clara.
- Si piden precio de un modelo y no existe exacto pero hay variantes cercanas, responde con alternativas directas y concisas, sin introducir preguntas adicionales.
- Si preguntan solo precio, responde solo precio (y, si aplica, alternativas cercanas), sin agregar envio, garantia ni promociones.

SALIDA OBLIGATORIA:
{
	"route": "ANSWER" | "NO_DATA" | "BLOCK",
	"message": "texto para el usuario"
}
"""

_VALID_ROUTES = {"ANSWER", "NO_DATA", "BLOCK"}
_DEFAULT_INVENTORY_MESSAGE = "No encontre datos suficientes para responder con precision."


def _extract_code_block(raw: str) -> str:
	raw = str(raw or "").strip()
	match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, flags=re.IGNORECASE | re.DOTALL)
	if match:
		return match.group(1).strip()
	if raw.startswith("```"):
		# Best effort if closing fence is missing.
		raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
		return raw.strip()
	return raw


def _normalize_query_text(value: str) -> str:
	text = str(value or "").strip().lower()
	text = text.translate(str.maketrans(
		"áéíóúüñ",
		"aeiouun",
	))
	text = re.sub(r"[^a-z0-9\s]", " ", text)
	text = re.sub(r"\s+", " ", text).strip()
	return text


def _format_currency(value: object) -> str | None:
	try:
		amount = float(value)
	except (TypeError, ValueError):
		return None
	formatted = f"{int(round(amount)):,}".replace(",", ".")
	return f"${formatted}"


def _normalize_model_name(name: object) -> str:
	return _normalize_query_text(str(name or ""))


def _dedupe_search_results(results: list[dict]) -> list[dict]:
	seen: set[str] = set()
	deduped: list[dict] = []
	for item in results:
		if not isinstance(item, dict):
			continue
		name_key = _normalize_model_name(item.get("name"))
		product_id = item.get("product_id")
		key = name_key or str(product_id)
		if not key or key in seen:
			continue
		seen.add(key)
		deduped.append(item)
	return deduped


def _is_price_query(query: str) -> bool:
	normalized = _normalize_query_text(query)
	return any(token in normalized for token in ["cuanto cuesta", "cuanto vale", "precio"])


def _is_stock_query(query: str) -> bool:
	normalized = _normalize_query_text(query)
	return any(token in normalized for token in ["stock", "disponible", "disponibilidad", "hay"])


def _is_free_shipping_query(query: str) -> bool:
	normalized = _normalize_query_text(query)
	return "envio gratis" in normalized or "envio es gratis" in normalized


def _is_shipping_days_query(query: str) -> bool:
	normalized = _normalize_query_text(query)
	return any(token in normalized for token in ["cuanto tarda", "cuanto demora", "llega", "tiempo de entrega"])


def _is_warranty_query(query: str) -> bool:
	return "garantia" in _normalize_query_text(query)


def _is_return_query(query: str) -> bool:
	normalized = _normalize_query_text(query)
	return "devol" in normalized or "devolver" in normalized


def _is_promotions_query(query: str) -> bool:
	normalized = _normalize_query_text(query)
	return any(token in normalized for token in ["promocion", "promociones", "descuento", "oferta"])


def _is_closed_attribute_query(query: str) -> bool:
	return any([
		_is_price_query(query),
		_is_stock_query(query),
		_is_free_shipping_query(query),
		_is_shipping_days_query(query),
		_is_warranty_query(query),
		_is_return_query(query),
		_is_promotions_query(query),
	])


def _find_latest_product_details(traces: list[dict]) -> dict | None:
	for trace in reversed(traces):
		tool_name = str(trace.get("tool_name", "")).strip()
		output_data = trace.get("output_data", {}) or {}
		if tool_name == "get_product_details" and isinstance(output_data, dict) and output_data.get("product_id") is not None:
			return output_data
	return None


def _strip_follow_up_question(text: str) -> str:
	cleaned = str(text or "").strip()
	cleaned = re.sub(r"\s*¿[^\n?]{1,240}\?\s*$", "", cleaned, flags=re.IGNORECASE)
	cleaned = re.sub(r"\n\s*¿[^\n?]{1,240}\?\s*$", "", cleaned, flags=re.IGNORECASE)
	return cleaned.strip()


def _recover_from_product_details(query: str, details: dict) -> dict | None:
	name = str(details.get("name") or f"producto {details.get('product_id')}").strip()
	product_id = details.get("product_id")
	price = _format_currency(details.get("price"))
	availability_units = details.get("availability_units")
	shipping_days = details.get("shipping_days")
	warranty_months = details.get("warranty_months")
	free_shipping = details.get("free_shipping")
	return_days = details.get("return_days")
	is_final_sale = details.get("is_final_sale")
	promotions = details.get("promotions") or []

	if _is_price_query(query) and price:
		return {"route": "ANSWER", "message": f"El producto **{name}** (producto {product_id}) cuesta **{price}**."}

	if _is_stock_query(query) and availability_units is not None:
		if details.get("is_available"):
			return {"route": "ANSWER", "message": f"Sí, el **{name}** (producto {product_id}) tiene **{availability_units} unidades disponibles**."}
		return {"route": "ANSWER", "message": f"No, el **{name}** (producto {product_id}) no tiene stock disponible en este momento."}

	if _is_free_shipping_query(query) and free_shipping is not None:
		answer = "Sí" if bool(free_shipping) else "No"
		suffix = "es gratis" if bool(free_shipping) else "no es gratis"
		return {"route": "ANSWER", "message": f"{answer}, el envío para el **{name}** (producto {product_id}) {suffix}."}

	if _is_shipping_days_query(query) and shipping_days is not None:
		return {"route": "ANSWER", "message": f"El producto **{name}** llega en **{shipping_days} días** desde la confirmación del pedido."}

	if _is_warranty_query(query) and warranty_months is not None:
		return {"route": "ANSWER", "message": f"El producto **{name}** (producto {product_id}) tiene una garantía de **{warranty_months} meses**."}

	if _is_return_query(query):
		if is_final_sale or return_days == 0:
			return {"route": "ANSWER", "message": f"El producto **{name}** es una **Venta Final** y no admite devolución según nuestra política."}
		if return_days is not None:
			return {"route": "ANSWER", "message": f"El producto **{name}** tiene un plazo de devolución de **{return_days} días** desde la entrega."}

	if _is_promotions_query(query):
		if promotions:
			promotion_lines = "\n".join(f"- {str(item).strip()}" for item in promotions if str(item).strip())
			return {"route": "ANSWER", "message": f"El producto **{name}** (producto {product_id}) tiene estas promociones activas:\n{promotion_lines}"}
		return {"route": "ANSWER", "message": f"El producto **{name}** (producto {product_id}) **no tiene promociones activas** en este momento."}

	return None


def _recover_from_search_results(query: str, results: list[dict]) -> dict | None:
	filtered = _dedupe_search_results(results)[:4]
	if not filtered:
		return None

	if _is_price_query(query):
		lines: list[str] = []
		for item in filtered:
			name = str(item.get("name") or f"producto {item.get('product_id')}").strip()
			price = _format_currency(item.get("price"))
			if not price:
				continue
			lines.append(f"- **{name}**: {price}")
		if lines:
			message = "Encontré estos modelos y precios disponibles:\n" + "\n".join(lines)
			return {"route": "ANSWER", "message": message}

	if _is_stock_query(query):
		lines = []
		for item in filtered:
			name = str(item.get("name") or f"producto {item.get('product_id')}").strip()
			availability_units = item.get("availability_units")
			if availability_units is None:
				continue
			lines.append(f"- **{name}**: {availability_units} unidades disponibles")
		if lines:
			return {"route": "ANSWER", "message": "Encontré estos productos disponibles:\n" + "\n".join(lines)}

	lines = []
	for item in filtered:
		name = str(item.get("name") or f"producto {item.get('product_id')}").strip()
		price = _format_currency(item.get("price"))
		availability_units = item.get("availability_units")
		parts = [f"**{name}**"]
		if price:
			parts.append(f"Precio: {price}")
		if availability_units is not None:
			parts.append(f"Disponible: {availability_units} unidades")
		lines.append("- " + ". ".join(parts) + ".")

	if lines:
		return {"route": "ANSWER", "message": "\n".join(lines)}

	return None


def _recover_inventory_result(query: str, raw: str, traces: list[dict], parsed_result: dict) -> dict:
	if parsed_result.get("message") != _DEFAULT_INVENTORY_MESSAGE:
		return parsed_result

	latest_product_details = None
	search_results: list[dict] = []
	for trace in traces:
		tool_name = str(trace.get("tool_name", "")).strip()
		output_data = trace.get("output_data", {}) or {}
		if tool_name == "get_product_details" and isinstance(output_data, dict) and output_data.get("product_id") is not None:
			latest_product_details = output_data
		elif tool_name == "search_product" and isinstance(output_data, dict):
			search_results.extend(output_data.get("results", []) or [])

	if latest_product_details is not None:
		recovered = _recover_from_product_details(query, latest_product_details)
		if recovered is not None:
			return recovered

	if search_results:
		recovered = _recover_from_search_results(query, search_results)
		if recovered is not None:
			return recovered

	cleaned_raw = _strip_follow_up_question(_extract_code_block(raw))
	if cleaned_raw and not cleaned_raw.startswith("{"):
		return {
			"route": "ANSWER",
			"message": cleaned_raw,
		}

	return parsed_result


def _parse_inventory_result(raw: str) -> dict:
	route = "NO_DATA"
	message = _DEFAULT_INVENTORY_MESSAGE

	def _extract_message_fallback(text: str) -> str | None:
		message_key = re.search(r'"message"\s*:\s*"', text, flags=re.IGNORECASE)
		if not message_key:
			return None

		index = message_key.end()
		captured: list[str] = []
		escaped = False
		while index < len(text):
			ch = text[index]
			if escaped:
				captured.append("\\" + ch)
				escaped = False
			elif ch == "\\":
				escaped = True
			elif ch == '"':
				break
			else:
				captured.append(ch)
			index += 1

		if not captured:
			return None

		candidate = "".join(captured)
		# Normalize common escaped sequences from LLM output.
		candidate = candidate.replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"')
		return candidate.strip() if candidate.strip() else None

	try:
		data = json.loads(raw)
		parsed_route = str(data.get("route", "")).strip().upper()
		if parsed_route in _VALID_ROUTES:
			route = parsed_route
		
		msg = data.get("message", message)
		
		
		if isinstance(msg, str):
			message = msg.strip()
		
		elif isinstance(msg, dict):
			message = json.dumps(msg, ensure_ascii=False, indent=2)
	
		elif isinstance(msg, list):
			message = "\n".join(str(item) for item in msg)
		# stringify
		else:
			message = str(msg).strip()
		
	except json.JSONDecodeError:
		route_match = re.search(r"\b(ANSWER|NO_DATA|BLOCK)\b", raw.upper())
		if route_match:
			route = route_match.group(1)

		message_from_raw = _extract_message_fallback(raw)
		if message_from_raw:
			message = message_from_raw
		

	return {
		"route": route,
		"message": message,
		
	}


_PUBLIC_TOOLS = [
	search_product,
	check_stock,
	get_product_details,
]

_PRIVATE_TOOLS = [
	get_customer_orders,
	get_order_details,
	get_order_shipping_status,
	get_item_return_eligibility,
	get_item_warranty,
	get_product_details,
]


_public_agent = None
_private_agent = None


def _create_public_agent():
	return Agent(
		model=model,
		system_prompt=public_system_prompt,
		tools=_PUBLIC_TOOLS,
		callback_handler=None,
	)


def _create_private_agent():
	return Agent(
		model=model,
		system_prompt=private_system_prompt,
		tools=_PRIVATE_TOOLS,
		callback_handler=None,
	)


def init_inventory_agents():
	global _public_agent, _private_agent
	_public_agent = _create_public_agent()
	_private_agent = _create_private_agent()


def reset_inventory_agents():
	init_inventory_agents()


register_reset_callback(reset_inventory_agents)

#init agents
init_inventory_agents()


def solve_inventory_query(input: str, query_type: str = "PUBLIC"):
	agent = _private_agent if query_type == "PRIVATE" else _public_agent
	if agent is None:
		init_inventory_agents()
		agent = _private_agent if query_type == "PRIVATE" else _public_agent
		if agent is None:
			return {
				"route": "NO_DATA",
				"message": "No pude inicializar el agente de inventario.",
				"response_data": None,
			}

	trace_start_idx = get_tool_trace_length()
	response = agent(input)
	raw = _extract_code_block(str(response).strip())
	result = _parse_inventory_result(raw)
	traces = get_tool_trace_since(trace_start_idx)
	latest_product_details = _find_latest_product_details(traces)
	if query_type == "PUBLIC" and latest_product_details is not None and _is_closed_attribute_query(input):
		tool_grounded = _recover_from_product_details(input, latest_product_details)
		if tool_grounded is not None:
			result = tool_grounded
	result = _recover_inventory_result(input, str(response).strip(), traces, result)
	if query_type == "PUBLIC" and _is_price_query(input):
		result["message"] = _strip_follow_up_question(result.get("message", ""))

	return {
		"route": result["route"],
		"message": result["message"],
		"response_data": response,
	}


