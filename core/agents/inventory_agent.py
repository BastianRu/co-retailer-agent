from strands.models.bedrock import BedrockModel
from strands import Agent
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
		max_tokens=1800,
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
Responde consultas públicas de catálogo.
Devuelve SIEMPRE JSON valido y SOLO JSON.

===TOOLS===
1. search_product(query): Busca productos. Devuelve lista con product_ids, name, price/price_min/price_max, availability_units, description.
2. check_stock(query): Si pides stock explícitamente.
3. get_product_details(product_id, product_ids): Detalles técnicos. SIEMPRE pasa product_id=product_ids[0] y product_ids=<lista completa de product_ids del search_product> para obtener stock consolidado.

===DECISIÓN PRINCIPAL===
Si la consulta menciona marca/producto (iphone, samsung, laptop, etc.) -> USA search_product.
Si NO hay marca/producto claro (ej. "cuanto cuesta ese?") -> Devuelve NO_DATA pidiendo aclaración.

===	EJEMPLOS ESPECÍFICOS ===
- iphone/apple -> search_product("iphone")
- samsung -> search_product("samsung")
- laptop -> search_product("laptop")
- Celulares genéricos (sin marca) -> search_product("iphone") Y search_product("samsung")
- Promociones genéricas -> search_product("iphone"), search_product("samsung"), search_product("laptop")

===REGLA CRÍTICA===
SI search_product DEVUELVE RESULTADOS (lista no vacía):
  -> Responde ANSWER con esos productos INMEDIATAMENTE.
  -> No esperes validación, no dudes, no busques confirmación adicional.
	-> Si obtuviste resultados de una tool, SIEMPRE USALOS.

SI search_product DEVUELVE LISTA VACÍA:
  -> Responde ANSWER diciendo "no está disponible".

===RESPUESTAS===
- Con search_product: Para cada producto -> nombre, precio (o rango si hay price_min/max), disponibilidad, descripción.
- Máx 5 productos por búsqueda.
- Si hay get_product_details: incluye specs, instalación, garantía, disponibilidad.
- Ambiguas (sin producto claro): pide aclaración en UNA sola frase.
- No puedes agregar características técnicas si no vienen en los datos de la herramienta.
- No uses conocimiento general del producto.

- Si el usuario pide "detalles", "información", "especificaciones":
  → SIEMPRE usar get_product_details

- Incluir:
  → descripción
  → especificaciones (si existen)
  → garantía
  → devolución
  → envío

	- Preguntas de recomendación NUNCA pueden devolver NO_DATA.

- Si hay productos disponibles:
  → SIEMPRE recomendar al menos uno.

- Si hay varios:
  → máximo 2 opciones.

===ESTILO===
- Factual, datos exactos de tools, sin suposiciones.
- Para discovery: breve y conciso.
- Para detalle técnico: completo y estructurado.

===SALIDA===
IMPORTANTE: Devuelve SIEMPRE un JSON con estructura exactamente así:
{
  "route": "ANSWER" o "NO_DATA",
  "message": "texto puro (STRING ÚNICAMENTE), nunca un objeto. Lista los productos con nombre, precio, disponibilidad. Una respuesta legible."
	
}

REGLA DE ORO para message:
- message SIEMPRE es un string de texto
- Nunca un objeto { }
- Nunca un array [ ]
- Si hay múltiples productos, escribe UNO POR UNO en LÍNEAS SEPARADAS dentro del string: "Producto 1: ...\nProducto 2: ..."
"""

_VALID_ROUTES = {"ANSWER", "NO_DATA", "BLOCK"}


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


def _parse_inventory_result(raw: str) -> dict:
	route = "NO_DATA"
	message = "No encontre datos suficientes para responder con precision."

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
		
		# Si message es un string, usalo directo
		if isinstance(msg, str):
			message = msg.strip()
		# Si es un objeto/dict, conviertealo a JSON string
		elif isinstance(msg, dict):
			message = json.dumps(msg, ensure_ascii=False, indent=2)
		# Si es una lista, convierteala a texto legible
		elif isinstance(msg, list):
			message = "\n".join(str(item) for item in msg)
		# Por defecto, stringify
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
]


def solve_inventory_query(input: str, query_type: str = "PUBLIC"):
	if query_type == "PRIVATE":
		system_prompt = private_system_prompt
		tools = _PRIVATE_TOOLS
	else:
		system_prompt = public_system_prompt
		tools = _PUBLIC_TOOLS

	inventory_agent = Agent(
		model=model,
		system_prompt=system_prompt,
		tools=tools,
		callback_handler=None,
	)

	response = inventory_agent(input)
	raw = _extract_code_block(str(response).strip())
	result = _parse_inventory_result(raw)

	return {
		"route": result["route"],
		"message": result["message"],
		"response_data": response,
	}


