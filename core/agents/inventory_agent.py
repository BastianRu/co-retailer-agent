from strands.models.bedrock import BedrockModel
from strands import Agent
from dotenv import load_dotenv
import json
import os
import re
import unicodedata

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

Tu trabajo es resolver consultas PUBLICAS de catalogo usando las tools disponibles.
Debes responder SIEMPRE con JSON valido y SOLO JSON.

========================
1) TOOLS PUBLICAS REALES
========================

1. search_product(query, product_id=None, filters=None, top_k=5)
	 - Es la tool principal para casi todas las consultas publicas.
	 - Devuelve lista de productos candidatos con campos como:
		 name, product_id, product_ids, description,
		 price o price_min/price_max,
		 availability_units, is_available, is_low_stock,
		 has_promotion, promotions.
	 - Ya consolida muchos duplicados probables.

2. check_stock(query, product_id=None, top_k=5)
	 - Solo para preguntas explicitamente enfocadas en stock/unidades
		 si search_product no fue suficiente.
	 - Normalmente search_product ya trae availability_units.

3. get_product_details(product_id)
	 - Solo para detalle tecnico profundo despues de identificar claramente un producto.
	 - Devuelve specifications, warranty_months, return_days,
		 shipping_days, requires_installation, promotion_summary, etc.

========================
2) REGLA MAESTRA
========================

Si la consulta menciona una marca, producto, familia o categoria reconocible,
debes llamar search_product primero antes de responder.

Nunca respondas desde suposiciones.
Nunca uses conocimiento externo del mundo para completar el catalogo.
Nunca inventes productos, precios, promociones o alternativas.

========================
3) MAPEO DE CONSULTAS A BUSQUEDA
========================

Usa estos mapeos antes de llamar tools:

- iphone, iphones, apple, celular apple -> query="iphone"
- samsung, celulares samsung -> query="samsung"
- hp, lenovo, dell -> query=<marca>
- laptop, laptops, computador, computadores, pc -> query="laptop"

Para consultas genericas de celulares sin marca:
- celular, celulares, telefono, telefonos, smartphone, smartphones, movil, moviles
	-> haz DOS busquedas: query="iphone" y query="samsung"
	-> combina ambos resultados
	-> no menciones que hiciste dos busquedas

Para promociones genericas sin producto claro:
- "que productos tienen descuento"
- "que productos tienen oferta"
- "hay promociones"
	-> haz estas busquedas: "iphone", "samsung", "laptop"
	-> recopila productos con has_promotion=true
	-> si ninguno tiene promocion, responde ANSWER diciendo que no hay promociones activas en este momento

Para precio bajo / barato / economico en celulares:
- usa la estrategia de celulares genericos: busca "iphone" y "samsung"
- si no hay opciones economicas, igual responde ANSWER con los disponibles y aclara que no encontraste opciones baratas

========================
4) FLUJO OBLIGATORIO DE DECISION
========================

Caso A: La consulta SI identifica producto, marca o categoria
1. Llama search_product con el query normalizado.
2. Si search_product devuelve resultados, responde ANSWER.
3. Si search_product devuelve [], responde ANSWER indicando que no esta disponible.
4. Solo muestra alternativas si las obtuviste explicitamente con otra llamada a search_product en ese mismo turno.

Caso B: La consulta NO identifica producto, marca o categoria
Usa NO_DATA solo para consultas ambiguas como:
- "cuanto cuesta ese producto?"
- "tienen unidades disponibles?"
- "este producto tiene oferta?"
- "que tal ese?"
- "cuanto vale?"
- "hay disponibles?"

En NO_DATA pide UNA sola aclaracion concreta.

========================
5) COMO INTERPRETAR search_product
========================

Si search_product devuelve resultados:
- Eso basta para responder ANSWER.
- Usa SOLO los productos devueltos por la tool.
- Copia el campo name tal como venga en la tool.
- Nunca renombres un producto.
- Si hay price_min y price_max, muestra rango de precios.
- Si solo hay price, muestra precio exacto.
- Usa availability_units para disponibilidad real.
- Si is_low_stock=true, puedes mencionar que el stock es bajo.
- Si has_promotion=false, di que no hay promociones activas.
- Si has_promotion=true, menciona solo promociones del campo promotions.

Si varios resultados tienen EXACTAMENTE el mismo name:
- tratalos como el mismo producto logico en la respuesta
- suma availability_units
- usa rango de precios si los precios difieren
- muestra el nombre una sola vez
- nunca inventes nombres como "Otro modelo", "variante", "version" o similares

Cuando la pregunta es por una marca o categoria amplia:
- lista hasta 5 productos unicos
- no uses NO_DATA si ya tienes resultados validos

========================
6) CUANDO USAR CADA TOOL
========================

search_product:
- discovery, catalogo, precio, disponibilidad, promociones, comparacion inicial, recomendaciones

check_stock:
- solo si el usuario pide explicitamente stock/unidades y search_product no alcanza
- no lo uses para precio, promociones o discovery general

get_product_details:
- solo despues de identificar claramente un producto con search_product
- usalo para preguntas de detalle tecnico como:
	specifications, garantia general del producto, return_days,
	shipping_days, requires_installation

Si search_product ya contiene suficiente informacion, no llames otra tool.

========================
7) REGLAS PARA RECOMENDACIONES Y COMPARACIONES
========================

Preguntas como:
- "que iphone es mejor?"
- "que celular recomiendas?"
- "que es mejor?"

NUNCA deben devolver NO_DATA si puedes buscar una categoria valida.

Reglas:
- Si solo hay un producto relevante, ese es el mejor disponible.
- Si el usuario pide una marca especifica, no introduzcas otras marcas salvo que el usuario pida alternativas o comparacion.
- Si el usuario no especifica marca y pregunta por celulares, puedes comparar iphone y samsung usando solo resultados reales.

========================
8) PROHIBICIONES DURAS
========================

- Nunca inventes alternativas no devueltas por tools.
- Nunca menciones modelos no presentes en resultados de tools.
- Nunca digas "iPhone 13", "iPhone 14", etc. si la tool no los devolvio.
- Nunca inventes promociones como "promocion de lanzamiento" si has_promotion es false.
- Nunca inventes etiquetas como "Otro modelo", "modelo similar", "version alternativa" o equivalentes.
- Nunca respondas "No encontre datos suficientes..." si la consulta identifica una marca/categoria y search_product podia haberse usado.
- Nunca menciones duplicados, variantes, consolidacion ni problemas internos del catalogo.
- Nunca agregues recomendaciones no pedidas fuera de preguntas de recomendacion/comparacion.
- Nunca uses frases de urgencia o persuasion como:
	"te recomiendo comprar pronto"
	"te sugerimos actuar rapido"
	"aprovecha"
	"revisa nuevamente pronto"
	salvo que el usuario pida una recomendacion.

========================
9) ESTILO
========================

- Respuesta breve, clara y factual.
- Usa datos exactos devueltos por tools.
- Para listas: maximo 5 productos unicos.
- Para no disponible: una frase directa es suficiente.
- Para NO_DATA: pide una sola aclaracion concreta.
- En preguntas informativas normales, responde con hechos, no con persuasion.

========================
10) PLANTILLAS RECOMENDADAS
========================

Usa estas plantillas como referencia fuerte:

A. Producto disponible unico
"Actualmente tenemos disponible el <name>. Precio: <price o rango>. Disponibilidad: <availability_units>."

B. Varios productos disponibles
"Actualmente tenemos disponibles:
1. <name exacto de tool> - Precio: <price o rango>. Disponibilidad: <availability_units>.
2. ..."

C. Producto no disponible
"El producto <consulta del usuario> no esta disponible actualmente en nuestro catalogo."

D. Sin promociones
"Actualmente no hay promociones activas en <producto o categoria>."

E. NO_DATA por ambiguedad
"Indica el nombre, marca o categoria del producto que quieres consultar."

========================
11) VOCABULARIO PROHIBIDO
========================

No uses estas palabras o frases salvo que el usuario pida explicitamente una recomendacion:
- recomiendo
- recomendamos
- sugerimos
- actuar pronto
- aprovecha
- revisa nuevamente pronto
- si te interesa

Nunca uses estas etiquetas inventadas:
- otro modelo
- modelo similar
- variante
- version alternativa
- opcion parecida

========================
12) SALIDA OBLIGATORIA
========================

Devuelve SIEMPRE JSON valido y SOLO JSON.
Sin markdown fuera del campo message.
Sin bloques de codigo.

Formato:
{
	"route": "ANSWER" | "NO_DATA",
	"message": "respuesta para el usuario"
}

Ejemplos de decision:
- "que iphones tienen?" -> search_product("iphone") -> ANSWER
- "que laptops tienen disponibles?" -> search_product("laptop") -> ANSWER
- "cuanto cuesta ese producto?" -> NO_DATA
- "que productos tienen descuento?" -> buscar iphone + samsung + laptop -> ANSWER
"""

_VALID_ROUTES = {"ANSWER", "NO_DATA", "BLOCK"}

_PUBLIC_MODEL_TOKENS = {"pro", "max", "plus", "ultra", "mini", "air"}
_PUBLIC_PHONE_TERMS = {
	"celular",
	"celulares",
	"telefono",
	"telefonos",
	"smartphone",
	"smartphones",
	"movil",
	"moviles",
}
_PUBLIC_LAPTOP_TERMS = {"laptop", "laptops", "computador", "computadores", "pc"}
_PUBLIC_PROMO_TERMS = {"promocion", "promociones", "descuento", "descuentos", "oferta", "ofertas"}
_PUBLIC_DETAIL_TERMS = {"detalle", "detalles", "incluye", "especificaciones", "garantia", "instalacion"}
_PUBLIC_RECOMMENDATION_TERMS = {"recomiendas", "recomendar", "mejor", "comparar", "comparacion"}
_PUBLIC_AMBIGUOUS_TERMS = {"ese", "esa", "este", "esta"}
_PUBLIC_SEARCH_STOPWORDS = {
	"algo",
	"barato",
	"baratos",
	"busca",
	"buscar",
	"catalogo",
	"con",
	"cuanto",
	"cuesta",
	"cuestan",
	"cual",
	"cuales",
	"dame",
	"de",
	"del",
	"disponible",
	"disponibles",
	"economico",
	"economicos",
	"el",
	"en",
	"esa",
	"ese",
	"esta",
	"este",
	"hay",
	"incluye",
	"la",
	"las",
	"lo",
	"los",
	"marca",
	"me",
	"mi",
	"modelo",
	"nombre",
	"no",
	"precio",
	"producto",
	"productos",
	"que",
	"quiero",
	"requiere",
	"stock",
	"su",
	"tal",
	"tiene",
	"tienen",
	"un",
	"una",
	"unidades",
	"vale",
	"ver",
	"exista",
	"detalles",
	"detalle",
	"oferta",
	"ofertas",
	"promocion",
	"promociones",
	"descuento",
	"descuentos",
}


def _normalize_public_text(value: str) -> str:
	text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode("ascii")
	text = re.sub(r"[^a-zA-Z0-9\s]", " ", text).lower()
	return re.sub(r"\s+", " ", text).strip()


def _tokenize_public_text(value: str) -> list[str]:
	return [token for token in _normalize_public_text(value).split() if token]


def _format_currency(value: object) -> str | None:
	try:
		amount = float(str(value))
	except (TypeError, ValueError):
		return None
	return f"${amount:,.0f}".replace(",", ".")


def _format_price_text(product: dict) -> str | None:
	price_min = product.get("price_min")
	price_max = product.get("price_max")
	if price_min is not None and price_max is not None:
		min_text = _format_currency(price_min)
		max_text = _format_currency(price_max)
		if min_text and max_text:
			if float(price_min) == float(price_max):
				return min_text
			return f"entre {min_text} y {max_text}"

	price = _format_currency(product.get("price"))
	return price


def _contains_any_token(tokens: set[str], expected: set[str]) -> bool:
	return bool(tokens & expected)


def _is_public_ambiguous_query(query_norm: str) -> bool:
	tokens = set(_tokenize_public_text(query_norm))
	if not tokens:
		return True
	if _contains_any_token(tokens, _PUBLIC_AMBIGUOUS_TERMS) and not any(token.isdigit() for token in tokens):
		if not _contains_any_token(tokens, _PUBLIC_PHONE_TERMS | _PUBLIC_LAPTOP_TERMS | {"iphone", "apple", "samsung", "hp", "lenovo", "dell"}):
			return True
	return False


def _is_public_recommendation_query(query_norm: str) -> bool:
	tokens = set(_tokenize_public_text(query_norm))
	return _contains_any_token(tokens, _PUBLIC_RECOMMENDATION_TERMS)


def _is_public_detail_query(query_norm: str) -> bool:
	tokens = set(_tokenize_public_text(query_norm))
	return _contains_any_token(tokens, _PUBLIC_DETAIL_TERMS)


def _is_generic_promotion_query(query_norm: str) -> bool:
	tokens = set(_tokenize_public_text(query_norm))
	if not _contains_any_token(tokens, _PUBLIC_PROMO_TERMS):
		return False
	return not _contains_any_token(
		tokens,
		_PUBLIC_PHONE_TERMS | _PUBLIC_LAPTOP_TERMS | {"iphone", "iphones", "apple", "samsung", "hp", "lenovo", "dell"},
	)


def _is_generic_phone_query(query_norm: str) -> bool:
	tokens = set(_tokenize_public_text(query_norm))
	if not _contains_any_token(tokens, _PUBLIC_PHONE_TERMS):
		return False
	return not _contains_any_token(tokens, {"iphone", "apple", "samsung"})


def _dedupe_public_results(results: list[dict]) -> list[dict]:
	by_name: dict[str, dict] = {}
	ordered: list[str] = []

	for item in results:
		name_key = _normalize_public_text(item.get("name", ""))
		if not name_key:
			continue

		existing = by_name.get(name_key)
		if existing is None:
			by_name[name_key] = dict(item)
			ordered.append(name_key)
			continue

		existing_ids = set(existing.get("product_ids") or [])
		incoming_ids = set(item.get("product_ids") or [])
		existing["product_ids"] = sorted(existing_ids | incoming_ids)
		existing["availability_units"] = max(int(existing.get("availability_units", 0)), int(item.get("availability_units", 0)))
		existing["is_available"] = bool(existing.get("is_available")) or bool(item.get("is_available"))
		existing["is_low_stock"] = bool(existing.get("is_low_stock")) or bool(item.get("is_low_stock"))

		existing_prices = [value for value in [existing.get("price"), existing.get("price_min"), existing.get("price_max")] if value is not None]
		incoming_prices = [value for value in [item.get("price"), item.get("price_min"), item.get("price_max")] if value is not None]
		all_prices = [float(value) for value in existing_prices + incoming_prices]
		if all_prices:
			existing["price_min"] = min(all_prices)
			existing["price_max"] = max(all_prices)
			existing.pop("price", None)

	return [by_name[name_key] for name_key in ordered]


def _derive_public_search_queries(user_input: str) -> list[str]:
	query_norm = _normalize_public_text(user_input)
	tokens = _tokenize_public_text(query_norm)
	token_set = set(tokens)
	has_specific_model = any(token.isdigit() for token in tokens) or _contains_any_token(token_set, _PUBLIC_MODEL_TOKENS)

	if _is_generic_promotion_query(query_norm):
		return ["iphone", "samsung", "laptop"]

	if _is_generic_phone_query(query_norm):
		return ["iphone", "samsung"]

	if _contains_any_token(token_set, _PUBLIC_LAPTOP_TERMS):
		return ["laptop"]

	if "iphone" in token_set or "apple" in token_set:
		if has_specific_model:
			filtered = ["iphone" if token == "apple" else token for token in tokens if token not in _PUBLIC_SEARCH_STOPWORDS]
			if filtered:
				return [" ".join(filtered)]
		return ["iphone"]

	if "samsung" in token_set:
		filtered = [token for token in tokens if token not in _PUBLIC_SEARCH_STOPWORDS]
		if has_specific_model and filtered:
			return [" ".join(filtered)]
		return ["samsung"]

	for brand in ("hp", "lenovo", "dell"):
		if brand in token_set:
			return [brand]

	filtered = [token for token in tokens if token not in _PUBLIC_SEARCH_STOPWORDS]
	if filtered:
		return [" ".join(filtered[:6])]

	return []


def _run_public_search_queries(queries: list[str]) -> list[dict]:
	results: list[dict] = []
	for query in queries:
		query_text = str(query or "").strip()
		if not query_text:
			continue
		results.extend(search_product(query_text))
	return _dedupe_public_results(results)


def _build_public_unavailable_message(user_input: str) -> str:
	queries = _derive_public_search_queries(user_input)
	subject = queries[0] if queries else "el producto consultado"
	subject = re.sub(r"\biphone\b", "iPhone", subject, flags=re.IGNORECASE)
	subject = re.sub(r"\bsamsung\b", "Samsung", subject, flags=re.IGNORECASE)
	return f"El producto {subject} no esta disponible actualmente en nuestro catalogo."


def _build_public_listing_message(user_input: str, products: list[dict]) -> str:
	visible_products = [product for product in products if product.get("is_available", True)][:5]
	if not visible_products:
		visible_products = products[:5]

	query_norm = _normalize_public_text(user_input)
	if len(visible_products) == 1:
		product = visible_products[0]
		message = f"Actualmente tenemos disponible el {product['name']}."
		price_text = _format_price_text(product)
		if price_text:
			message += f" Precio: {price_text}."
		availability_units = product.get("availability_units")
		if availability_units is not None:
			message += f" Disponibilidad: {availability_units} unidades."
		return message

	if _contains_any_token(set(_tokenize_public_text(query_norm)), _PUBLIC_PHONE_TERMS) and _contains_any_token(set(_tokenize_public_text(query_norm)), {"barato", "baratos", "economico", "economicos"}):
		intro = "Actualmente no encontramos celulares baratos en el catalogo consultado. Estas son las opciones disponibles:"
	elif "samsung" in query_norm:
		intro = "Actualmente tenemos disponibles los siguientes productos de Samsung:"
	elif "iphone" in query_norm or "apple" in query_norm:
		intro = "Actualmente tenemos disponibles los siguientes productos Apple:"
	elif _contains_any_token(set(_tokenize_public_text(query_norm)), _PUBLIC_LAPTOP_TERMS):
		intro = "Actualmente tenemos disponibles las siguientes laptops:"
	else:
		intro = "Actualmente tenemos disponibles:"

	lines = []
	for index, product in enumerate(visible_products, start=1):
		line = f"{index}. {product['name']}"
		price_text = _format_price_text(product)
		if price_text:
			line += f" - Precio: {price_text}"
		availability_units = product.get("availability_units")
		if availability_units is not None:
			line += f". Disponibilidad: {availability_units} unidades"
		if product.get("is_low_stock"):
			line += " (stock bajo)"
		line += "."
		lines.append(line)

	return intro + "\n\n" + "\n".join(lines)


def _build_public_detail_message(user_input: str) -> dict | None:
	queries = _derive_public_search_queries(user_input)
	if not queries:
		return None

	products = _run_public_search_queries(queries)
	if not products:
		return {
			"route": "ANSWER",
			"message": _build_public_unavailable_message(user_input),
		}

	product = products[0]
	product_id = product.get("product_id")
	if product_id is None:
		return None

	details = get_product_details(int(product_id))
	name = product.get("name", "Producto")
	query_norm = _normalize_public_text(user_input)

	if "garantia" in query_norm:
		warranty_months = details.get("warranty_months")
		if warranty_months is None:
			message = f"No tengo informacion de garantia disponible para el {name}."
		else:
			message = f"El {name} tiene una garantia de {warranty_months} meses."
		return {"route": "ANSWER", "message": message}

	if "instalacion" in query_norm:
		requires_installation = bool(details.get("requires_installation"))
		message = f"El {name} {'requiere' if requires_installation else 'no requiere'} instalacion."
		installation_notes = details.get("installation_notes")
		if installation_notes and str(installation_notes) != "nan":
			message += f" {installation_notes}."
		return {"route": "ANSWER", "message": message}

	parts = [f"Detalles del {name}: {details.get('description') or product.get('description')}."]
	if details.get("specifications"):
		parts.append(f"Especificaciones: {details['specifications']}.")
	if details.get("warranty_months") is not None:
		parts.append(f"Garantia: {details['warranty_months']} meses.")
	if details.get("return_days") is not None:
		parts.append(f"Devolucion: {details['return_days']} dias.")
	if details.get("shipping_days") is not None:
		parts.append(f"Envio estimado: {details['shipping_days']} dias.")
	parts.append(f"Requiere instalacion: {'si' if details.get('requires_installation') else 'no'}.")
	price_text = _format_price_text(product)
	if price_text:
		parts.append(f"Precio: {price_text}.")
	availability_units = product.get("availability_units")
	if availability_units is not None:
		parts.append(f"Disponibilidad: {availability_units} unidades.")

	return {
		"route": "ANSWER",
		"message": " ".join(parts),
	}


def _sanitize_public_message(user_input: str, message: str) -> str:
	query_norm = _normalize_public_text(user_input)
	cleaned = str(message or "").replace("**", "").replace("\\$", "$" ).strip()
	cleaned = re.sub(r"\((?:Otro modelo|otro modelo|Modelo similar|modelo similar|Version alternativa|version alternativa|Opcion parecida|opcion parecida)\)", "", cleaned)

	if not _is_public_recommendation_query(query_norm):
		cleaned = re.sub(r"Si buscas opciones[^.]*\.", "", cleaned, flags=re.IGNORECASE)
		cleaned = re.sub(r"Si buscas opciones[^\n]*$", "", cleaned, flags=re.IGNORECASE)
		cleaned = re.sub(r"Te recomiendo[^.]*\.", "", cleaned, flags=re.IGNORECASE)
		cleaned = re.sub(r"te recomiendo[^.]*\.", "", cleaned, flags=re.IGNORECASE)
		cleaned = re.sub(r"revisa nuevamente pronto[^.]*\.", "", cleaned, flags=re.IGNORECASE)
		cleaned = re.sub(r"si te interesa[^.]*\.", "", cleaned, flags=re.IGNORECASE)
		if re.search(r"no (?:tenemos|esta) disponible", _normalize_public_text(cleaned)):
			cleaned = re.split(r"\bSin embargo\b", cleaned, maxsplit=1, flags=re.IGNORECASE)[0].strip()

	message_norm = _normalize_public_text(cleaned)
	if _is_generic_promotion_query(query_norm) and (
		"no hay promociones activas" in message_norm
		or "no hay productos con promociones" in message_norm
		or "no hay productos con descuentos" in message_norm
		or "no hay descuentos" in message_norm
	):
		cleaned = "Actualmente no hay promociones activas en el catalogo consultado."

	cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
	cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
	cleaned = re.sub(r" +", " ", cleaned)
	cleaned = cleaned.strip(" .")
	if cleaned and cleaned[-1] not in ".!?":
		cleaned += "."
	return cleaned


def _apply_public_guardrails(user_input: str, result: dict) -> dict:
	guardrailed = dict(result)
	query_norm = _normalize_public_text(user_input)

	if guardrailed.get("route") == "NO_DATA" and not _is_public_ambiguous_query(query_norm):
		queries = _derive_public_search_queries(user_input)
		if queries:
			products = _run_public_search_queries(queries)
			if products:
				guardrailed = {
					"route": "ANSWER",
					"message": _build_public_listing_message(user_input, products),
				}
			else:
				guardrailed = {
					"route": "ANSWER",
					"message": _build_public_unavailable_message(user_input),
				}

	if guardrailed.get("route") == "ANSWER" and _is_public_detail_query(query_norm):
		message_norm = _normalize_public_text(guardrailed.get("message", ""))
		if not any(term in message_norm for term in ["especificaciones", "garantia", "instalacion", "detalles del"]):
			detail_result = _build_public_detail_message(user_input)
			if detail_result is not None:
				guardrailed = detail_result

	guardrailed["message"] = _sanitize_public_message(user_input, guardrailed.get("message", ""))
	return guardrailed


def _extract_code_block(raw: str) -> str:
	if raw.startswith("```"):
		raw = raw.strip("`")
		raw = raw.replace("json", "", 1).strip()
	return raw


def _parse_inventory_result(raw: str) -> dict:
	route = "NO_DATA"
	message = "No encontre datos suficientes para responder con precision."

	try:
		data = json.loads(raw)
		parsed_route = str(data.get("route", "")).strip().upper()
		if parsed_route in _VALID_ROUTES:
			route = parsed_route
		message = str(data.get("message", message)).strip()
		
	except json.JSONDecodeError:
		route_match = re.search(r"\b(ANSWER|NO_DATA|BLOCK)\b", raw.upper())
		if route_match:
			route = route_match.group(1)

		message_match = re.search(r'"message"\s*:\s*"([^"]+)"', raw, flags=re.IGNORECASE)
		

		if message_match:
			message = message_match.group(1).strip()
		

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
	if query_type != "PRIVATE":
		result = _apply_public_guardrails(input, result)

	return {
		"route": result["route"],
		"message": result["message"],
		"response_data": response,
	}


