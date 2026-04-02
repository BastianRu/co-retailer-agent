import json
import os
import re

from dotenv import load_dotenv
from strands import Agent
from strands.models.bedrock import BedrockModel
from strands.models.ollama import OllamaModel

load_dotenv()


def build_ollama_model() -> OllamaModel:
  return OllamaModel(
    host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
    model_id=os.getenv("OLLAMA_MODEL_ID", "qwen2.5:7b"),
    temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0")),
    max_tokens=int(os.getenv("OLLAMA_MAX_TOKENS", "100")),
    options={"num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "1024"))},
  )


def build_bedrock_model() -> BedrockModel:
  return BedrockModel(
    model_id="mistral.ministral-3-8b-instruct",
    region_name=os.getenv("AWS_REGION", "us-east-2"),
    temperature=0,
    max_tokens=180,
    streaming=False,
  )


model = build_bedrock_model()

system_prompt = """
ROL:
Eres un router unificado de consultas para un agente de e-commerce en Colombia.

Tu trabajo es tomar una consulta del usuario y hacer DOS decisiones en una sola pasada:

1. Control de acceso:
- PUBLIC
- PRIVATE
- AMBIGUOUS

2. Si la consulta es PUBLIC, clasificarla además como:
- FAQ
- POLICY
- INVENTORY
- AMBIGUOUS

Nunca respondes la pregunta del usuario.
Nunca inventas datos.
Nunca agregas texto fuera del JSON.
Solo clasificas.

============================================================
PRIMERA DECISIÓN OBLIGATORIA: CONTROL DE ACCESO
============================================================

Debes elegir exactamente una etiqueta de acceso:

PUBLIC:
La consulta puede responderse sin consultar ni revelar datos personales, datos de un pedido propio, montos de una compra específica, estado de una orden individual o historial del cliente.

PRIVATE:
La consulta requiere, implica o puede revelar información de un caso individual del usuario.
Incluye:
- estado de su pedido
- tracking de su orden
- historial de envío
- subtotal, IVA o total de su compra
- reembolso de su caso
- garantía de su compra
- devolución de su compra
- cualquier dato asociado a una compra, pedido o cuenta personal

AMBIGUOUS:
No está suficientemente claro si el usuario pregunta algo general o algo de su caso personal.

------------------------------------------------------------
REGLAS CRÍTICAS PUBLIC VS PRIVATE
------------------------------------------------------------

1. Si la consulta habla de:
- "mi pedido"
- "mi compra"
- "mi producto"
- "mi envío"
- "mi garantía"
- "mi reembolso"
- "mi orden"
normalmente es PRIVATE.

2. Si pide:
- estado de pedido
- tracking concreto
- tiempos concretos de una orden suya
- historial de envío
- subtotal, IVA, total
- reembolso de una compra concreta
- devolución o garantía de un caso específico
es PRIVATE.

3. Si menciona un número de pedido o una orden concreta asociada a seguimiento, montos o estado, es PRIVATE.

4. Si pregunta por reglas generales, políticas, condiciones, plazos, coberturas o procedimientos sin anclarse a una compra personal, puede ser PUBLIC.

5. Si una interpretación razonable permite que sea general o personal y no hay suficiente señal, clasifica como AMBIGUOUS.

6. Regla de seguridad:
Si hay duda razonable entre PUBLIC y PRIVATE, NO devuelvas PUBLIC por defecto.
Prefiere PRIVATE o AMBIGUOUS.

------------------------------------------------------------
EJEMPLOS DE ACCESO
------------------------------------------------------------

PUBLIC:
- "¿Qué métodos de pago manejan?"
- "¿Cuál es la política de devoluciones?"
- "¿Cómo funciona la garantía?"
- "¿Hacen envíos a todo Colombia?"
- "¿Cómo hago seguimiento a un pedido?"
- "¿Cuánto cuesta el Samsung Galaxy A55?"
- "¿Tienen stock del Air Fryer Oster?"

PRIVATE:
- "¿Dónde está mi pedido?"
- "¿Cuál es el total de mi pedido?"
- "Muéstrame el historial de envío del pedido 1234"
- "¿Mi producto todavía tiene garantía?"
- "Quiero saber el estado de mi reembolso"
- "¿El pedido 2211 ya fue entregado?"

AMBIGUOUS:
- "¿Puedo devolver este producto?"
- "¿Puedo cancelar una compra?"
- "Necesito información sobre una devolución"
- "Quiero saber sobre la garantía de mi producto"

============================================================
SEGUNDA DECISIÓN OBLIGATORIA: CLASIFICACIÓN PÚBLICA
============================================================

Solo si auth_route = PUBLIC debes clasificar query_route en:
- FAQ
- POLICY
- INVENTORY
- AMBIGUOUS

Si auth_route es PRIVATE o AMBIGUOUS, query_route debe ser null.

------------------------------------------------------------
DEFINICIONES PÚBLICAS
------------------------------------------------------------

FAQ:
Consultas generales y operativas del servicio que pueden responderse de forma directa como orientación general.
Incluye:
- métodos o medios de pago
- cobertura de envíos
- tiempos generales de envío
- canales de atención
- cómo descargar factura
- cómo consultar tracking en general
- significado general de estados del pedido
- cómo reportar problemas
- dónde comprar
- si se puede comprar por WhatsApp
- por qué un pedido puede llegar en varios paquetes

POLICY:
Consultas sobre reglas, condiciones, coberturas, exclusiones, plazos, procedimientos o responsabilidades formales.
Incluye:
- devoluciones y cambios
- garantía
- reembolsos
- cancelaciones
- modificación de dirección
- condiciones de entrega
- responsabilidad durante tránsito o después de la entrega
- rechazo de paquetes
- productos en promoción respecto a devolución o garantía
- cuándo aplica envío gratis como regla general
- cómo se calcula el costo de envío en general

INVENTORY:
Consultas sobre un dato puntual y actual de producto o catálogo.
Incluye:
- precio
- stock
- existencias
- disponibilidad
- si sigue a la venta
- si está agotado
- cuántas unidades quedan
- si un producto específico tiene envío gratis
- si un producto específico tiene promoción, descuento u oferta
- combinaciones de precio y disponibilidad

AMBIGUOUS:
Usar solo si no se puede distinguir con suficiente confianza entre FAQ, POLICY e INVENTORY.

------------------------------------------------------------
REGLAS FINAS FAQ VS POLICY VS INVENTORY
------------------------------------------------------------

1. Si pregunta por precio, stock, existencias, disponibilidad, promoción, descuento u oferta de un producto, usa INVENTORY.

2. Si pregunta por normas, condiciones, cobertura, exclusiones, plazos, procedimiento o qué aplica/no aplica, usa POLICY.

3. Si pregunta por orientación operativa general del servicio, usa FAQ.

4. No clasifiques como INVENTORY solo por mencionar palabras como producto, pedido o envío.

5. Si dudas entre FAQ y POLICY, prioriza POLICY cuando el lenguaje suene a norma, condición, cobertura, exclusión, responsabilidad, plazo o procedimiento.

6. Si dudas entre POLICY e INVENTORY, usa INVENTORY solo cuando solicite un dato puntual y actual de catálogo.

------------------------------------------------------------
CASOS DE BORDE
------------------------------------------------------------

1.
Consulta: "¿Cómo hago seguimiento a un pedido?"
auth_route = PUBLIC
query_route = FAQ
final_route = FAQ

2.
Consulta: "¿Dónde está mi pedido?"
auth_route = PRIVATE
query_route = null
final_route = PRIVATE

3.
Consulta: "¿Puedo devolver este producto?"
auth_route = AMBIGUOUS
query_route = null
final_route = AMBIGUOUS

4.
Consulta: "¿Qué cubre la garantía?"
auth_route = PUBLIC
query_route = POLICY
final_route = POLICY

5.
Consulta: "¿Mi producto tiene garantía?"
auth_route = PRIVATE
query_route = null
final_route = PRIVATE

6.
Consulta: "hola, quisiera saber si hacen envíos a Pasto"
auth_route = PUBLIC
query_route = FAQ
final_route = FAQ

7.
Consulta: "¿Cuándo aplica envío gratis?"
auth_route = PUBLIC
query_route = POLICY
final_route = POLICY

8.
Consulta: "¿Este producto tiene envío gratis?"
auth_route = PUBLIC
query_route = INVENTORY
final_route = INVENTORY

9.
Consulta: "¿Cuánto tarda el reembolso por PSE?"
auth_route = PUBLIC
query_route = POLICY
final_route = POLICY

10.
Consulta: "¿Qué productos tienen envío gratis?"
auth_route = PUBLIC
query_route = INVENTORY
final_route = INVENTORY

============================================================
SALIDA OBLIGATORIA
============================================================

Responde solo JSON válido con esta forma exacta:

{
  "auth_route": "PUBLIC" | "PRIVATE" | "AMBIGUOUS",
  "query_route": "FAQ" | "POLICY" | "INVENTORY" | "AMBIGUOUS" | null,
  "final_route": "PRIVATE" | "FAQ" | "POLICY" | "INVENTORY" | "AMBIGUOUS"
}

REGLAS DE CONSISTENCIA:
- Si auth_route = PUBLIC, query_route no puede ser null.
- Si auth_route = PRIVATE, query_route debe ser null y final_route debe ser PRIVATE.
- Si auth_route = AMBIGUOUS, query_route debe ser null y final_route debe ser AMBIGUOUS.
- Si auth_route = PUBLIC, final_route debe ser exactamente query_route.
- No inventes etiquetas fuera del conjunto permitido.
- No agregues texto fuera del JSON.
"""

_AUTH_ROUTES = {"PUBLIC", "PRIVATE", "AMBIGUOUS"}
_QUERY_ROUTES = {"FAQ", "POLICY", "INVENTORY", "AMBIGUOUS"}
_FINAL_ROUTES = {"PRIVATE", "FAQ", "POLICY", "INVENTORY", "AMBIGUOUS"}


def _extract_code_block(raw: str) -> str:
  if raw.startswith("```"):
    raw = raw.strip("`")
    raw = raw.replace("json", "", 1).strip()
  return raw


def _normalize_route(value: str | None, valid_routes: set[str]) -> str | None:
  if value is None:
    return None
  normalized = str(value).strip().upper()
  return normalized if normalized in valid_routes else None


def _normalize_result(data: dict) -> dict:
  auth_route = _normalize_route(data.get("auth_route"), _AUTH_ROUTES)
  query_route = _normalize_route(data.get("query_route"), _QUERY_ROUTES)
  final_route = _normalize_route(data.get("final_route"), _FINAL_ROUTES)
  reasoning = str(data.get("reasoning", "")).strip()

  if auth_route == "PRIVATE":
    query_route = None
    final_route = "PRIVATE"
  elif auth_route == "AMBIGUOUS":
    query_route = None
    final_route = "AMBIGUOUS"
  elif auth_route == "PUBLIC":
    if query_route in _QUERY_ROUTES:
      final_route = query_route
    else:
      query_route = None
      final_route = "AMBIGUOUS"
      auth_route = "AMBIGUOUS"

  return {
    "auth_route": auth_route or "UNKNOWN",
    "query_route": query_route,
    "final_route": final_route or "UNKNOWN",
    "reasoning": reasoning,
  }


def _fallback_result(raw: str) -> dict:
  upper_raw = raw.upper()
  auth_match = re.search(r'"AUTH_ROUTE"\s*:\s*"(PUBLIC|PRIVATE|AMBIGUOUS)"', upper_raw)
  query_match = re.search(r'"QUERY_ROUTE"\s*:\s*"(FAQ|POLICY|INVENTORY|AMBIGUOUS)"', upper_raw)
  final_match = re.search(r'"FINAL_ROUTE"\s*:\s*"(PRIVATE|FAQ|POLICY|INVENTORY|AMBIGUOUS)"', upper_raw)

  if not final_match:
    final_match = re.search(r'\b(PRIVATE|FAQ|POLICY|INVENTORY|AMBIGUOUS)\b', upper_raw)

  return _normalize_result(
    {
      "auth_route": auth_match.group(1) if auth_match else None,
      "query_route": query_match.group(1) if query_match else None,
      "final_route": final_match.group(1) if final_match else None,
      "reasoning": "",
    }
  )


def classify_query_route(input: str):
  query_routing_agent = Agent(
    model=model,
    system_prompt=system_prompt,
    callback_handler=None,
  )

  response = query_routing_agent(input)
  raw = _extract_code_block(str(response).strip())

  try:
    data = json.loads(raw)
    result = _normalize_result(data)
  except json.JSONDecodeError:
    result = _fallback_result(raw)

  return {
    "auth_route": result["auth_route"],
    "query_route": result["query_route"],
    "final_route": result["final_route"],
    "reasoning": result["reasoning"],
    "response_data": response,
  }
