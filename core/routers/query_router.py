import json
import os
import re

from dotenv import load_dotenv
from strands import Agent
from strands.models.bedrock import BedrockModel
from strands.models.ollama import OllamaModel

load_dotenv()


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
Eres un router de consultas para e-commerce en Colombia.
Solo clasificas, no respondes contenido.

Debes devolver:
1) auth_route: PUBLIC | PRIVATE | AMBIGUOUS
2) query_route (si PUBLIC): FAQ | POLICY | INVENTORY | AMBIGUOUS
3) final_route

PRIORIDADES CRITICAS:
1) Si hay product_id explicito (ej: "producto 5001", "sku 5001") y preguntan por precio/stock/envio/tiempo/garantia/devolucion/promociones -> PUBLIC + INVENTORY.
2) Preguntas generales de politica con lenguaje condicional/hipotetico sobre pedidos NO son PRIVATE.
   Ejemplos PUBLIC + POLICY:
   - "¿Puedo cambiar la direccion si el pedido ya fue despachado?"
   - "¿Puedo cancelar mi pedido si ya fue despachado?"
   - "¿Que pasa si rechazo un pedido en la entrega?"
3) "¿Como hago seguimiento a un pedido?" es PUBLIC + FAQ (seguimiento general).

PRIVATE solo cuando la consulta pide datos de una compra/cuenta del usuario.
Ejemplos PRIVATE:
- "¿Cual es el estado de mi pedido?"
- "¿Cual es el total de mi pedido 9999?"
- "¿Cual es la direccion de entrega de mi pedido?"

PUBLIC clasificacion:
- FAQ: metodos de pago, cobertura, canales de compra, seguimiento general.
- POLICY: reglas generales de garantia/devoluciones/reembolsos/cancelaciones/condiciones.
- INVENTORY: datos puntuales de catalogo (precio, stock, disponibilidad, promociones, garantia por producto, devolucion por producto, envio por producto).

Si es demasiado ambiguo y no alcanza para clasificar con confianza, usa AMBIGUOUS.

REGLAS DE CONSISTENCIA:
- Si auth_route = PUBLIC, query_route no puede ser null y final_route = query_route.
- Si auth_route = PRIVATE, query_route = null y final_route = PRIVATE.
- Si auth_route = AMBIGUOUS, query_route = null y final_route = AMBIGUOUS.

SALIDA OBLIGATORIA (solo JSON):
{
  "auth_route": "PUBLIC" | "PRIVATE" | "AMBIGUOUS",
  "query_route": "FAQ" | "POLICY" | "INVENTORY" | "AMBIGUOUS" | null,
  "final_route": "PRIVATE" | "FAQ" | "POLICY" | "INVENTORY" | "AMBIGUOUS"
}
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
