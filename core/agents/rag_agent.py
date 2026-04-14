from strands.models.bedrock import BedrockModel
from strands import Agent
from core.tools.retrieval_context import retrieval_context
from core.session_context import register_reset_callback, get_tool_trace_length, get_tool_trace_since
from dotenv import load_dotenv
import json
import re
import os

load_dotenv()

#Model providers 

#Bedrock
def build_bedrock_model() -> BedrockModel:
  return BedrockModel(
    model_id="mistral.ministral-3-8b-instruct",
    region_name=os.getenv("AWS_REGION", "us-east-2"),
    temperature=0,
    max_tokens=1500,
    streaming=False
  )

model = build_bedrock_model()

system_prompt = """
Eres RAG_AGENT para politicas de e-commerce.

Reglas obligatorias:
- Usa retrieval_context(query) antes de responder.
- Responde solo con informacion textual del contexto recuperado.
- No inventes, no extrapoles y no combines reglas para crear una regla nueva.
- Mantener respuesta breve y literal.
- Los tiempos de devolucion de un prodcuto son siempre desde la fecha de ENTREGA no de COMPRA.

Precision critica:
- Si hay regla general y regla por categoria, reportalas por separado, sin conectores causales inventados.
- No uses frases como "independientemente de", "incluso si", "aunque" si no aparecen en la fuente.
- No uses formulaciones especulativas como "puede no aplicar", "podria aplicar" o "normalmente" cuando el texto fuente no lo dice.
- No uses absolutos no textuales como "en ningun caso", "siempre", "nunca" o equivalentes, salvo que aparezcan explicitamente en la fuente.
- Si una pregunta pide un medio especifico (ej. contraentrega) y ese medio no aparece explicitamente en el texto, di que no hay plazo especifico documentado.
- No infieras condiciones no presentes (ej. urbano/rural, intentos distintos, excepciones no escritas).
- Si el mensaje es ambiguo por referencia contextual (ej. "y eso aplica...") y no hay antecedente claro en el mensaje actual, pide aclaracion breve.

Formato recomendado para garantia y agua:
- "En general: <regla general textual>. En electronica: <regla de electronica textual>."
- Para este tema, usa literalmente la excepcion IP de la regla general y evita reinterpretarla.
- Para este tema, evita agregar frases como "en ningun caso"; usa literal: "En electronica: no cubre danos por agua".

Salida obligatoria: JSON valido y solo JSON.
{
  "route": "ANSWER" | "NO_CONTEXT" | "BLOCK",
  "message": "respuesta para el usuario con la informacion solicitada",
  "reason": "explicacion breve"
}
"""


_VALID_ROUTES = {"ANSWER", "NO_CONTEXT", "BLOCK"}
_DEFAULT_RAG_MESSAGE = "No encontre contexto suficiente para responder con precision."
_DEFAULT_RAG_REASON = "fallback"


def _extract_code_block(raw: str) -> str:
  raw = str(raw or "").strip()
  match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, flags=re.IGNORECASE | re.DOTALL)
  if match:
    return match.group(1).strip()
  if raw.startswith("```"):
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    return raw.strip()
  return raw


def _extract_json_string_field(text: str, field_name: str) -> str | None:
  field_key = re.search(rf'"{field_name}"\s*:\s*"', text, flags=re.IGNORECASE)
  if not field_key:
    return None

  index = field_key.end()
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
  candidate = candidate.replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"')
  candidate = candidate.strip()
  return candidate or None


def _infer_rag_route_from_text(text: str) -> str:
  lowered = str(text or "").strip().lower()
  if not lowered:
    return "NO_CONTEXT"
  if "no puedo procesar" in lowered:
    return "BLOCK"
  if "no encontre contexto suficiente" in lowered or "no encontré contexto suficiente" in lowered:
    return "NO_CONTEXT"
  return "ANSWER"


def _recover_plain_text_rag_result(raw: str, traces: list[dict]) -> dict | None:
  cleaned = _extract_code_block(raw).strip()
  if not cleaned or cleaned.startswith("{"):
    return None

  has_context = False
  for trace in traces:
    if str(trace.get("tool_name", "")).strip() != "retrieval_context":
      continue
    output_data = trace.get("output_data", {}) or {}
    if isinstance(output_data, dict) and output_data.get("results"):
      has_context = True
      break

  if not has_context:
    return None

  return {
    "route": _infer_rag_route_from_text(cleaned),
    "message": cleaned,
    "reason": "plain_text_recovery",
  }


def _parse_rag_result(raw: str) -> dict:
  route = "NO_CONTEXT"
  message = _DEFAULT_RAG_MESSAGE
  reason = _DEFAULT_RAG_REASON

  try:
    data = json.loads(raw)
    parsed_route = str(data.get("route", "")).strip().upper()
    if parsed_route in _VALID_ROUTES:
      route = parsed_route
    message = str(data.get("message", message)).strip()
    reason = str(data.get("reason", reason)).strip()
  except json.JSONDecodeError:
    route_match = re.search(r"\b(ANSWER|NO_CONTEXT|BLOCK)\b", raw.upper())
    if route_match:
      route = route_match.group(1)

    message_match = _extract_json_string_field(raw, "message")
    reason_match = _extract_json_string_field(raw, "reason")

    if message_match:
      message = message_match.strip()
    if reason_match:
      reason = reason_match.strip()

  return {
    "route": route,
    "message": message,
    "reason": reason,
  }

_rag_agent = None


def _create_rag_agent():
  return Agent(
      model=model,
      system_prompt=system_prompt,
      tools=[retrieval_context],
      callback_handler=None,
  )


def init_rag_agent():
  global _rag_agent
  _rag_agent = _create_rag_agent()


def reset_rag_agent():
  init_rag_agent()


register_reset_callback(reset_rag_agent)
init_rag_agent()


def solve_rag_query(input: str):
  trace_start_idx = get_tool_trace_length()
  agent = _rag_agent
  if agent is None:
    init_rag_agent()
    agent = _rag_agent
    if agent is None:
      return {
        "route": "NO_CONTEXT",
        "message": _DEFAULT_RAG_MESSAGE,
        "reason": "agent_not_initialized",
        "response_data": None,
      }

  response = agent(input)
  raw = _extract_code_block(str(response).strip())
  result = _parse_rag_result(raw)

  if result["message"] == _DEFAULT_RAG_MESSAGE:
    traces = get_tool_trace_since(trace_start_idx)
    recovered = _recover_plain_text_rag_result(str(response).strip(), traces)
    if recovered is not None:
      result = recovered

  return {
    "route": result["route"],
    "message": result["message"],
    "reason": result["reason"],
    "response_data": response,
  }

    

