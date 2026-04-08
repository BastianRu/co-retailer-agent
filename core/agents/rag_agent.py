from strands.models.bedrock import BedrockModel
from strands import Agent
from core.tools.retrieval_context import retrieval_context
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
    max_tokens=800,
    streaming=False
  )

model = build_bedrock_model()

system_prompt = """
Eres RAG_AGENT para soporte de politicas de e-commerce.

Reglas minimas:
- Usa la tool retrieval_context(query) para recuperar secciones relevantes antes de responder.
- Responde unicamente con informacion sustentada por el contexto recuperado.
- Si no hay contexto suficiente, dilo de forma breve y clara.
- No inventes ni infieras informacion. (politicas, plazos o coberturas)

Salida obligatoria: JSON valido y solo JSON.
{
  "route": "ANSWER" | "NO_CONTEXT" | "BLOCK",
  "message": "respuesta breve para el usuario",
  "reason": "explicacion breve"
}
"""


_VALID_ROUTES = {"ANSWER", "NO_CONTEXT", "BLOCK"}


def _extract_code_block(raw: str) -> str:
  if raw.startswith("```"):
    raw = raw.strip("`")
    raw = raw.replace("json", "", 1).strip()
  return raw


def _parse_rag_result(raw: str) -> dict:
  route = "NO_CONTEXT"
  message = "No encontre contexto suficiente para responder con precision."
  reason = "fallback"

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

def solve_rag_query(input: str):
  rag_agent = Agent(
        model=model,
        system_prompt=system_prompt,
        tools=[retrieval_context],
        callback_handler=None,
    )
  response = rag_agent(input)
  raw = _extract_code_block(str(response).strip())
  result = _parse_rag_result(raw)

  return {
    "route": result["route"],
    "message": result["message"],
    "reason": result["reason"],
    "response_data": response,
  }

    

