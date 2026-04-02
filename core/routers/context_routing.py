import os
from dotenv import load_dotenv
from strands import Agent
from strands.models.bedrock import BedrockModel
from strands.models.ollama import OllamaModel
import json
import re

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
    max_tokens=60,
    streaming=False
  )

model = build_bedrock_model()

system_prompt = """
Eres un clasificador de contexto conversacional para un agente de e-commerce.

Tu única tarea es clasificar el mensaje actual del usuario en exactamente una de estas categorías:

- AUTO
- FOLLOW
- TALK

Definiciones:

1. AUTO
El mensaje es autónomo y se entiende por sí solo, sin depender del turno anterior.
Contiene una intención o pregunta completa que puede enviarse directamente al router principal.

Ejemplos:
- "¿Qué métodos de pago manejan?"
- "¿Cuánto cuesta este producto?"
- "Quiero saber el estado de mi pedido"
- "¿Cuál es la política de devoluciones?"
- "¿Cuánto vale [X producto]"

2. FOLLOW
El mensaje depende del contexto previo y no se entiende completamente por sí solo.
Suele usar referencias como:
- lo, la, los, las
- eso, ese, esa, este, esta
- ahí, entonces, también
- todavía, ya, ya mismo
- y cuánto vale, y aplica, y eso, y cómo así

También aplica cuando:
- el mensaje es muy corto y claramente continúa el tema anterior
- el usuario retoma una entidad mencionada antes sin repetirla
- el mensaje actual necesita reescritura usando contexto previo

Ejemplos:
- "¿Lo tienen?"
- "¿Y cuánto vale?"
- "¿Todavía hay?"
- "¿Y aplica garantía?"
- "¿Y si ya fue despachado?"
- "Eso también tiene envío gratis?"

3. TALK
El mensaje es charla social o conversacional y no debe pasar al router de negocio.
Incluye:
- saludos
- despedidas
- agradecimientos
- confirmaciones simples
- reacciones breves
- cortesía sin intención de negocio clara

Ejemplos:
- "hola"
- "buenas"
- "gracias"
- "ok"
- "dale"
- "perfecto"
- "jajaja"

Reglas importantes:

- Si el mensaje expresa una intención completa por sí solo, clasifica como AUTO.
- Si el mensaje depende del turno anterior para saber de qué habla, clasifica como FOLLOW.
- Si el mensaje es solo charla social, clasifica como TALK.
- Si dudas entre AUTO y FOLLOW, usa FOLLOW solo cuando realmente falte contexto para entender el mensaje.
- Si dudas entre TALK y AUTO, usa TALK solo cuando no exista intención de negocio clara.

Formato de salida OBLIGATORIO:

Responde únicamente en JSON válido con esta forma exacta:
{"route":"AUTO"}
o
{"route":"FOLLOW"}
o
{"route":"TALK"}

No agregues explicaciones.
No agregues texto adicional.
"""

def classify_context_route(input: str):
  context_routing_agent = Agent(
  model=model,
  system_prompt=system_prompt,
  callback_handler=None,
)
  response = context_routing_agent(input)
  raw = str(response).strip()

  if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.replace("json", "", 1).strip()

  route = "UNKNOWN"
  try:
        data = json.loads(raw)
        route = data.get("route", "UNKNOWN")
  except json.JSONDecodeError:
        m = re.search(r"\b(AUTO|FOLLOW|TALK)\b", raw.upper())
        route = m.group(1) if m else "UNKNOWN"

  return {
    "route": route,
    "response_data": response
  }





