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
Eres un clasificador de intención para un agente de e-commerce en Colombia.

Tu única tarea es clasificar la consulta del usuario en exactamente una de estas categorías:

- FAQ
- POLICY
- INVENTORY
- AMBIGUOUS

Definiciones:

1. FAQ
Consultas generales y operativas que pueden responderse directamente sin consultar datos específicos de productos ni aplicar recuperación documental obligatoria.
Ejemplos:
- métodos o medios de pago
- cobertura de envíos
- tiempos generales de envío
- canales de atención
- cómo descargar factura
- cómo consultar tracking en general
- significado general de estados del pedido
- cómo reportar un problema
- dónde comprar o si se puede comprar por WhatsApp

2. POLICY
Consultas sobre reglas, condiciones, cobertura, exclusiones, plazos o procesos formales de:
- devoluciones y cambios
- garantía
- envíos
- reembolsos
- cancelaciones
- modificación de dirección
- responsabilidad de la empresa
- condiciones de entrega
- productos en promoción respecto a devolución/garantía
- cuándo aplica envío gratis como regla general

Estas consultas deben tratarse como políticas incluso si mencionan palabras como "producto", "pedido", "envío" o "garantía".

3. INVENTORY
Consultas que requieren conocer información específica de catálogo o disponibilidad actual de productos.
Incluye:
- precio
- stock
- existencias
- disponibilidad
- si un producto sigue a la venta
- si está agotado
- cuántas unidades quedan
- si tiene envío gratis a nivel de producto
- si tiene promoción, descuento u oferta a nivel de producto
- precio con promoción
- combinación de precio + stock + disponibilidad

Si la consulta pide un valor o estado específico de un producto o referencia, clasifica como INVENTORY.

4. AMBIGUOUS
Usa esta categoría solo si la consulta no permite distinguir de forma confiable entre dos o más categorías anteriores.

Reglas de decisión importantes:

- Si la consulta trata sobre precio, stock, existencias, disponibilidad, promoción, descuento, oferta o envío gratis de un producto específico, clasifica como INVENTORY.
- Si la consulta trata sobre normas, condiciones, cobertura, tiempos, exclusiones o procedimientos, clasifica como POLICY.
- Si la consulta trata sobre orientación general de uso del servicio, canales, pagos, cobertura, tracking general, factura o estados generales, clasifica como FAQ.
- No clasifiques como INVENTORY solo porque aparezcan palabras como "producto", "comprar", "pedido" o "envío".
- No clasifiques como FAQ si la consulta pregunta por reglas o condiciones formales.
- Si dudas entre FAQ y POLICY, prefiere POLICY cuando la consulta suene a regla, condición, cobertura, exclusión, plazo o proceso.
- Si dudas entre POLICY e INVENTORY, prefiere INVENTORY solo cuando la consulta pida un dato específico y actual de catálogo.

Formato de salida obligatorio:

Responde únicamente en JSON válido con esta forma exacta:
{"route":"FAQ"}
o
{"route":"POLICY"}
o
{"route":"INVENTORY"}
o
{"route":"AMBIGUOUS"}

No agregues explicaciones.
No agregues texto adicional.
"""

def classify_public_route(input: str):
  public_routing_agent = Agent(
  model=model,
  system_prompt=system_prompt,
  callback_handler=None,
  )
  response = public_routing_agent(input)
  raw = str(response).strip()

  if raw.startswith("```"):
    raw = raw.strip("`")
    raw = raw.replace("json", "", 1).strip()

  route = "UNKNOWN"
  try:
        data = json.loads(raw)
        route = data.get("route", "UNKNOWN")
  except json.JSONDecodeError:
        m = re.search(r"\b(POLICY|FAQ|INVENTORY|AMBIGUOUS)\b", raw.upper())
        route = m.group(1) if m else "UNKNOWN"

  return {
    "route": route,
    "response_data": response
  }




