from __future__ import annotations

from strands.models.bedrock import BedrockModel
from strands import Agent
from dotenv import load_dotenv

from core.session_context import add_tool_trace

import json
import os
import re


load_dotenv()


CONSOLIDATOR_MODEL_ID = (
    os.getenv("BEDROCK_CATALOG_CONSOLIDATOR_MODEL_ID")
    or os.getenv("BEDROCK_CONSOLIDATOR_MODEL_ID")
    or "mistral.ministral-3-8b-instruct"
)


def build_bedrock_model() -> BedrockModel:
    return BedrockModel(
        model_id=CONSOLIDATOR_MODEL_ID,
        region_name=os.getenv("AWS_REGION", "us-east-2"),
        temperature=0,
        max_tokens=int(os.getenv("BEDROCK_CATALOG_CONSOLIDATOR_MAX_TOKENS") or 1400),
        streaming=False,
    )


model = build_bedrock_model()


consolidator_system_prompt = """
Eres PRODUCT_CANDIDATE_CONSOLIDATOR para un retailer e-commerce en Colombia.

Objetivo:
Decidir que filas candidatas representan el mismo producto logico y devolver una
agrupacion consolidada para el catalogo publico.

Contexto:
- La query ya fue usada para recuperar candidatos por busqueda lexica.
- Los candidatos pueden incluir duplicados reales, variantes reales y ruido sintetico.
- Tu trabajo NO es responder al usuario final.
- Tu trabajo SI es decidir como agrupar candidatos que representen el mismo producto.

Debes devolver SIEMPRE JSON valido y SOLO JSON con esta estructura exacta:
{
    "groups": [
        {
            "product_ids": [5001, 5002],
            "representative_product_id": 5001,
            "canonical_name": "nombre canonico"
        }
    ]
}

Reglas obligatorias:
- Cada product_id de entrada debe aparecer exactamente una vez en toda la salida.
- representative_product_id debe pertenecer al mismo grupo.
- Nunca inventes product_ids.
- Nunca inventes un producto nuevo.
- Si la evidencia es ambigua o insuficiente, NO agrupes.
- Es preferible separar de mas que fusionar productos realmente distintos.

Reglas para canonical_name:
- Debe ser uno de los nombres presentes en el grupo, o
- una limpieza minima de uno de esos nombres eliminando ruido obvio como tokens duplicados adyacentes o el mismo año repetido dos veces.
- Nunca inventes una familia de producto, modelo, capacidad, almacenamiento, tamano, BTU, generacion, procesador, edicion o color que no exista literalmente en la entrada.
- Si no estas completamente seguro, usa el nombre exacto del representative_product_id.

Como decidir si varios rows son el mismo producto:
- Prioriza la evidencia de:
    - brand_id
    - category_id
    - name
    - description
    - specifications
- Trata como campos potencialmente ruidosos o menos confiables cuando entren en conflicto con la evidencia principal:
    - weight_kg
    - installation_notes
    - warranty_months
    - return_days
    - shipping_days
    - free_shipping
    - is_final_sale
- Los sufijos Plus, Max, Lite, Pro y 2026 deben tratarse como ruido por defecto.
- Solo conserva esos sufijos como diferencia real si description o specifications muestran una diferencia concreta de producto.
- Si dos rows tienen la misma descripcion base y las mismas especificaciones esenciales, un sufijo aislado NO basta para separarlos.
- Un año aislado como "2026" NO crea un producto distinto por si solo.
- Un adjetivo repetido o apilado como "Pro Pro" o "Max Max" NO crea un producto distinto por si solo.

Debes manejar bien estos casos:
- Pesos fisicamente improbables para el tipo de producto.
- installation_notes que parecen copiadas desde otra categoria.
- Nombres duplicados o casi duplicados con tokens repetidos como "Max Max", "Pro Pro" o "2026 2026".
- Filas del mismo producto con diferencias poco creibles en warranty_months, return_days, shipping_days, free_shipping o is_final_sale.
- Productos casi identicos donde la diferencia real si existe en capacidad, tamano, generacion, BTU, procesador, edicion o especificaciones.

Ejemplos de filas que SI debes agrupar cuando description/specifications no muestran una diferencia real:
- "iPhone 15 Pro Max", "iPhone 15 Pro Max Max", "iPhone 15 Pro Max Pro"
- "Agenda Ejecutiva 2026", "Agenda Ejecutiva 2026 Pro", "Agenda Ejecutiva 2026 2026"
- "Licuadora Oster Pro", "Licuadora Oster Pro Pro", "Licuadora Oster Pro Plus", "Licuadora Oster Pro 2026"
- "Set Skincare La Roche-Posay 2026" repetido con pequeños cambios de warranty, return_days, shipping o weight_kg

Ejemplos de filas que NO debes agrupar:
- mismo nombre base pero diferente almacenamiento o capacidad real
- mismo nombre base pero diferentes especificaciones tecnicas fundamentales
- mismo producto familiar pero distinta generacion confirmada por description/specifications

Casos en los que NO debes agrupar:
- Diferente capacidad, almacenamiento, tamano de pantalla, BTU, procesador, generacion o edicion cuando eso cambia el producto real.
- Diferencias claras en description/specifications que indiquen productos distintos.
- Cuando no puedas defender claramente que son el mismo producto logico.

Ordena los groups desde el mejor match general con la query hasta el peor.

No expliques tu razonamiento.
No devuelvas texto adicional.
Devuelve solo JSON.
""".strip()


def _to_int(value: object) -> int | None:
    try:
        if value is None:
            return None
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def _to_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        raw = str(value).strip()
        if not raw:
            return None
        return float(raw)
    except (TypeError, ValueError):
        return None


def _prepare_candidate(candidate: dict) -> dict:
    return {
        "product_id": _to_int(candidate.get("product_id")),
        "name": candidate.get("name"),
        "description": candidate.get("description"),
        "specifications": candidate.get("specifications"),
        "category_id": _to_int(candidate.get("category_id")),
        "brand_id": _to_int(candidate.get("brand_id")),
        "warranty_months": _to_int(candidate.get("warranty_months")),
        "return_days": _to_int(candidate.get("return_days")),
        "is_final_sale": candidate.get("is_final_sale"),
        "free_shipping": candidate.get("free_shipping"),
        "shipping_days": _to_int(candidate.get("shipping_days")),
        "price": _to_float(candidate.get("price")),
        "active": candidate.get("active"),
        "weight_kg": _to_float(candidate.get("weight_kg")),
        "requires_installation": candidate.get("requires_installation"),
        "installation_notes": candidate.get("installation_notes"),
        "match_type": candidate.get("_match_type"),
        "score": round(float(candidate.get("_score", 0.0)), 2),
    }


def _extract_code_block(raw: str) -> str:
    text = str(raw or "").strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    if text.startswith("```"):
        return re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE).strip()
    return text


def _parse_groups(raw: str) -> list[dict] | None:
    payload = _extract_code_block(raw)
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        json_match = re.search(r"\{.*\}", payload, flags=re.DOTALL)
        if not json_match:
            return None
        try:
            data = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            return None

    groups = data.get("groups")
    if not isinstance(groups, list):
        return None

    normalized_groups: list[dict] = []
    for group in groups:
        if not isinstance(group, dict):
            continue

        product_ids: list[int] = []
        for product_id in group.get("product_ids", []):
            parsed = _to_int(product_id)
            if parsed is not None and parsed not in product_ids:
                product_ids.append(parsed)

        if not product_ids:
            continue

        representative_product_id = _to_int(group.get("representative_product_id"))
        if representative_product_id not in product_ids:
            representative_product_id = product_ids[0]

        canonical_name = str(group.get("canonical_name") or "").strip() or None

        normalized_groups.append(
            {
                "product_ids": product_ids,
                "representative_product_id": representative_product_id,
                "canonical_name": canonical_name,
            }
        )

    return normalized_groups or None


def group_product_candidates(query: str, candidates: list[dict]) -> list[dict] | None:
    input_data = {
        "query": query,
        "candidate_count": len(candidates),
        "candidate_product_ids": [candidate.get("product_id") for candidate in candidates],
    }

    def _trace_return(output: list[dict] | None, error: str | None = None) -> list[dict] | None:
        traced_output = {"groups": output, "model_id": CONSOLIDATOR_MODEL_ID}
        if error:
            traced_output["error"] = error
        add_tool_trace("group_product_candidates", input_data, traced_output)
        return output

    prepared_candidates = [
        prepared
        for prepared in (_prepare_candidate(candidate) for candidate in candidates)
        if prepared.get("product_id") is not None
    ]
    if len(prepared_candidates) < 2:
        return _trace_return(None)

    prompt_payload = {
        "query": query,
        "candidate_count": len(prepared_candidates),
        "candidates": prepared_candidates,
    }

    try:
        agent = Agent(
            model=model,
            system_prompt=consolidator_system_prompt,
            callback_handler=None,
        )

        response = agent(json.dumps(prompt_payload, ensure_ascii=False, indent=2))
        groups = _parse_groups(str(response))
        return _trace_return(groups)
    except Exception as exc:
        return _trace_return(None, error=f"{type(exc).__name__}: {exc}")