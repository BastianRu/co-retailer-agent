from __future__ import annotations

from strands.models.bedrock import BedrockModel
from strands import Agent
from dotenv import load_dotenv

from core.session_context import add_tool_trace

import json
import os
import re


load_dotenv()


DETAIL_CONSOLIDATOR_MODEL_ID = (
    os.getenv("BEDROCK_CATALOG_DETAIL_CONSOLIDATOR_MODEL_ID")
    or os.getenv("BEDROCK_CATALOG_CONSOLIDATOR_MODEL_ID")
    or os.getenv("BEDROCK_CONSOLIDATOR_MODEL_ID")
    or "mistral.ministral-3-8b-instruct"
)


def build_bedrock_model() -> BedrockModel:
    return BedrockModel(
        model_id=DETAIL_CONSOLIDATOR_MODEL_ID,
        region_name=os.getenv("AWS_REGION", "us-east-2"),
        temperature=0,
        max_tokens=int(os.getenv("BEDROCK_CATALOG_DETAIL_CONSOLIDATOR_MAX_TOKENS") or 1600),
        streaming=False,
    )


model = build_bedrock_model()


detail_consolidator_system_prompt = """
Eres PRODUCT_DETAIL_CONSOLIDATOR para un retailer e-commerce en Colombia.

Objetivo:
Recibir varias filas que ya representan el mismo producto logico o versiones muy cercanas de ese producto y producir un unico payload canonico para detalle de catalogo.

Debes devolver SIEMPRE JSON valido y SOLO JSON con esta estructura exacta:
{
    "representative_product_id": 5006,
    "name": "nombre canonico",
    "description": "descripcion canonica o null",
    "specifications": "especificaciones canonicas o null",
    "category_id": 1,
    "brand_id": 3,
    "warranty_months": 24,
    "return_days": 60,
    "is_final_sale": false,
    "free_shipping": false,
    "shipping_days": 2,
    "price": null,
    "price_min": 6481556.0,
    "price_max": 6516599.0,
    "active": true,
    "weight_kg": 2.31,
    "requires_installation": false,
    "installation_notes": null
}

Principios:
- Nunca inventes campos ni valores que no puedan defenderse con la entrada.
- Si un valor puntual no es defendible, usa null.
- Si hay multiples precios plausibles para el mismo producto logico, devuelve un rango con price_min y price_max y deja price en null.
- Si hay un solo precio claramente canonico, puedes usar price y dejar price_min/price_max en null.
- El nombre debe ser el mas limpio y representativo posible sin inventar modelos ni especificaciones nuevas.
- Si un sufijo como Plus, Max, Lite, Pro o 2026 parece ruido y no cambia el producto real, no hace falta conservarlo en el nombre canonico.
- Si description o specifications prueban una diferencia real de producto, respeta esa diferencia.

Campos potencialmente ruidosos:
- weight_kg
- installation_notes
- warranty_months
- return_days
- shipping_days
- free_shipping
- is_final_sale
- active

Cuando esos campos entren en conflicto, prioriza la consistencia del producto logico y usa el valor mas defendible. Si no hay uno claramente defendible, usa null.

Debes manejar bien estos patrones de ruido:
- pesos fisicamente improbables para el tipo de producto
- installation_notes copiadas desde otra categoria
- nombres con tokens repetidos como "Max Max", "Pro Pro" o "2026 2026"
- multiples rows con mismas description/specifications pero diferencias operativas poco creibles

No expliques el razonamiento.
No devuelvas texto adicional.
Devuelve solo JSON.
""".strip()


def _to_int(value: object) -> int | None:
    try:
        if value is None:
            return None
        raw = str(value).strip()
        if not raw:
            return None
        return int(float(raw))
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


def _to_bool(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value

    parsed = str(value).strip().lower()
    if not parsed:
        return None
    if parsed in {"1", "true", "t", "yes", "y", "si", "s"}:
        return True
    if parsed in {"0", "false", "f", "no", "n"}:
        return False
    return None


def _prepare_row(row: dict) -> dict:
    return {
        "product_id": _to_int(row.get("product_id")),
        "name": row.get("name"),
        "description": row.get("description"),
        "specifications": row.get("specifications"),
        "category_id": _to_int(row.get("category_id")),
        "brand_id": _to_int(row.get("brand_id")),
        "warranty_months": _to_int(row.get("warranty_months")),
        "return_days": _to_int(row.get("return_days")),
        "is_final_sale": _to_bool(row.get("is_final_sale")),
        "free_shipping": _to_bool(row.get("free_shipping")),
        "shipping_days": _to_int(row.get("shipping_days")),
        "price": _to_float(row.get("price")),
        "active": _to_bool(row.get("active")),
        "weight_kg": _to_float(row.get("weight_kg")),
        "requires_installation": _to_bool(row.get("requires_installation")),
        "installation_notes": row.get("installation_notes"),
    }


def _extract_code_block(raw: str) -> str:
    text = str(raw or "").strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    if text.startswith("```"):
        return re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE).strip()
    return text


def _parse_detail(raw: str) -> dict | None:
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

    if not isinstance(data, dict):
        return None

    return {
        "representative_product_id": _to_int(data.get("representative_product_id")),
        "name": str(data.get("name") or "").strip() or None,
        "description": str(data.get("description") or "").strip() or None,
        "specifications": str(data.get("specifications") or "").strip() or None,
        "category_id": _to_int(data.get("category_id")),
        "brand_id": _to_int(data.get("brand_id")),
        "warranty_months": _to_int(data.get("warranty_months")),
        "return_days": _to_int(data.get("return_days")),
        "is_final_sale": _to_bool(data.get("is_final_sale")),
        "free_shipping": _to_bool(data.get("free_shipping")),
        "shipping_days": _to_int(data.get("shipping_days")),
        "price": _to_float(data.get("price")),
        "price_min": _to_float(data.get("price_min")),
        "price_max": _to_float(data.get("price_max")),
        "active": _to_bool(data.get("active")),
        "weight_kg": _to_float(data.get("weight_kg")),
        "requires_installation": _to_bool(data.get("requires_installation")),
        "installation_notes": str(data.get("installation_notes") or "").strip() or None,
    }


def consolidate_product_details(rows: list[dict]) -> dict | None:
    input_data = {
        "row_count": len(rows),
        "product_ids": [row.get("product_id") for row in rows],
    }

    def _trace_return(output: dict | None, error: str | None = None) -> dict | None:
        traced_output = {"detail": output, "model_id": DETAIL_CONSOLIDATOR_MODEL_ID}
        if error:
            traced_output["error"] = error
        add_tool_trace("consolidate_product_details", input_data, traced_output)
        return output

    prepared_rows = [
        prepared
        for prepared in (_prepare_row(row) for row in rows)
        if prepared.get("product_id") is not None
    ]
    if not prepared_rows:
        return _trace_return(None)

    prompt_payload = {
        "row_count": len(prepared_rows),
        "rows": prepared_rows,
    }

    try:
        agent = Agent(
            model=model,
            system_prompt=detail_consolidator_system_prompt,
            callback_handler=None,
        )
        response = agent(json.dumps(prompt_payload, ensure_ascii=False, indent=2))
        detail = _parse_detail(str(response))
        return _trace_return(detail)
    except Exception as exc:
        return _trace_return(None, error=f"{type(exc).__name__}: {exc}")