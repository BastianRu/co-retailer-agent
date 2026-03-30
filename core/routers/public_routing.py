from semantic_router import Route
from semantic_router.encoders import OllamaEncoder
from semantic_router.routers import SemanticRouter
import unicodedata
from typing import Any, Optional

#Class thresholds (Hyperparameters)
INVENTORY_THRESHOLD = 0.60
MIN_DELTA = 0.04

routes = [
    Route(
        name="FAQ",
        utterances=[
            "que metodos de pago manejan",
            "aceptan nequi o daviplata",
            "hacen envios a todo colombia",
            "¿Hacen envíos a todo Colombia?",
            "hacen envios a mi ciudad",
            "cuales son los canales de atencion",
            "como contacto soporte",
            "como hago seguimiento a un pedido en general",
            "donde puedo consultar el estado de un pedido",
            "como descargar la factura",
            "como reportar un problema con un pedido",
            "cuanto tarda un envio a ciudades principales",
            "cuanto tarda un envio a zonas rurales"
           # "Cuanto tarde un envío a zonas rurales" #The accent has a high impact
        ]
    ),
    Route(
        name="Politicas",
        utterances=[
            "cual es la politica de devoluciones",
            "cual es la politica de garantia",
            "cuales son las politicas de envio",
            "que cubre la garantia",
            "que no cubre la garantia",
            "cuanto tiempo tengo para devolver un producto",
            "cuanto tarda el reembolso",
            "puedo cancelar una compra antes del despacho",
            "puedo cambiar un producto",
            "que productos no tienen devolucion",
            "que pasa si compre un producto en promocion",
            "puedo cambiar la direccion de envio antes del despacho",
            #"¿Cuánto tarda un reembolso?", #acc
            #"¿Qué pasa si el producto fue comprado en promoción?", #acc
            #"¿Qué productos no tienen devolución?" #acc
        ]
    ),
    Route(
        name="Inventario",
        utterances=[
            "cuanto cuesta este producto",
            "cual es el precio del producto",
            "tienen stock disponible",
            "hay unidades disponibles",
            "el producto tiene envio gratis",
            "que precio tiene el iphone 13",
            "cuanto vale una air fryer oster",
            "hay inventario del samsung galaxy a55",
            "este producto esta disponible",
            "cuantas unidades quedan",
            "el producto esta agotado",
            "que productos tienen envio gratis",
            "sigue disponible este articulo",
            "hay disponibilidad de este articulo",
            "lo tienen disponible",
            "se puede comprar todavia",
            "este producto aun esta a la venta",
            "esta habilitada la compra de este producto",
            "necesito saber si esta disponible",
            "quisiera saber sobre disponibilidad",
            "me interesa este producto, hay",
            "buenas me interesa este articulo esta disponible",
            "disculpa cuanto cuesta esto",
            "quisiera saber si aun tienen unidades",
            "cuanto cuesta y si hay stock",
            "cual es el precio y disponibilidad",
            "cuanto cuesta con envio",
            "el precio incluye envio",
            "esta en oferta este producto",
            "este producto tiene descuento",
            "este producto esta en promocion",
            "se puede comprar y cuanto cuesta",
            "¿Esta agotado?",
            "Oye, ¿todavia venden este producto?",
            "¿Este producto está en promoción?"
        ],
        score_threshold=INVENTORY_THRESHOLD
    )
]


encoder = OllamaEncoder(
    name="nomic-embed-text",
    base_url="http://localhost:11434",
    score_threshold=0.5,
)

router = SemanticRouter(
    encoder=encoder,
    routes=routes,
    auto_sync="local",
)

VOWELS_TABLE = str.maketrans("", "", "aeiouAEIOU")

def remove_accent(input: str) -> str:
    t = unicodedata.normalize("NFKD", input)
    return "".join(c for c in t if not unicodedata.combining(c))

def classify_public_intent(input: str):
    lower_input = input.lower()
    normalized = remove_accent(lower_input)
    return router(normalized)

def inspect_public_routing(
    input: str,
    limit: Optional[int] = None,
    route_filter: Optional[list[str]] = None,
) -> dict[str, Any]:
    #Return routing diagnostics including raw top-k and threshold-aware results
    lower_input = input.lower()
    normalized = remove_accent(lower_input)

    vector = router._encode(text=[normalized], input_type="queries")
    query_vector = vector[0]

    raw_scores, raw_routes = router.index.query(
        vector=query_vector,
        top_k=router.top_k,
        route_filter=route_filter,
    )

    raw_top_k = [
        {
            "route": str(route_name),
            "score": float(score),
        }
        for route_name, score in zip(raw_routes, raw_scores)
    ]

    passed = router(
        normalized,
        limit=limit,
        route_filter=route_filter,
    )

    if isinstance(passed, list):
        passed_routes = passed
    elif getattr(passed, "name", None):
        passed_routes = [passed]
    else:
        passed_routes = []

    passed_serialized = [
        {
            "name": route_choice.name,
            "similarity_score": (
                float(route_choice.similarity_score)
                if route_choice.similarity_score is not None
                else None
            ),
        }
        for route_choice in passed_routes
    ]

    thresholds_by_route = {
        route.name: (
            float(route.score_threshold) if route.score_threshold is not None else None
        )
        for route in router.routes
    }

    return {
        "router_top_k": router.top_k,
        "thresholds_by_route": thresholds_by_route,
        "raw_top_k": raw_top_k,
        "passed_routes": passed_serialized,
    }


def passes_delta(
    input: str,
    min_delta: float = MIN_DELTA,
    route_filter: Optional[list[str]] = None,
) -> dict:
    """Return top-2 route margin check using aggregated route scores."""
    lower_input = input.lower()
    normalized = remove_accent(lower_input)

    vector = router._encode(text=[normalized], input_type="queries")
    query_vector = vector[0]

    raw_scores, raw_routes = router.index.query(
        vector=query_vector,
        top_k=router.top_k,
        route_filter=route_filter,
    )

    query_results = [
        {"route": str(route_name), "score": float(score)}
        for route_name, score in zip(raw_routes, raw_scores)
    ]
    scored_routes = router._score_routes(query_results=query_results)

    if len(scored_routes) < 2:
        return {
            "passes": True,
            "delta": None
        }

    delta = float(scored_routes[0][1] - scored_routes[1][1])
    return {
        "passes": delta >= min_delta,
        "delta": delta
    }
