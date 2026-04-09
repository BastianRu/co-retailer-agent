from difflib import SequenceMatcher
from itertools import combinations

import pandas as pd
from strands import tool

from core.data_store import load_s3_data
from core.session_context import add_tool_trace
from core.tools.inventory.search_products import _normalize_text, _score_fuzzy, _score_partial


def _price_similarity(a: float | None, b: float | None) -> float:
    if a is None or b is None or a <= 0 or b <= 0:
        return 0.5
    return max(0.0, 1.0 - (abs(a - b) / max(a, b)))


def _text_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


@tool
def verify_product_identity(query: str, top_k: int = 8) -> dict:
    """
    Verify whether matched products are highly likely to be the same logical product with noisy names.

    Use this tool when search results for a query look duplicated/variant-like and you need
    a boolean confidence signal to decide if they should be consolidated as one product.

    Args:
        query: Product query text (for example: "iphone 15 pro max").
        top_k: Max number of matched products to analyze (1..12).

    Returns:
        dict: Verification payload with:
            - is_same_product_identity: bool
            - confidence: float (0..1)
            - candidate_count: int
            - reason: str
        Returns False confidence when there are not enough candidates or source data is unavailable.
    """
    input_data = {"query": query, "top_k": top_k}

    def _trace_return(output: dict) -> dict:
        add_tool_trace("verify_product_identity", input_data, output)
        return output

    query_norm = _normalize_text(query)
    if not query_norm:
        return _trace_return(
            {
                "is_same_product_identity": False,
                "confidence": 0.0,
                "candidate_count": 0,
                "reason": "query vacio",
            }
        )

    products = load_s3_data("products.csv")
    if not isinstance(products, pd.DataFrame) or products.empty:
        return _trace_return(
            {
                "is_same_product_identity": False,
                "confidence": 0.0,
                "candidate_count": 0,
                "reason": "products table unavailable",
            }
        )

    df = products.copy()
    required = {"name", "brand_id", "category_id", "description", "price"}
    if not required.issubset(df.columns):
        return _trace_return(
            {
                "is_same_product_identity": False,
                "confidence": 0.0,
                "candidate_count": 0,
                "reason": "products schema mismatch",
            }
        )

    df["_name_norm"] = df["name"].map(_normalize_text)
    df["_desc_norm"] = df["description"].map(_normalize_text)
    df["_brand"] = pd.to_numeric(df["brand_id"], errors="coerce")
    df["_category"] = pd.to_numeric(df["category_id"], errors="coerce")
    df["_price"] = pd.to_numeric(df["price"], errors="coerce")

    # Match candidates similarly to discovery behavior: partial first, then fuzzy.
    candidates = df[df["_name_norm"].str.contains(query_norm, na=False)].copy()
    if candidates.empty:
        fuzzy = df.copy()
        fuzzy["_score"] = fuzzy["_name_norm"].apply(lambda n: _score_fuzzy(query_norm, n))
        candidates = fuzzy[fuzzy["_score"] >= 60.0].copy()
    else:
        candidates["_score"] = candidates["_name_norm"].apply(lambda n: _score_partial(query_norm, n))

    k = max(1, min(int(top_k), 12))
    candidates = candidates.sort_values(by=["_score", "name"], ascending=[False, True]).head(k)

    if len(candidates) < 2:
        return _trace_return(
            {
                "is_same_product_identity": False,
                "confidence": 0.0,
                "candidate_count": int(len(candidates)),
                "reason": "insufficient comparable candidates",
            }
        )

    pair_scores: list[float] = []
    same_brand_pairs = 0
    same_category_pairs = 0
    total_pairs = 0

    rows = list(candidates.to_dict(orient="records"))
    for left, right in combinations(rows, 2):
        total_pairs += 1

        name_sim = _text_similarity(left.get("_name_norm", ""), right.get("_name_norm", ""))
        desc_sim = _text_similarity(left.get("_desc_norm", ""), right.get("_desc_norm", ""))

        same_brand = left.get("_brand") == right.get("_brand")
        same_category = left.get("_category") == right.get("_category")
        if same_brand:
            same_brand_pairs += 1
        if same_category:
            same_category_pairs += 1

        price_sim = _price_similarity(left.get("_price"), right.get("_price"))

        score = (
            (0.40 * name_sim)
            + (0.15 * (1.0 if same_brand else 0.0))
            + (0.15 * (1.0 if same_category else 0.0))
            + (0.20 * desc_sim)
            + (0.10 * price_sim)
        )
        pair_scores.append(score)

    confidence = sum(pair_scores) / len(pair_scores) if pair_scores else 0.0
    brand_ratio = same_brand_pairs / total_pairs if total_pairs else 0.0
    category_ratio = same_category_pairs / total_pairs if total_pairs else 0.0

    is_same = confidence >= 0.82 and brand_ratio >= 0.8 and category_ratio >= 0.8

    reason = (
        f"confidence={confidence:.2f}, same_brand_ratio={brand_ratio:.2f}, "
        f"same_category_ratio={category_ratio:.2f}"
    )

    return _trace_return(
        {
            "is_same_product_identity": is_same,
            "confidence": round(confidence, 4),
            "candidate_count": int(len(candidates)),
            "reason": reason,
        }
    )
