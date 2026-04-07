from core.data_store import load_s3_data
from core.session_context import add_tool_trace
from strands import tool
from difflib import SequenceMatcher
from datetime import date
import pandas as pd
import re
import unicodedata


def _normalize_text(value: object) -> str:
    text = str(value or "").strip().lower()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value

    parsed = str(value).strip().lower()
    return parsed in {"1", "true", "t", "yes", "y", "si", "s"}


def _to_int(value: object) -> int | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None

    raw = str(value).strip()
    if not raw:
        return None

    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return None


def _split_pipe_ids(value: object) -> set[int]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return set()

    output: set[int] = set()
    for token in str(value).split("|"):
        parsed = _to_int(token)
        if parsed is not None:
            output.add(parsed)
    return output


def _score_partial(query_norm: str, name_norm: str) -> float:
    if not query_norm:
        return 0.0
    if name_norm.startswith(query_norm):
        return 98.0
    if f" {query_norm} " in f" {name_norm} ":
        return 95.0
    return 90.0


def _score_fuzzy(query_norm: str, name_norm: str) -> float:
    if not query_norm or not name_norm:
        return 0.0

    # Blend full-string ratio with token-level best match for typo tolerance.
    global_ratio = SequenceMatcher(None, query_norm, name_norm).ratio()
    token_ratios = [SequenceMatcher(None, token, name_norm).ratio() for token in query_norm.split()]
    token_best = max(token_ratios) if token_ratios else 0.0
    return ((0.8 * global_ratio) + (0.2 * token_best)) * 100.0


def _build_stock_lookup() -> dict[int, dict]:
    stock_data = load_s3_data("stock.csv")
    if not isinstance(stock_data, pd.DataFrame) or stock_data.empty:
        return {}

    stock_df = stock_data.copy()
    required_columns = {
        "product_id",
        "warehouse_location",
        "stock_qty",
        "reserved_qty",
        "low_stock_threshold",
    }
    if not required_columns.issubset(stock_df.columns):
        return {}

    stock_df["_pid"] = stock_df["product_id"].map(_to_int)
    stock_df = stock_df[stock_df["_pid"].notna()].copy()
    if stock_df.empty:
        return {}

    stock_df["_stock_qty"] = pd.to_numeric(stock_df["stock_qty"], errors="coerce").fillna(0)
    stock_df["_reserved_qty"] = pd.to_numeric(stock_df["reserved_qty"], errors="coerce").fillna(0)
    stock_df["_low_stock"] = pd.to_numeric(stock_df["low_stock_threshold"], errors="coerce")

    grouped = stock_df.groupby("_pid", dropna=True)
    stock_lookup: dict[int, dict] = {}

    for pid, group in grouped:
        pid_int = _to_int(pid)
        if pid_int is None:
            continue

        stock_total = int(group["_stock_qty"].sum())
        reserved_total = int(group["_reserved_qty"].sum())
        availability_units = stock_total - reserved_total

        low_stock_series = group["_low_stock"].dropna()
        low_stock_threshold = int(low_stock_series.min()) if not low_stock_series.empty else None

        warehouses = sorted({str(v).strip() for v in group["warehouse_location"] if str(v).strip()})

        stock_lookup[pid_int] = {
            "stock_qty": stock_total,
            "reserved_qty": reserved_total,
            "availability_units": availability_units,
            "is_available": availability_units > 0,
            "low_stock_threshold": low_stock_threshold,
            "is_low_stock": (
                low_stock_threshold is not None and availability_units <= low_stock_threshold
            ),
            "warehouse_locations": warehouses,
        }

    return stock_lookup


def _promo_is_current(row: pd.Series, start_col: str | None, end_col: str | None, today: date) -> bool:
    start_date = None
    end_date = None

    if start_col:
        start = pd.to_datetime(row[start_col], errors="coerce")
        if pd.notna(start):
            start_date = start.date()

    if end_col:
        end = pd.to_datetime(row[end_col], errors="coerce")
        if pd.notna(end):
            end_date = end.date()

    if start_date and today < start_date:
        return False
    if end_date and today > end_date:
        return False
    return True


def _promotion_status(row: pd.Series, today: date) -> str:
    """
    Returns one of: active, inactive, upcoming, expired.
    """
    if not _to_bool(row["active"]):
        return "inactive"

    start = pd.to_datetime(row["start_date"], errors="coerce")
    end = pd.to_datetime(row["end_date"], errors="coerce")

    if pd.notna(start) and today < start.date():
        return "upcoming"
    if pd.notna(end) and today > end.date():
        return "expired"
    return "active"


def _build_promotion_lookup() -> tuple[dict[int, list[dict]], dict[int, list[dict]]]:
    promo_data = load_s3_data("promotions.csv")
    if not isinstance(promo_data, pd.DataFrame) or promo_data.empty:
        return {}, {}

    promo_df = promo_data.copy()
    required_columns = {
        "promotion_id",
        "promotion_name",
        "description",
        "discount_type",
        "discount_value",
        "min_purchase_amount",
        "start_date",
        "end_date",
        "active",
        "applicable_category_ids",
        "applicable_product_ids",
    }
    if not required_columns.issubset(promo_df.columns):
        return {}, {}

    product_promos: dict[int, list[dict]] = {}
    category_promos: dict[int, list[dict]] = {}
    today = date.today()

    for _, row in promo_df.iterrows():
        status = _promotion_status(row, today)
        payload = {
            "promotion_id": row["promotion_id"],
            "discount_type": row["discount_type"],
            "discount_value": row["discount_value"],
            "min_purchase_amount": row["min_purchase_amount"],
            "start_date": row["start_date"],
            "end_date": row["end_date"],
            "active": _to_bool(row["active"]),
            "status": status,
            "message_1": row["promotion_name"],
            "message_2": row["description"],
        }

        for product_id in _split_pipe_ids(row["applicable_product_ids"]):
            promo_product = dict(payload)
            promo_product["applies_by"] = "product"
            product_promos.setdefault(product_id, []).append(promo_product)

        for category_id in _split_pipe_ids(row["applicable_category_ids"]):
            promo_category = dict(payload)
            promo_category["applies_by"] = "category"
            category_promos.setdefault(category_id, []).append(promo_category)

    return product_promos, category_promos


def _apply_filters(
    df: pd.DataFrame,
    filters: dict,
) -> pd.DataFrame:
    output = df

    category_val = filters.get("category")
    if category_val is not None:
        category_id = _to_int(category_val)
        if category_id is not None:
            output = output[output["category_id"] == category_id]

    brand_val = filters.get("brand")
    if brand_val is not None:
        brand_id = _to_int(brand_val)
        if brand_id is not None:
            output = output[output["brand_id"] == brand_id]

    if filters.get("price_min") is not None:
        price_min = float(filters["price_min"])
        prices = pd.to_numeric(output["price"], errors="coerce")
        output = output[prices >= price_min]

    if filters.get("price_max") is not None:
        price_max = float(filters["price_max"])
        prices = pd.to_numeric(output["price"], errors="coerce")
        output = output[prices <= price_max]

    if filters.get("available") is not None:
        expected_available = _to_bool(filters["available"])
        available_series = output["active"].map(_to_bool)
        output = output[available_series == expected_available]

    return output


def _build_results(
    df: pd.DataFrame,
    stock_lookup: dict[int, dict],
    product_promos: dict[int, list[dict]],
    category_promos: dict[int, list[dict]],
) -> list[dict]:
    results: list[dict] = []
    for _, row in df.iterrows():
        product_id = _to_int(row["product_id"])
        category_id = _to_int(row["category_id"])

        item = {
            "product_id": row["product_id"],
            "category_id": row["category_id"],
            "brand_id": row["brand_id"],
            "name": row["name"],
            "description": row["description"],
            "specifications": row["specifications"],
            "warranty_months": row["warranty_months"],
            "return_days": row["return_days"],
            "is_final_sale": row["is_final_sale"],
            "free_shipping": row["free_shipping"],
            "shipping_days": row["shipping_days"],
            "price": row["price"],
            "active": row["active"],
            "weight_kg": row["weight_kg"],
            "requires_installation": row["requires_installation"],
            "installation_notes": row["installation_notes"],
            "match_type": row["_match_type"],
            "score": round(float(row["_score"]), 2),
        }

        if product_id is not None and product_id in stock_lookup:
            item.update(stock_lookup[product_id])
        else:
            item["stock_qty"] = 0
            item["reserved_qty"] = 0
            item["availability_units"] = 0
            item["is_available"] = False
            item["low_stock_threshold"] = None
            item["is_low_stock"] = False
            item["warehouse_locations"] = []

        promotions_by_id: dict[str, dict] = {}

        def _merge_promo(promo: dict) -> None:
            promo_key = str(promo.get("promotion_id"))
            existing = promotions_by_id.get(promo_key)
            if existing is None:
                normalized = dict(promo)
                applies_by = normalized.get("applies_by")
                normalized["applies_by"] = [applies_by] if applies_by else []
                promotions_by_id[promo_key] = normalized
                return

            applies_by = promo.get("applies_by")
            if applies_by and applies_by not in existing["applies_by"]:
                existing["applies_by"].append(applies_by)

        if product_id is not None:
            for promo in product_promos.get(product_id, []):
                _merge_promo(promo)

        if category_id is not None:
            for promo in category_promos.get(category_id, []):
                _merge_promo(promo)

        all_promotions = list(promotions_by_id.values())
        active_promotions = [p for p in all_promotions if p.get("status") == "active"]
        inactive_promotions = [p for p in all_promotions if p.get("status") != "active"]

        item["has_promotion"] = len(active_promotions) > 0
        item["promotions"] = active_promotions
        item["promotions_other"] = inactive_promotions
        item["promotion_summary"] = {
            "active": len(active_promotions),
            "non_active": len(inactive_promotions),
            "total": len(all_promotions),
        }

        results.append(item)

    return results


def _build_search_results_min(
    df: pd.DataFrame,
    stock_lookup: dict[int, dict],
    product_promos: dict[int, list[dict]],
    category_promos: dict[int, list[dict]],
) -> list[dict]:
    """
    Lightweight payload for product discovery.
    Includes minimal product identity + availability + active promo messages.
    """
    results: list[dict] = []

    for _, row in df.iterrows():
        product_id = _to_int(row["product_id"])
        category_id = _to_int(row["category_id"])

        item = {
            "product_id": row["product_id"],
            "name": row["name"],
            "description": row["description"],
            "category_id": row["category_id"],
            "brand_id": row["brand_id"],
            "active": row["active"],
            "match_type": row["_match_type"],
            "score": round(float(row["_score"]), 2),
        }

        if product_id is not None and product_id in stock_lookup:
            stock = stock_lookup[product_id]
            item["stock_qty"] = stock["stock_qty"]
            item["reserved_qty"] = stock["reserved_qty"]
            item["availability_units"] = stock["availability_units"]
            item["is_available"] = stock["is_available"]
            item["is_low_stock"] = stock["is_low_stock"]
        else:
            item["stock_qty"] = 0
            item["reserved_qty"] = 0
            item["availability_units"] = 0
            item["is_available"] = False
            item["is_low_stock"] = False

        promotions_by_id: dict[str, dict] = {}

        def _merge_promo(promo: dict) -> None:
            promo_key = str(promo.get("promotion_id"))
            existing = promotions_by_id.get(promo_key)
            if existing is None:
                normalized = dict(promo)
                applies_by = normalized.get("applies_by")
                normalized["applies_by"] = [applies_by] if applies_by else []
                promotions_by_id[promo_key] = normalized
                return

            applies_by = promo.get("applies_by")
            if applies_by and applies_by not in existing["applies_by"]:
                existing["applies_by"].append(applies_by)

        if product_id is not None:
            for promo in product_promos.get(product_id, []):
                _merge_promo(promo)

        if category_id is not None:
            for promo in category_promos.get(category_id, []):
                _merge_promo(promo)

        active_promos = [p for p in promotions_by_id.values() if p.get("status") == "active"]
        item["has_promotion"] = len(active_promos) > 0
        item["promotions"] = [
            {
                "promotion_id": p.get("promotion_id"),
                "message_1": p.get("message_1"),
                "message_2": p.get("message_2"),
                "discount_type": p.get("discount_type"),
                "discount_value": p.get("discount_value"),
            }
            for p in active_promos
        ]

        results.append(item)

    return results

@tool
def search_product(
    query: str | int | None = None,
    product_id: int | None = None,
    filters: dict | None = None,
    top_k: int = 5
):
    """
    Search products for discovery and selection in 3 levels: exact, partial, fuzzy.

    Args:
        query: Product name text (or numeric-like string/int) used for search.
            If product_id is not provided, query may also be interpreted as product id.
        product_id: Optional direct product identifier for exact lookup.
        filters: Optional constraints dict. Supported keys:
            - category: category_id
            - brand: brand_id
            - price_min: minimum price
            - price_max: maximum price
            - available: boolean over products.active
        top_k: Maximum number of results to return (clamped to 1..5).

    Returns:
        list[dict]: Lightweight product payload ordered by relevance, each item containing:
            - identity: product_id, name, description, category_id, brand_id, active
            - matching: match_type, score
            - stock summary: stock_qty, reserved_qty, availability_units, is_available, is_low_stock
            - promotion summary: has_promotion, promotions (active only)
    """
    input_data = {
        "query": query,
        "product_id": product_id,
        "filters": filters,
        "top_k": top_k,
    }

    def _trace_return(output: list[dict]) -> list[dict]:
        add_tool_trace("search_product", input_data, {"results": output})
        return output

    all_products = load_s3_data("products.csv")

    if not isinstance(all_products, pd.DataFrame):
        return _trace_return([])

    if not isinstance(filters, dict):
        filters = {}

    k = max(1, min(int(top_k), 5))
    df = all_products.copy()

    required_product_columns = {
        "product_id",
        "category_id",
        "brand_id",
        "name",
        "description",
        "specifications",
        "warranty_months",
        "return_days",
        "is_final_sale",
        "free_shipping",
        "shipping_days",
        "price",
        "active",
        "weight_kg",
        "requires_installation",
        "installation_notes",
    }
    if not required_product_columns.issubset(df.columns):
        return _trace_return([])

    df["product_id"] = pd.to_numeric(df["product_id"], errors="coerce")
    df["category_id"] = pd.to_numeric(df["category_id"], errors="coerce")
    df["brand_id"] = pd.to_numeric(df["brand_id"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    stock_lookup = _build_stock_lookup()
    product_promos, category_promos = _build_promotion_lookup()

    df["_name_norm"] = df["name"].map(_normalize_text)

    df = _apply_filters(
        df=df,
        filters=filters,
    )

    if df.empty:
        return _trace_return([])

    target_pid = _to_int(product_id)
    if target_pid is None:
        target_pid = _to_int(query)

    # Direct id search first if possible.
    if target_pid is not None:
        found = df[df["product_id"] == target_pid].copy().head(k)
        if found.empty:
            return _trace_return([])
        found["_match_type"] = "id"
        found["_score"] = 100.0
        return _trace_return(_build_search_results_min(
            found,
            stock_lookup,
            product_promos,
            category_promos,
        ))

    query_norm = _normalize_text(query)
    if not query_norm:
        return _trace_return([])

    # Level 1: Exact match
    exact = df[df["_name_norm"] == query_norm].copy()
    if not exact.empty:
        exact["_match_type"] = "exact"
        exact["_score"] = 100.0
        top = exact.sort_values(by=["_score", "name"], ascending=[False, True]).head(k)
        return _trace_return(_build_search_results_min(
            top,
            stock_lookup,
            product_promos,
            category_promos,
        ))

    # Level 2: Partial match
    partial = df[df["_name_norm"].str.contains(query_norm, na=False)].copy()
    if not partial.empty:
        partial["_match_type"] = "partial"
        partial["_score"] = partial["_name_norm"].apply(lambda n: _score_partial(query_norm, n))
        top = partial.sort_values(by=["_score", "name"], ascending=[False, True]).head(k)
        return _trace_return(_build_search_results_min(
            top,
            stock_lookup,
            product_promos,
            category_promos,
        ))

    # Level 3: Fuzzy match
    fuzzy = df.copy()
    fuzzy["_score"] = fuzzy["_name_norm"].apply(lambda n: _score_fuzzy(query_norm, n))
    fuzzy = fuzzy[fuzzy["_score"] >= 60.0].copy()
    if fuzzy.empty:
        return _trace_return([])

    fuzzy["_match_type"] = "fuzzy"
    top = fuzzy.sort_values(by=["_score", "name"], ascending=[False, True]).head(k)
    return _trace_return(_build_search_results_min(
        top,
        stock_lookup,
        product_promos,
        category_promos,
    ))


@tool
def search_products(
    query: str | int | None = None,
    product_id: int | None = None,
    filters: dict | None = None,
    top_k: int = 5,
):
    """
    Compatibility alias for search_product.

    Args:
        query: Same as search_product.
        product_id: Same as search_product.
        filters: Same as search_product.
        top_k: Same as search_product.

    Returns:
        list[dict]: Same output as search_product.
    """
    return search_product(
        query=query,
        product_id=product_id,
        filters=filters,
        top_k=top_k,
    )


if __name__ == "__main__":
    import time
    s = time.perf_counter()
    results = search_product("Samsung")
    for r in results:
      print(f"{r}\n")
    f = time.perf_counter()
    print(f"Time: {f - s}")
    s = time.perf_counter()
    results = search_product("Samsung")
    for r in results:
      print(f"{r}\n")
    f = time.perf_counter()
    print(f"Time (warm): {f - s}")
    

