from core.data_store import load_s3_data
from core.session_context import add_tool_trace, get_handle_dataset_inconsistencies
from strands import tool
from difflib import SequenceMatcher
from datetime import date
from collections import Counter
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


def _to_float(value: object) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None

    raw = str(value).strip()
    if not raw:
        return None

    try:
        return float(raw)
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


def _tokenize_normalized(text: str) -> list[str]:
    return [token for token in _normalize_text(text).split() if token]


_STRICT_MODEL_TOKENS = {
    "pro",
    "max",
    "plus",
    "ultra",
    "mini",
    "lite",
    "fe",
    "air",
}

_GENERIC_QUERY_TOKENS = {
    "celular",
    "celulares",
    "telefono",
    "telefonos",
    "smartphone",
    "smartphones",
    "movil",
    "moviles",
    "producto",
    "productos",
    "marca",
    "modelo",
}


def _is_fuzzy_compatible(query_norm: str, candidate_name_norm: str) -> bool:
    query_tokens = _tokenize_normalized(query_norm)
    candidate_tokens = set(_tokenize_normalized(candidate_name_norm))
    if not query_tokens or not candidate_tokens:
        return False

    numeric_tokens = {token for token in query_tokens if token.isdigit()}
    if numeric_tokens and not numeric_tokens.issubset(candidate_tokens):
        return False

    strict_model_tokens = {token for token in query_tokens if token in _STRICT_MODEL_TOKENS}
    if len(strict_model_tokens) >= 2 and not strict_model_tokens.issubset(candidate_tokens):
        return False

    informative_tokens = [
        token
        for token in query_tokens
        if token not in _GENERIC_QUERY_TOKENS
    ]
    if informative_tokens:
        exact_overlap = sum(1 for token in informative_tokens if token in candidate_tokens)
        min_required_overlap = max(1, len(informative_tokens) // 2)
        if exact_overlap < min_required_overlap:
            return False

    return True


def _text_similarity(a: object, b: object) -> float:
    left = _normalize_text(a)
    right = _normalize_text(b)
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    return str(value).strip() == ""


def _price_similarity(a: object, b: object) -> float:
    left = _to_float(a)
    right = _to_float(b)

    if left is None or right is None or left <= 0 or right <= 0:
        return 0.5

    return max(0.0, 1.0 - (abs(left - right) / max(left, right)))


def _value_similarity(a: object, b: object) -> float:
    if _is_missing(a) and _is_missing(b):
        return 1.0
    if _is_missing(a) or _is_missing(b):
        return 0.0

    left_num = _to_float(a)
    right_num = _to_float(b)
    if left_num is not None and right_num is not None:
        if left_num == right_num:
            return 1.0
        if left_num == 0 or right_num == 0:
            return 0.0
        return max(0.0, 1.0 - (abs(left_num - right_num) / max(abs(left_num), abs(right_num))))

    left_bool = str(a).strip().lower()
    right_bool = str(b).strip().lower()
    if left_bool in {"true", "false", "0", "1"} and right_bool in {"true", "false", "0", "1"}:
        return 1.0 if _to_bool(a) == _to_bool(b) else 0.0

    return 1.0 if _normalize_text(a) == _normalize_text(b) else _text_similarity(a, b)


_IDENTITY_TEXT_FIELDS = [
    "description",
    "specifications",
    "installation_notes",
]

_IDENTITY_CORE_EXACT_FIELDS = [
    "brand_id",
    "category_id",
    "requires_installation",
]

_IDENTITY_VARIANT_FIELDS = [
    "warranty_months",
    "return_days",
    "is_final_sale",
    "free_shipping",
    "shipping_days",
    "weight_kg",
]


def _average_similarity(values: list[float], default: float = 1.0) -> float:
    return (sum(values) / len(values)) if values else default


def _same_product_identity(left: dict, right: dict) -> bool:
    if _to_int(left.get("brand_id")) != _to_int(right.get("brand_id")):
        return False
    if _to_int(left.get("category_id")) != _to_int(right.get("category_id")):
        return False

    exact_similarities = [_value_similarity(left.get(field), right.get(field)) for field in _IDENTITY_CORE_EXACT_FIELDS]
    text_similarities = [_value_similarity(left.get(field), right.get(field)) for field in _IDENTITY_TEXT_FIELDS]
    variant_similarities = [_value_similarity(left.get(field), right.get(field)) for field in _IDENTITY_VARIANT_FIELDS]

    exact_avg = _average_similarity(exact_similarities)
    text_avg = _average_similarity(text_similarities)
    variant_avg = _average_similarity(variant_similarities)
    name_sim = _text_similarity(left.get("name"), right.get("name"))
    price_sim = _price_similarity(left.get("price"), right.get("price"))

    # When the catalog exposes the same model more than once with the exact same
    # display name and description, treat it as one logical product even if price
    # or stock snapshots differ.
    if (
        _normalize_text(left.get("name"))
        and _normalize_text(left.get("name")) == _normalize_text(right.get("name"))
        and _normalize_text(left.get("description")) == _normalize_text(right.get("description"))
        and exact_avg >= 0.99
    ):
        return True

    confidence = (
        (0.30 * name_sim)
        + (0.35 * exact_avg)
        + (0.25 * text_avg)
        + (0.05 * variant_avg)
        + (0.05 * price_sim)
    )

    return (
        confidence >= 0.88
        and exact_avg >= 0.99
        and text_avg >= 0.90
        and name_sim >= 0.72
        and price_sim >= 0.80
    )


def _tokenize_display_name(name: object) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", str(name or ""))


def _build_consensus_name(names: list[object]) -> str:
    tokenized_names = [_tokenize_display_name(name) for name in names if _tokenize_display_name(name)]
    if not tokenized_names:
        return str(names[0] if names else "").strip()

    total_names = len(tokenized_names)
    min_support = max(2, (total_names // 2) + 1)
    merged_tokens: list[str] = []
    max_len = max(len(tokens) for tokens in tokenized_names)

    for index in range(max_len):
        normalized_counter: Counter[str] = Counter()
        display_counter: dict[str, Counter[str]] = {}

        for tokens in tokenized_names:
            if index >= len(tokens):
                continue

            display_token = tokens[index]
            normalized_token = _normalize_text(display_token)
            if not normalized_token:
                continue

            normalized_counter[normalized_token] += 1
            display_counter.setdefault(normalized_token, Counter())[display_token] += 1

        if not normalized_counter:
            break

        normalized_token, support = max(
            normalized_counter.items(),
            key=lambda item: (item[1], -len(item[0]), item[0]),
        )
        if support < min_support:
            break

        display_token = max(
            display_counter[normalized_token].items(),
            key=lambda item: (item[1], -len(item[0]), item[0]),
        )[0]
        merged_tokens.append(display_token)

    if merged_tokens:
        return " ".join(merged_tokens)

    return str(names[0] if names else "").strip()


def _pick_representative_text(rows: list[dict], field: str) -> object:
    values = [row.get(field) for row in rows if not _is_missing(row.get(field))]
    if not values:
        return rows[0].get(field)

    counts = Counter(_normalize_text(value) for value in values)
    best_key = max(counts.items(), key=lambda item: (item[1], -len(item[0]), item[0]))[0]
    for value in values:
        if _normalize_text(value) == best_key:
            return value
    return values[0]


def _is_exact_duplicate_candidate(left: dict, right: dict) -> bool:
    return (
        bool(_normalize_text(left.get("name")))
        and _normalize_text(left.get("name")) == _normalize_text(right.get("name"))
        and _normalize_text(left.get("description")) == _normalize_text(right.get("description"))
        and _to_int(left.get("brand_id")) == _to_int(right.get("brand_id"))
        and _to_int(left.get("category_id")) == _to_int(right.get("category_id"))
    )


def _cluster_records(records: list[dict], enable_identity_merge: bool = True) -> list[list[dict]]:
    if len(records) < 2:
        return [[record] for record in records]

    parent = list(range(len(records)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left_index: int, right_index: int) -> None:
        left_root = find(left_index)
        right_root = find(right_index)
        if left_root != right_root:
            parent[right_root] = left_root

    for left_index in range(len(records)):
        for right_index in range(left_index + 1, len(records)):
            if _is_exact_duplicate_candidate(records[left_index], records[right_index]):
                union(left_index, right_index)
                continue

            if enable_identity_merge and _same_product_identity(records[left_index], records[right_index]):
                union(left_index, right_index)

    grouped: dict[int, list[dict]] = {}
    for index, record in enumerate(records):
        grouped.setdefault(find(index), []).append(record)

    return list(grouped.values())


def _merge_promotions_for_product(
    product_ids: list[int],
    category_id: int | None,
    product_promos: dict[int, list[dict]],
    category_promos: dict[int, list[dict]],
) -> list[dict]:
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

    for product_id in product_ids:
        for promo in product_promos.get(product_id, []):
            _merge_promo(promo)

    if category_id is not None:
        for promo in category_promos.get(category_id, []):
            _merge_promo(promo)

    return list(promotions_by_id.values())


def _build_discovery_item(
    row: dict,
    stock_lookup: dict[int, dict],
    product_promos: dict[int, list[dict]],
    category_promos: dict[int, list[dict]],
) -> dict:
    product_ids = row.get("product_ids") or []
    if not product_ids:
        parsed_product_id = _to_int(row.get("product_id"))
        product_ids = [parsed_product_id] if parsed_product_id is not None else []

    category_id = _to_int(row.get("category_id"))
    item = {
        "product_id": row.get("product_id"),
        "product_ids": product_ids,
        "name": row.get("name"),
        "description": row.get("description"),
        "category_id": row.get("category_id"),
        "brand_id": row.get("brand_id"),
        "active": row.get("active"),
        "match_type": row.get("_match_type"),
        "score": round(float(row.get("_score", 0.0)), 2),
        "merged_product_count": int(row.get("merged_product_count", max(len(product_ids), 1))),
    }

    price_min = _to_float(row.get("price_min", row.get("price")))
    price_max = _to_float(row.get("price_max", row.get("price")))
    if price_min is not None and price_max is not None:
        if abs(price_max - price_min) < 1e-9:
            item["price"] = price_min
        else:
            item["price_min"] = price_min
            item["price_max"] = price_max
    elif price_min is not None:
        item["price"] = price_min
    else:
        item["price"] = row.get("price")

    aggregated_stock_qty = 0
    aggregated_reserved_qty = 0
    aggregated_availability = 0
    is_available = False
    low_stock_thresholds: list[int] = []

    for product_id in product_ids:
        stock = stock_lookup.get(product_id)
        if not stock:
            continue
        aggregated_stock_qty += int(stock["stock_qty"])
        aggregated_reserved_qty += int(stock["reserved_qty"])
        aggregated_availability += int(stock["availability_units"])
        is_available = is_available or bool(stock["is_available"])
        low_stock_threshold = stock.get("low_stock_threshold")
        if low_stock_threshold is not None:
            low_stock_thresholds.append(int(low_stock_threshold))

    is_low_stock = False
    if low_stock_thresholds:
        is_low_stock = aggregated_availability <= sum(low_stock_thresholds)

    item["stock_qty"] = aggregated_stock_qty
    item["reserved_qty"] = aggregated_reserved_qty
    item["availability_units"] = aggregated_availability
    item["is_available"] = is_available
    item["is_low_stock"] = is_low_stock

    active_promos = [
        promo
        for promo in _merge_promotions_for_product(product_ids, category_id, product_promos, category_promos)
        if promo.get("status") == "active"
    ]
    item["has_promotion"] = len(active_promos) > 0
    item["promotions"] = [
        {
            "promotion_id": promo.get("promotion_id"),
            "message_1": promo.get("message_1"),
            "message_2": promo.get("message_2"),
            "discount_type": promo.get("discount_type"),
            "discount_value": promo.get("discount_value"),
        }
        for promo in active_promos
    ]

    return item


def _consolidate_product_candidates(df: pd.DataFrame, enable_identity_merge: bool = True) -> list[dict]:
    records = df.to_dict(orient="records")
    consolidated: list[dict] = []

    for group in _cluster_records(records, enable_identity_merge=enable_identity_merge):
        if len(group) == 1:
            row = dict(group[0])
            parsed_product_id = _to_int(row.get("product_id"))
            row["product_ids"] = [parsed_product_id] if parsed_product_id is not None else []
            row["merged_product_count"] = 1
            consolidated.append(row)
            continue

        ordered_group = sorted(group, key=lambda row: (-float(row.get("_score", 0.0)), str(row.get("name", ""))))
        representative = dict(ordered_group[0])
        product_ids = [parsed for parsed in (_to_int(row.get("product_id")) for row in ordered_group) if parsed is not None]
        numeric_prices = [
            price
            for price in (_to_float(row.get("price")) for row in ordered_group)
            if price is not None
        ]

        representative["product_ids"] = product_ids
        representative["merged_product_count"] = len(ordered_group)
        representative["name"] = _build_consensus_name([row.get("name") for row in ordered_group])
        representative["description"] = _pick_representative_text(ordered_group, "description")
        representative["_score"] = max(float(row.get("_score", 0.0)) for row in ordered_group)

        if numeric_prices:
            representative["price_min"] = min(numeric_prices)
            representative["price_max"] = max(numeric_prices)
            representative["price"] = representative["price_min"]

        consolidated.append(representative)

    return consolidated


def _build_discovery_results_from_rows(
    rows: list[dict],
    stock_lookup: dict[int, dict],
    product_promos: dict[int, list[dict]],
    category_promos: dict[int, list[dict]],
) -> list[dict]:
    return [
        _build_discovery_item(row, stock_lookup, product_promos, category_promos)
        for row in rows
    ]


def _dedupe_adjacent_name_tokens(name: object) -> str:
    tokens = str(name or "").strip().split()
    deduped: list[str] = []
    for token in tokens:
        if deduped and _normalize_text(token) == _normalize_text(deduped[-1]):
            continue
        deduped.append(token)
    return " ".join(deduped).strip()


def _is_supported_canonical_name(canonical_name: object, group: list[dict]) -> bool:
    normalized_canonical = _normalize_text(canonical_name)
    if not normalized_canonical:
        return False

    candidate_names = [str(row.get("name") or "").strip() for row in group if str(row.get("name") or "").strip()]
    if normalized_canonical in {_normalize_text(name) for name in candidate_names}:
        return True

    return any(
        normalized_canonical == _normalize_text(_dedupe_adjacent_name_tokens(name))
        for name in candidate_names
    )


def _merge_group_records(
    group: list[dict],
    representative_product_id: int | None = None,
    canonical_name: str | None = None,
) -> dict:
    ordered_group = sorted(group, key=lambda row: (-float(row.get("_score", 0.0)), str(row.get("name", ""))))

    representative = None
    if representative_product_id is not None:
        for row in ordered_group:
            if _to_int(row.get("product_id")) == representative_product_id:
                representative = dict(row)
                break

    if representative is None:
        representative = dict(ordered_group[0])

    product_ids = [parsed for parsed in (_to_int(row.get("product_id")) for row in ordered_group) if parsed is not None]
    numeric_prices = [
        price
        for price in (_to_float(row.get("price")) for row in ordered_group)
        if price is not None
    ]

    representative["product_ids"] = product_ids
    representative["merged_product_count"] = len(ordered_group)
    representative["description"] = _pick_representative_text(ordered_group, "description")
    representative["specifications"] = _pick_representative_text(ordered_group, "specifications")
    representative["installation_notes"] = _pick_representative_text(ordered_group, "installation_notes")
    representative["_score"] = max(float(row.get("_score", 0.0)) for row in ordered_group)

    if canonical_name and _is_supported_canonical_name(canonical_name, ordered_group):
        representative["name"] = canonical_name.strip()
    elif len(ordered_group) > 1:
        representative["name"] = _build_consensus_name([row.get("name") for row in ordered_group])

    if numeric_prices:
        representative["price_min"] = min(numeric_prices)
        representative["price_max"] = max(numeric_prices)
        representative["price"] = representative["price_min"]

    return representative


def _consolidate_candidate_pool(query_norm: str, df: pd.DataFrame) -> list[dict]:
    records = df.to_dict(orient="records")
    if len(records) < 2:
        return _consolidate_product_candidates(df, enable_identity_merge=False)

    try:
        from core.tools.inventory.consolidate_product_candidates import group_product_candidates

        group_specs = group_product_candidates(query=query_norm, candidates=records)
    except Exception:
        group_specs = None

    if not group_specs:
        return _consolidate_product_candidates(df, enable_identity_merge=True)

    candidates_by_id = {
        parsed_product_id: record
        for record in records
        if (parsed_product_id := _to_int(record.get("product_id"))) is not None
    }
    ordered_product_ids = [
        parsed_product_id
        for record in records
        if (parsed_product_id := _to_int(record.get("product_id"))) is not None
    ]

    assigned_product_ids: set[int] = set()
    consolidated_rows: list[dict] = []

    for group_spec in group_specs:
        group_product_ids: list[int] = []
        for product_id in group_spec.get("product_ids", []):
            parsed_product_id = _to_int(product_id)
            if parsed_product_id is None:
                continue
            if parsed_product_id not in candidates_by_id:
                continue
            if parsed_product_id in assigned_product_ids:
                continue
            group_product_ids.append(parsed_product_id)

        if not group_product_ids:
            continue

        group_set = set(group_product_ids)
        group_rows = [candidates_by_id[product_id] for product_id in ordered_product_ids if product_id in group_set]
        if not group_rows:
            continue

        assigned_product_ids.update(group_product_ids)
        consolidated_rows.append(
            _merge_group_records(
                group_rows,
                representative_product_id=_to_int(group_spec.get("representative_product_id")),
                canonical_name=group_spec.get("canonical_name"),
            )
        )

    for product_id in ordered_product_ids:
        if product_id in assigned_product_ids:
            continue
        consolidated_rows.append(_merge_group_records([candidates_by_id[product_id]]))

    return sorted(
        consolidated_rows,
        key=lambda row: (-float(row.get("_score", 0.0)), str(row.get("name", ""))),
    )


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
            item.pop("warehouse_locations", None)
        else:
            item["stock_qty"] = 0
            item["reserved_qty"] = 0
            item["availability_units"] = 0
            item["is_available"] = False
            item["low_stock_threshold"] = None
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

        all_promotions = list(promotions_by_id.values())
        active_promotions = [p for p in all_promotions if p.get("status") == "active"]
        inactive_promotions = [p for p in all_promotions if p.get("status") != "active"]

        item["has_promotion"] = len(active_promotions) > 0
        item["promotions"] = active_promotions
        item["promotion_summary"] = {
            "active": len(active_promotions),
            "total": len(all_promotions),
        }

        results.append(item)

    return results


def _build_search_results_min(
    df: pd.DataFrame,
    stock_lookup: dict[int, dict],
    product_promos: dict[int, list[dict]],
    category_promos: dict[int, list[dict]],
    enable_identity_merge: bool = True,
) -> list[dict]:
    """
    Lightweight payload for product discovery.
    Includes minimal product identity + availability + active promo messages.
    """
    consolidated_rows = _consolidate_product_candidates(df, enable_identity_merge=enable_identity_merge)
    return _build_discovery_results_from_rows(
        consolidated_rows,
        stock_lookup,
        product_promos,
        category_promos,
    )


def _build_search_results_raw(
    df: pd.DataFrame,
    stock_lookup: dict[int, dict],
    product_promos: dict[int, list[dict]],
    category_promos: dict[int, list[dict]],
) -> list[dict]:
    return _build_discovery_results_from_rows(
        df.to_dict(orient="records"),
        stock_lookup,
        product_promos,
        category_promos,
    )

@tool
def search_product(
    query: str | int | None = None,
    product_id: int | None = None,
    filters: dict | None = None,
    top_k: int = 5
):
    """
    Primary catalog discovery tool for public inventory questions.

    Use this first for queries about products, brands, category discovery, prices,
    promotions, and general availability. Prefer a single, concrete query string
    (for example: "iphone", "samsung", "laptop", "iphone 15 pro max").

    Important usage rules for the caller:
    - Do not call this tool with route labels or control words (for example: "NO_DATA").
    - Do not call with promotional words alone (for example: "oferta", "descuento").
      Instead, search the target product/category and inspect has_promotion/promotions.
    - If this tool already returns enough information, avoid calling extra tools.

    Args:
        query: Product text query. If product_id is not provided, numeric-like query
            values may be interpreted as product ids.
        product_id: Optional exact product identifier.
        filters: Optional constraints dict. Supported keys:
            - category: category_id
            - brand: brand_id
            - price_min: minimum price
            - price_max: maximum price
            - available: boolean over products.active
        top_k: Maximum number of results. Runtime clamps this value to 1..5.

    Returns:
        list[dict]: Relevance-ordered product summaries with identity fields,
            match metadata, stock summary, and promotion flags/details.
            When dataset inconsistency handling is enabled for the current
            session, likely duplicate rows for the same logical product are
            merged before returning results. Merged rows aggregate stock,
            expose `product_ids`, and when prices differ they return
            `price_min` / `price_max` instead of a single exact price.
            Key fields usually used by agents:
            - name, product_id, product_ids
            - availability_units, is_available, is_low_stock
            - price or price_min/price_max
            - has_promotion, promotions
            Returns [] when there are no matches or required source data is unavailable.
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
    candidate_pool_size = max(k * 3, 10)
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
        found = df[df["product_id"] == target_pid].copy().head(1)
        if found.empty:
            return _trace_return([])
        found["_match_type"] = "id"
        found["_score"] = 100.0
        return _trace_return(_build_search_results_raw(
            found,
            stock_lookup,
            product_promos,
            category_promos,
        ))

    query_norm = _normalize_text(query)
    if not query_norm:
        return _trace_return([])

    # Level 1: Partial match
    partial = df[df["_name_norm"].str.contains(query_norm, na=False)].copy()
    if not partial.empty:
        partial["_match_type"] = "partial"
        partial["_score"] = partial["_name_norm"].apply(lambda n: _score_partial(query_norm, n))
        top = partial.sort_values(by=["_score", "name"], ascending=[False, True]).head(candidate_pool_size)
        if get_handle_dataset_inconsistencies():
            rows = _consolidate_candidate_pool(query_norm, top)
            results = _build_discovery_results_from_rows(
                rows,
                stock_lookup,
                product_promos,
                category_promos,
            )
        else:
            results = _build_search_results_raw(
                top,
                stock_lookup,
                product_promos,
                category_promos,
            )
        return _trace_return(results[:k])

    # Level 2: Fuzzy match
    fuzzy = df.copy()
    fuzzy["_score"] = fuzzy["_name_norm"].apply(lambda n: _score_fuzzy(query_norm, n))
    fuzzy = fuzzy[fuzzy["_name_norm"].apply(lambda n: _is_fuzzy_compatible(query_norm, n))].copy()
    fuzzy = fuzzy[fuzzy["_score"] >= 60.0].copy()
    if fuzzy.empty:
        return _trace_return([])

    fuzzy["_match_type"] = "fuzzy"
    top = fuzzy.sort_values(by=["_score", "name"], ascending=[False, True]).head(candidate_pool_size)
    if get_handle_dataset_inconsistencies():
        rows = _consolidate_candidate_pool(query_norm, top)
        results = _build_discovery_results_from_rows(
            rows,
            stock_lookup,
            product_promos,
            category_promos,
        )
    else:
        results = _build_search_results_raw(
            top,
            stock_lookup,
            product_promos,
            category_promos,
        )
    return _trace_return(results[:k])


@tool
def search_products(
    query: str | int | None = None,
    product_id: int | None = None,
    filters: dict | None = None,
    top_k: int = 5,
):
    """
    Backward-compatible alias for search_product.

    Prefer calling search_product in new prompts. This alias exists only for
    compatibility with older prompt routes.

    Args:
        query: Same behavior as search_product.
        product_id: Same behavior as search_product.
        filters: Same behavior as search_product.
        top_k: Same behavior as search_product.

    Returns:
        list[dict]: Same payload contract as search_product.
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
    results = search_product("Iphone")
    for r in results:
      print(f"{r}\n")
    f = time.perf_counter()
    print(f"Time: {f - s}")
    
    

