from core.data_store import load_s3_data
from core.session_context import add_tool_trace
from core.tools.inventory.consolidate_product_details import consolidate_product_details
from core.tools.inventory.search_products import (
	_merge_promotions_for_product,
	_build_promotion_lookup,
	_build_stock_lookup,
	_to_bool,
	_to_float,
	_to_int,
)
from strands import tool
import pandas as pd
from typing import Optional


@tool
def get_product_details(product_id: int, product_ids: Optional[list] = None) -> dict:
	"""
	Return full catalog details for a product, with consolidated stock across all variants.

	Use this only after a product has already been identified (usually via search_product).
	This tool is intended for deep detail requests such as specifications, price,
	warranty/return flags, shipping_days, and full promotion context.

	Important usage rules for the caller:
	- product_id: the representative product ID (e.g. first element of product_ids from search_product).
	- product_ids: ALWAYS pass the full list of product_ids from search_product results to get consolidated stock.
	- Do not call with missing/guessed product_id.
	- Do not call for broad listing queries; use search_product instead.

	Args:
		product_id: Representative product identifier for base detail lookup.
		product_ids: Full list of variant product IDs (from search_product) for consolidated stock aggregation.

	Returns:
		dict: Full product payload with consolidated stock, specs, warranty, shipping, and active promotions.
			Returns {} when product_id is invalid, missing, or source data is unavailable.
	"""
	input_data = {"product_id": product_id, "product_ids": product_ids}

	def _trace_return(output: dict) -> dict:
		add_tool_trace("get_product_details", input_data, output)
		return output

	pid = _to_int(product_id)
	if pid is None:
		return _trace_return({})

	all_products = load_s3_data("products.csv")
	if not isinstance(all_products, pd.DataFrame):
		return _trace_return({})

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
		return _trace_return({})

	df["product_id"] = pd.to_numeric(df["product_id"], errors="coerce")
	all_pids = [_to_int(p) for p in (product_ids or [])] if product_ids else []
	all_pids = [p for p in all_pids if p is not None]
	if not all_pids:
		all_pids = [pid]
	elif pid not in all_pids:
		all_pids.insert(0, pid)

	group_df = df[df["product_id"].isin(all_pids)].copy()
	if group_df.empty:
		return _trace_return({})

	stock_lookup = _build_stock_lookup()
	product_promos, category_promos = _build_promotion_lookup()

	group_records = group_df.to_dict(orient="records")
	canonical = consolidate_product_details(group_records) if len(group_records) > 1 else None
	if canonical is None:
		representative_row = group_df[group_df["product_id"] == pid].copy().head(1)
		if representative_row.empty:
			representative_row = group_df.head(1)
		representative = representative_row.iloc[0].to_dict()
		canonical = {
			"representative_product_id": _to_int(representative.get("product_id")) or pid,
			"name": representative.get("name"),
			"description": representative.get("description"),
			"specifications": representative.get("specifications"),
			"category_id": _to_int(representative.get("category_id")),
			"brand_id": _to_int(representative.get("brand_id")),
			"warranty_months": _to_int(representative.get("warranty_months")),
			"return_days": _to_int(representative.get("return_days")),
			"is_final_sale": _to_bool(representative.get("is_final_sale")),
			"free_shipping": _to_bool(representative.get("free_shipping")),
			"shipping_days": _to_int(representative.get("shipping_days")),
			"price": _to_float(representative.get("price")),
			"price_min": None,
			"price_max": None,
			"active": _to_bool(representative.get("active")),
			"weight_kg": _to_float(representative.get("weight_kg")),
			"requires_installation": _to_bool(representative.get("requires_installation")),
			"installation_notes": representative.get("installation_notes"),
		}

	representative_product_id = canonical.get("representative_product_id") or pid
	category_id = canonical.get("category_id")

	stock_total = sum(stock_lookup.get(p, {}).get("stock_qty", 0) for p in all_pids)
	reserved_total = sum(stock_lookup.get(p, {}).get("reserved_qty", 0) for p in all_pids)
	availability_total = stock_total - reserved_total
	thresholds = [
		stock_lookup[p]["low_stock_threshold"]
		for p in all_pids
		if p in stock_lookup and stock_lookup[p].get("low_stock_threshold") is not None
	]
	low_stock_threshold = min(thresholds) if thresholds else None
	is_available = availability_total > 0
	is_low_stock = low_stock_threshold is not None and availability_total <= low_stock_threshold

	active_promotions = [
		promo
		for promo in _merge_promotions_for_product(all_pids, category_id, product_promos, category_promos)
		if promo.get("status") == "active"
	]

	result = {
		"product_id": representative_product_id,
		"product_ids": all_pids,
		"category_id": category_id,
		"brand_id": canonical.get("brand_id"),
		"name": canonical.get("name"),
		"description": canonical.get("description"),
		"specifications": canonical.get("specifications"),
		"warranty_months": canonical.get("warranty_months"),
		"return_days": canonical.get("return_days"),
		"is_final_sale": canonical.get("is_final_sale"),
		"free_shipping": canonical.get("free_shipping"),
		"shipping_days": canonical.get("shipping_days"),
		"active": canonical.get("active"),
		"weight_kg": canonical.get("weight_kg"),
		"requires_installation": canonical.get("requires_installation"),
		"installation_notes": canonical.get("installation_notes"),
		"match_type": "id",
		"score": 100.0,
		"merged_product_count": len(all_pids),
		"stock_qty": stock_total,
		"reserved_qty": reserved_total,
		"availability_units": availability_total,
		"is_available": is_available,
		"low_stock_threshold": low_stock_threshold,
		"is_low_stock": is_low_stock,
		"has_promotion": len(active_promotions) > 0,
		"promotions": active_promotions,
		"promotion_summary": {
			"active": len(active_promotions),
			"total": len(active_promotions),
		},
	}

	price = canonical.get("price")
	price_min = canonical.get("price_min")
	price_max = canonical.get("price_max")
	if price_min is not None and price_max is not None:
		if abs(price_max - price_min) < 1e-9:
			result["price"] = price_min
		else:
			result["price_min"] = price_min
			result["price_max"] = price_max
	elif price is not None:
		result["price"] = price
	elif price_min is not None:
		result["price"] = price_min

	return _trace_return(result)


if __name__ == "__main__":
    import time
    s = time.perf_counter()
    results = get_product_details(5056)
    print(results)
    f = time.perf_counter()
    print(f"Time: {f - s}")
    s = time.perf_counter()
    results = get_product_details(5056)
    print(results)
    f = time.perf_counter()
    print(f"Time (warm): {f - s}")
    #for r in results:
     #   print(f"{r}\n")

