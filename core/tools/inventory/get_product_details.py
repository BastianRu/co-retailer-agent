from core.data_store import load_s3_data
from core.session_context import add_tool_trace
from core.tools.inventory.search_products import (
	_build_promotion_lookup,
	_build_results,
	_build_stock_lookup,
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
	match = df[df["product_id"] == pid].copy().head(1)
	if match.empty:
		return _trace_return({})

	stock_lookup = _build_stock_lookup()
	product_promos, category_promos = _build_promotion_lookup()

	match["_match_type"] = "id"
	match["_score"] = 100.0
	details = _build_results(match, stock_lookup, product_promos, category_promos)
	if not details:
		return _trace_return({})

	result = details[0]

	# Aggregate consolidated stock across all variant IDs if provided
	all_pids = [_to_int(p) for p in (product_ids or [])] if product_ids else []
	all_pids = [p for p in all_pids if p is not None]
	if not all_pids:
		all_pids = [pid]

	if len(all_pids) > 1:
		stock_total = sum(stock_lookup.get(p, {}).get("stock_qty", 0) for p in all_pids)
		reserved_total = sum(stock_lookup.get(p, {}).get("reserved_qty", 0) for p in all_pids)
		availability_total = stock_total - reserved_total
		thresholds = [stock_lookup[p]["low_stock_threshold"] for p in all_pids if p in stock_lookup and stock_lookup[p].get("low_stock_threshold") is not None]
		low_stock_threshold = min(thresholds) if thresholds else None
		result["stock_qty"] = stock_total
		result["reserved_qty"] = reserved_total
		result["availability_units"] = availability_total
		result["is_available"] = availability_total > 0
		result["is_low_stock"] = low_stock_threshold is not None and availability_total <= low_stock_threshold
		result["low_stock_threshold"] = low_stock_threshold

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

