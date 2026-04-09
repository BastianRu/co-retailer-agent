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


@tool
def get_product_details(product_id: int) -> dict:
	"""
	Return full catalog details for one exact product_id.

	Use this only after a product has already been identified (usually via search_product).
	This tool is intended for deep detail requests such as specifications, price,
	warranty/return flags, shipping_days, and full promotion context.

	Important usage rules for the caller:
	- Do not call with missing/guessed product_id.
	- Do not call for broad listing queries; use search_product instead.
	- If the user only needs basic availability/listing, avoid this extra call.

	Args:
		product_id: Product identifier for exact lookup.

	Returns:
		dict: Full product payload including base product fields, stock summary,
			warehouse locations, promotion summary, and promotion details.
			Returns {} when product_id is invalid, missing, or source data is unavailable.
	"""
	input_data = {"product_id": product_id}

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
	return _trace_return(details[0] if details else {})


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

