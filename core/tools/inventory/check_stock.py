from core.data_store import load_s3_data
from core.session_context import add_tool_trace
from core.tools.inventory.search_products import (
	_build_stock_lookup,
	_normalize_text,
	_score_fuzzy,
	_score_partial,
	_to_int,
)
from strands import tool
import pandas as pd


def _base_stock_item(row: pd.Series, stock_lookup: dict[int, dict], match_type: str, score: float) -> dict:
	product_id = _to_int(row["product_id"])
	item = {
		"product_id": row["product_id"],
		"name": row["name"],
		"category_id": row["category_id"],
		"brand_id": row["brand_id"],
		"active": row["active"],
		"match_type": match_type,
		"score": round(float(score), 2),
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

	return item


def _results_from_df(
	df: pd.DataFrame,
	stock_lookup: dict[int, dict],
	match_type: str,
	score_col: str,
) -> list[dict]:
	results: list[dict] = []
	for _, row in df.iterrows():
		results.append(_base_stock_item(row, stock_lookup, match_type, row[score_col]))
	return results


@tool
def check_stock(
	query: str | int | None = None,
	product_id: int | None = None,
	top_k: int = 5,
):
	"""
	Stock-focused lookup for already identified product queries.

	Use this tool when the user explicitly asks about stock units, availability,
	or warehouse distribution for a product. It is not a general discovery tool.
	For broad catalog discovery (products/prices/promotions), prefer search_product first.

	Important usage rules for the caller:
	- Do not call this tool with route/control tokens (for example: "NO_DATA").
	- Do not call this tool for shipping/order status.
	- If search_product already provides enough availability data, avoid redundant calls.

	Args:
		query: Product text query. Can also be numeric-like text that may map to a product id.
		product_id: Optional exact product identifier. Takes priority over fuzzy name matching.
		top_k: Max number of results to return. Runtime clamps this value to 1..5.

	Returns:
		list[dict]: Ordered candidate products with stock-focused fields, including
			product identity, match metadata (match_type/score), stock totals,
			availability flags, low-stock flags, and warehouse_locations.
			Returns [] when there is no match or required source data is unavailable.
	"""
	input_data = {
		"query": query,
		"product_id": product_id,
		"top_k": top_k,
	}

	def _trace_return(output: list[dict]) -> list[dict]:
		add_tool_trace("check_stock", input_data, {"results": output})
		return output

	products = load_s3_data("products.csv")
	if not isinstance(products, pd.DataFrame):
		return _trace_return([])

	df = products.copy()
	required = {"product_id", "category_id", "brand_id", "name", "active"}
	if not required.issubset(df.columns):
		return _trace_return([])

	df["product_id"] = pd.to_numeric(df["product_id"], errors="coerce")
	df["category_id"] = pd.to_numeric(df["category_id"], errors="coerce")
	df["brand_id"] = pd.to_numeric(df["brand_id"], errors="coerce")
	df["_name_norm"] = df["name"].map(_normalize_text)

	stock_lookup = _build_stock_lookup()
	k = max(1, min(int(top_k), 5))

	target_pid = _to_int(product_id)
	if target_pid is None:
		target_pid = _to_int(query)

	# 1) Direct lookup by product_id when available.
	if target_pid is not None:
		found = df[df["product_id"] == target_pid].copy().head(k)
		if found.empty:
			return _trace_return([])
		found["_score"] = 100.0
		return _trace_return(_results_from_df(found, stock_lookup, "id", "_score"))

	# 2) Name-based lookup.
	query_norm = _normalize_text(query)
	if not query_norm:
		return _trace_return([])

	exact = df[df["_name_norm"] == query_norm].copy()
	if not exact.empty:
		exact["_score"] = 100.0
		top = exact.sort_values(by=["_score", "name"], ascending=[False, True]).head(k)
		return _trace_return(_results_from_df(top, stock_lookup, "exact", "_score"))

	partial = df[df["_name_norm"].str.contains(query_norm, na=False)].copy()
	if not partial.empty:
		partial["_score"] = partial["_name_norm"].apply(lambda name: _score_partial(query_norm, name))
		top = partial.sort_values(by=["_score", "name"], ascending=[False, True]).head(k)
		return _trace_return(_results_from_df(top, stock_lookup, "partial", "_score"))

	fuzzy = df.copy()
	fuzzy["_score"] = fuzzy["_name_norm"].apply(lambda name: _score_fuzzy(query_norm, name))
	fuzzy = fuzzy[fuzzy["_score"] >= 60.0].copy()
	if fuzzy.empty:
		return _trace_return([])

	top = fuzzy.sort_values(by=["_score", "name"], ascending=[False, True]).head(k)
	return _trace_return(_results_from_df(top, stock_lookup, "fuzzy", "_score"))
  

if __name__ == "__main__":
    results = check_stock("Samsung")
    print(results[0])
    #for r in results:
     #   print(f"{r}\n")
