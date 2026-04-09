from datetime import date
from typing import Any

import pandas as pd
from strands import tool

from core.data_store import load_s3_data
import core.session_context as session_context
from core.tools.inventory.search_products import _to_bool, _to_int


def _clean_value(value: Any) -> Any:
	if pd.isna(value):
		return None
	return value


def _base_error(customer_id: int | None, order_id: int | None, reason: str, authenticated: bool) -> dict:
	return {
		"authenticated": authenticated,
		"customer_id": customer_id,
		"order_id": order_id,
		"item_id": None,
		"product_id": None,
		"eligible_for_return": False,
		"return_deadline": None,
		"days_remaining": None,
		"item_status": None,
		"is_final_sale": None,
		"reason": reason,
	}


def _find_customer_order(customer_id: int, order_id: int) -> bool:
	orders = load_s3_data("orders.csv")
	if not isinstance(orders, pd.DataFrame):
		return False

	if not {"order_id", "customer_id"}.issubset(orders.columns):
		return False

	orders_df = orders.copy()
	orders_df["_order_id"] = orders_df["order_id"].map(_to_int)
	orders_df["_customer_id"] = orders_df["customer_id"].map(_to_int)

	match = orders_df[
		(orders_df["_order_id"] == order_id) & (orders_df["_customer_id"] == customer_id)
	]
	return not match.empty


def _resolve_item(
	items_df: pd.DataFrame,
	order_id: int,
	item_id: int | None,
	product_id: int | None,
) -> pd.DataFrame:
	filtered = items_df.copy()
	filtered["_order_id"] = filtered["order_id"].map(_to_int)
	filtered = filtered[filtered["_order_id"] == order_id].copy()

	if filtered.empty:
		return filtered

	if item_id is not None:
		filtered["_item_id"] = filtered["item_id"].map(_to_int)
		filtered = filtered[filtered["_item_id"] == item_id].copy()

	if product_id is not None:
		filtered["_product_id"] = filtered["product_id"].map(_to_int)
		filtered = filtered[filtered["_product_id"] == product_id].copy()

	if filtered.empty:
		return filtered

	if "item_id" in filtered.columns:
		filtered["_item_id_sort"] = filtered["item_id"].map(_to_int)
		filtered = filtered.sort_values(by="_item_id_sort", ascending=True, na_position="last")

	return filtered.head(1)


@tool
def get_item_return_eligibility(
	order_id: int,
	item_id: int | None = None,
	product_id: int | None = None,
) -> dict:
	"""
	Evaluate whether one order item is eligible for return.

	Use this tool for return-window and return-policy checks on a specific customer order item.
	Requires an authenticated session and an order that belongs to that customer.

	Args:
		order_id: Required order identifier.
		item_id: Optional order line identifier.
		product_id: Optional product identifier to disambiguate the line.
			When both item_id and product_id are provided, both filters are applied.

	Returns:
		dict: Stable response payload with keys:
			- authenticated, customer_id, order_id, item_id, product_id
			- eligible_for_return: bool
			- return_deadline: str | None (YYYY-MM-DD)
			- days_remaining: int | None
			- item_status: str | None
			- is_final_sale: bool | None
			- reason: str | None
			When validation fails, eligibility is false and reason explains the cause.
	"""
	input_data = {"order_id": order_id, "item_id": item_id, "product_id": product_id}

	session_customer = session_context.get_session_customer()
	if session_context._SESSION_CUSTOMER is None or session_customer is None:
		output = _base_error(None, None, "customer not authenticated", False)
		session_context.add_tool_trace("get_item_return_eligibility", input_data, output)
		return output

	customer_id = _to_int(session_customer.get("customer_id"))
	target_order_id = _to_int(order_id)
	target_item_id = _to_int(item_id)
	target_product_id = _to_int(product_id)

	if customer_id is None:
		output = _base_error(None, target_order_id, "invalid session customer_id", False)
		session_context.add_tool_trace("get_item_return_eligibility", input_data, output)
		return output

	if target_order_id is None:
		output = _base_error(customer_id, None, "invalid order_id", True)
		session_context.add_tool_trace("get_item_return_eligibility", input_data, output)
		return output

	if not _find_customer_order(customer_id, target_order_id):
		output = _base_error(
			customer_id,
			target_order_id,
			"order not found for authenticated customer",
			True,
		)
		session_context.add_tool_trace("get_item_return_eligibility", input_data, output)
		return output

	order_items = load_s3_data("order_items.csv")
	if not isinstance(order_items, pd.DataFrame):
		output = _base_error(customer_id, target_order_id, "order_items table unavailable", True)
		session_context.add_tool_trace("get_item_return_eligibility", input_data, output)
		return output

	if not {"order_id", "item_id", "product_id", "return_deadline", "item_status"}.issubset(order_items.columns):
		output = _base_error(customer_id, target_order_id, "order_items schema mismatch", True)
		session_context.add_tool_trace("get_item_return_eligibility", input_data, output)
		return output

	target_item = _resolve_item(order_items, target_order_id, target_item_id, target_product_id)
	if target_item.empty:
		output = _base_error(customer_id, target_order_id, "item not found in order", True)
		session_context.add_tool_trace("get_item_return_eligibility", input_data, output)
		return output

	row = target_item.iloc[0]
	resolved_item_id = _to_int(row.get("item_id"))
	resolved_product_id = _to_int(row.get("product_id"))
	item_status = str(_clean_value(row.get("item_status")) or "").strip().lower()

	deadline_ts = pd.to_datetime(row.get("return_deadline"), errors="coerce")
	deadline_date = deadline_ts.date() if pd.notna(deadline_ts) else None
	today = date.today()
	days_remaining = (deadline_date - today).days if deadline_date else None

	is_final_sale = None
	if resolved_product_id is not None:
		products = load_s3_data("products.csv")
		if isinstance(products, pd.DataFrame) and {"product_id", "is_final_sale"}.issubset(products.columns):
			products_df = products.copy()
			products_df["_product_id"] = products_df["product_id"].map(_to_int)
			product_match = products_df[products_df["_product_id"] == resolved_product_id].head(1)
			if not product_match.empty:
				is_final_sale = _to_bool(product_match.iloc[0].get("is_final_sale"))

	reason = None
	eligible_for_return = True
	closed_statuses = {"returned", "refunded", "replaced"}

	if item_status in closed_statuses:
		eligible_for_return = False
		reason = "item is not returnable by status"
	elif is_final_sale is True:
		eligible_for_return = False
		reason = "item is final sale"
	elif deadline_date is None:
		eligible_for_return = False
		reason = "return deadline unavailable"
	elif days_remaining is not None and days_remaining < 0:
		eligible_for_return = False
		reason = "return deadline passed"

	output = {
		"authenticated": True,
		"customer_id": customer_id,
		"order_id": target_order_id,
		"item_id": resolved_item_id,
		"product_id": resolved_product_id,
		"eligible_for_return": eligible_for_return,
		"return_deadline": str(deadline_date) if deadline_date else None,
		"days_remaining": days_remaining,
		"item_status": item_status or None,
		"is_final_sale": is_final_sale,
		"reason": reason,
	}
	session_context.add_tool_trace("get_item_return_eligibility", input_data, output)
	return output


if __name__ == "__main__":
	import time

	session_context.set_session_customer(1234, "test_user")

	s = time.perf_counter()
	result = get_item_return_eligibility(order_id=74, item_id=221)
	print(result)
	f = time.perf_counter()
	print(f"Time: {f - s}")

	s = time.perf_counter()
	result = get_item_return_eligibility(order_id=74, item_id=221)
	print(result)
	f = time.perf_counter()
	print(f"Time (warm): {f - s}")
