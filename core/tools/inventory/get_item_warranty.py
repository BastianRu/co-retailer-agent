from datetime import date
from typing import Any

import pandas as pd
from strands import tool

from core.data_store import load_s3_data
import core.session_context as session_context
from core.tools.inventory.search_product import _to_int


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
		"in_warranty": False,
		"warranty_expires_at": None,
		"days_remaining": None,
		"item_status": None,
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
def get_item_warranty(
	order_id: int,
	item_id: int | None = None,
	product_id: int | None = None,
) -> dict:
	"""
	Check warranty coverage for one item within an authenticated customer's order.

	Args:
		order_id: Required order identifier.
		item_id: Optional item identifier to target a specific line item.
		product_id: Optional product identifier to target a line item by product.
			When both item_id and product_id are provided, both are applied as filters.

	Returns:
		dict: Fixed response structure:
			- authenticated, customer_id, order_id, item_id, product_id
			- in_warranty: bool
			- warranty_expires_at: str | None (YYYY-MM-DD)
			- days_remaining: int | None
			- item_status: str | None
			- reason: str | None
	"""
	input_data = {"order_id": order_id, "item_id": item_id, "product_id": product_id}

	session_customer = session_context.get_session_customer()
	if session_context._SESSION_CUSTOMER is None or session_customer is None:
		output = _base_error(None, None, "customer not authenticated", False)
		session_context.add_tool_trace("get_item_warranty", input_data, output)
		return output

	customer_id = _to_int(session_customer.get("customer_id"))
	target_order_id = _to_int(order_id)
	target_item_id = _to_int(item_id)
	target_product_id = _to_int(product_id)

	if customer_id is None:
		output = _base_error(None, target_order_id, "invalid session customer_id", False)
		session_context.add_tool_trace("get_item_warranty", input_data, output)
		return output

	if target_order_id is None:
		output = _base_error(customer_id, None, "invalid order_id", True)
		session_context.add_tool_trace("get_item_warranty", input_data, output)
		return output

	if not _find_customer_order(customer_id, target_order_id):
		output = _base_error(
			customer_id,
			target_order_id,
			"order not found for authenticated customer",
			True,
		)
		session_context.add_tool_trace("get_item_warranty", input_data, output)
		return output

	order_items = load_s3_data("order_items.csv")
	if not isinstance(order_items, pd.DataFrame):
		output = _base_error(customer_id, target_order_id, "order_items table unavailable", True)
		session_context.add_tool_trace("get_item_warranty", input_data, output)
		return output

	if not {"order_id", "item_id", "product_id", "warranty_expires_at", "item_status"}.issubset(order_items.columns):
		output = _base_error(customer_id, target_order_id, "order_items schema mismatch", True)
		session_context.add_tool_trace("get_item_warranty", input_data, output)
		return output

	target_item = _resolve_item(order_items, target_order_id, target_item_id, target_product_id)
	if target_item.empty:
		output = _base_error(customer_id, target_order_id, "item not found in order", True)
		session_context.add_tool_trace("get_item_warranty", input_data, output)
		return output

	row = target_item.iloc[0]
	resolved_item_id = _to_int(row.get("item_id"))
	resolved_product_id = _to_int(row.get("product_id"))
	item_status = str(_clean_value(row.get("item_status")) or "").strip().lower()

	expires_raw = _clean_value(row.get("warranty_expires_at"))
	expires_ts = pd.to_datetime(str(expires_raw) if expires_raw is not None else None, errors="coerce")
	expires_date = expires_ts.date() if pd.notna(expires_ts) else None
	today = date.today()
	days_remaining = (expires_date - today).days if expires_date else None

	reason = None
	in_warranty = True
	closed_statuses = {"returned", "refunded", "replaced"}

	if item_status in closed_statuses:
		in_warranty = False
		reason = "item is not covered by warranty due to status"
	elif expires_date is None:
		in_warranty = False
		reason = "warranty expiration unavailable"
	elif days_remaining is not None and days_remaining < 0:
		in_warranty = False
		reason = "warranty expired"

	output = {
		"authenticated": True,
		"customer_id": customer_id,
		"order_id": target_order_id,
		"item_id": resolved_item_id,
		"product_id": resolved_product_id,
		"in_warranty": in_warranty,
		"warranty_expires_at": str(expires_date) if expires_date else None,
		"days_remaining": days_remaining,
		"item_status": item_status or None,
		"reason": reason,
	}
	session_context.add_tool_trace("get_item_warranty", input_data, output)
	return output


if __name__ == "__main__":
	import time

	session_context.set_session_customer(1234, "test_user")

	s = time.perf_counter()
	result = get_item_warranty(order_id=74, item_id=221)
	print(result)
	f = time.perf_counter()
	print(f"Time: {f - s}")

	s = time.perf_counter()
	result = get_item_warranty(order_id=74, item_id=221)
	print(result)
	f = time.perf_counter()
	print(f"Time (warm): {f - s}")
