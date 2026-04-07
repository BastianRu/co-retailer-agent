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


def _base_error(customer_id: int | None, reason: str, authenticated: bool) -> dict:
	return {
		"authenticated": authenticated,
		"customer_id": customer_id,
		"order_id": None,
		"order": {},
		"items": [],
		"reason": reason,
	}


@tool
def get_order_details(order_id: int | None = None) -> dict:
	"""
	Return full details of one authenticated customer's order.

	Args:
		order_id: Optional order identifier.
			- If provided, returns that specific order (must belong to authenticated customer).
			- If omitted, returns the most recent order for the authenticated customer.

	Returns:
		dict: Fixed response structure:
			- authenticated: bool
			- customer_id: int | None
			- order_id: int | None
			- order: dict with full orders.csv fields
			- items: list[dict] with full order_items.csv fields for the order
			- reason: str | None
	"""
	input_data = {"order_id": order_id}

	session_customer = session_context.get_session_customer()
	if session_context._SESSION_CUSTOMER is None or session_customer is None:
		output = _base_error(None, "customer not authenticated", False)
		session_context.add_tool_trace("get_order_details", input_data, output)
		return output

	customer_id = _to_int(session_customer.get("customer_id"))
	if customer_id is None:
		output = _base_error(None, "invalid session customer_id", False)
		session_context.add_tool_trace("get_order_details", input_data, output)
		return output

	orders_data = load_s3_data("orders.csv")
	if not isinstance(orders_data, pd.DataFrame):
		output = _base_error(customer_id, "orders table unavailable", True)
		session_context.add_tool_trace("get_order_details", input_data, output)
		return output

	orders_df = orders_data.copy()
	required_order_columns = {"order_id", "customer_id"}
	if not required_order_columns.issubset(orders_df.columns):
		output = _base_error(customer_id, "orders schema mismatch", True)
		session_context.add_tool_trace("get_order_details", input_data, output)
		return output

	orders_df["_customer_id"] = orders_df["customer_id"].map(_to_int)
	orders_df["_order_id"] = orders_df["order_id"].map(_to_int)
	orders_df = orders_df[orders_df["_customer_id"] == customer_id].copy()

	if orders_df.empty:
		output = _base_error(customer_id, "customer has no orders", True)
		session_context.add_tool_trace("get_order_details", input_data, output)
		return output

	if "order_date" in orders_df.columns:
		orders_df["_order_date"] = pd.to_datetime(orders_df["order_date"], errors="coerce")
		orders_df = orders_df.sort_values(by="_order_date", ascending=False, na_position="last")

	target_order_id = _to_int(order_id)
	if target_order_id is not None:
		target = orders_df[orders_df["_order_id"] == target_order_id].copy().head(1)
		if target.empty:
			output = _base_error(customer_id, "order not found for authenticated customer", True)
			session_context.add_tool_trace("get_order_details", input_data, output)
			return output
	else:
		target = orders_df.head(1).copy()

	row = target.iloc[0]
	resolved_order_id = _to_int(row["order_id"])
	if resolved_order_id is None:
		output = _base_error(customer_id, "invalid order_id in data", True)
		session_context.add_tool_trace("get_order_details", input_data, output)
		return output

	order_payload = {
		col: _clean_value(row[col])
		for col in target.columns
		if not col.startswith("_")
	}

	items_payload: list[dict] = []
	order_items_data = load_s3_data("order_items.csv")
	if isinstance(order_items_data, pd.DataFrame) and not order_items_data.empty:
		items_df = order_items_data.copy()
		if "order_id" in items_df.columns:
			items_df["_order_id"] = items_df["order_id"].map(_to_int)
			items_df = items_df[items_df["_order_id"] == resolved_order_id].copy()

			if not items_df.empty:
				for _, item_row in items_df.iterrows():
					item_payload = {
						col: _clean_value(item_row[col])
						for col in items_df.columns
						if not col.startswith("_")
					}
					items_payload.append(item_payload)

	output = {
		"authenticated": True,
		"customer_id": customer_id,
		"order_id": resolved_order_id,
		"order": order_payload,
		"items": items_payload,
		"reason": None,
	}
	session_context.add_tool_trace("get_order_details", input_data, output)
	return output


if __name__ == "__main__":
	import time

	session_context.set_session_customer(1234, "test_user")

	s = time.perf_counter()
	result = get_order_details(order_id=74)
	print(result)
	f = time.perf_counter()
	print(f"Time: {f - s}")

	s = time.perf_counter()
	result = get_order_details(order_id=74)
	print(result)
	f = time.perf_counter()
	print(f"Time (warm): {f - s}")
