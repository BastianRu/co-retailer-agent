from core.data_store import load_s3_data
import core.session_context as session_context
from core.tools.inventory.search_products import _to_int
from strands import tool
import pandas as pd
from typing import Any


def _clean_value(value: Any) -> Any:
	if pd.isna(value):
		return None
	return value


@tool
def get_customer_orders() -> dict:
	"""
	Return a summary list of orders for the authenticated customer.

	Use this tool when the user asks for their orders list or order history.
	Returns only top-level order summary fields; use get_order_details for
	full order contents, items, and shipping address.

	Returns:
		dict: Stable response payload with keys:
			- authenticated: bool
			- customer_id: int | None
			- orders: list[dict] each with: order_id, order_date, status, total, payment_method
			- reason: str | None
			When auth or data checks fail, orders is [] and reason explains why.
	"""
	input_data = {}

	# Validate authenticated session from the shared context.
	session_customer = session_context.get_session_customer()
	if session_context._SESSION_CUSTOMER is None or session_customer is None:
		output = {
			"authenticated": False,
			"customer_id": None,
			"orders": [],
			"reason": "customer not authenticated",
		}
		session_context.add_tool_trace("get_customer_orders", input_data, output)
		return output

	customer_id = _to_int(session_customer.get("customer_id"))
	if customer_id is None:
		output = {
			"authenticated": False,
			"customer_id": None,
			"orders": [],
			"reason": "invalid session customer_id",
		}
		session_context.add_tool_trace("get_customer_orders", input_data, output)
		return output

	orders_data = load_s3_data("orders.csv")
	if not isinstance(orders_data, pd.DataFrame):
		output = {
			"authenticated": True,
			"customer_id": customer_id,
			"orders": [],
			"reason": "orders table unavailable",
		}
		session_context.add_tool_trace("get_customer_orders", input_data, output)
		return output

	orders_df = orders_data.copy()
	required_order_columns = {"order_id", "customer_id"}
	if not required_order_columns.issubset(orders_df.columns):
		output = {
			"authenticated": True,
			"customer_id": customer_id,
			"orders": [],
			"reason": "orders schema mismatch",
		}
		session_context.add_tool_trace("get_customer_orders", input_data, output)
		return output

	orders_df["_customer_id"] = orders_df["customer_id"].map(_to_int)
	orders_df = orders_df[orders_df["_customer_id"] == customer_id].copy()

	if orders_df.empty:
		output = {
			"authenticated": True,
			"customer_id": customer_id,
			"orders": [],
			"reason": None,
		}
		session_context.add_tool_trace("get_customer_orders", input_data, output)
		return output

	if "order_date" in orders_df.columns:
		orders_df["_order_date"] = pd.to_datetime(orders_df["order_date"], errors="coerce")
		orders_df = orders_df.sort_values(by="_order_date", ascending=False, na_position="last")

	_SUMMARY_FIELDS = ["order_id", "order_date", "status", "total", "payment_method"]

	response_orders: list[dict] = []
	for _, row in orders_df.iterrows():
		order = {
			field: _clean_value(row[field])
			for field in _SUMMARY_FIELDS
			if field in orders_df.columns
		}
		response_orders.append(order)

	output = {
		"authenticated": True,
		"customer_id": customer_id,
		"orders": response_orders,
		"reason": None,
	}
	session_context.add_tool_trace("get_customer_orders", input_data, output)
	return output


if __name__ == "__main__":
      import time
      session_context.set_session_customer(1001, "test_user")
      s = time.perf_counter()
      results = get_customer_orders()
      for r in results["orders"]:
         print(f"{r}\n")

      f = time.perf_counter()
      print(f"Time: {f - s}")
      s = time.perf_counter()
      results = get_customer_orders()

      f = time.perf_counter()
      print(f"Time (warm): {f - s}")
      for r in results["orders"]:
         print(f"{r}\n")
