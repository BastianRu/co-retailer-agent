from typing import Any

import pandas as pd
from strands import tool

from core.data_store import load_s3_data
import core.session_context as session_context
from core.tools.inventory.search_products import _to_int


def _clean_value(value: Any) -> Any:
	if pd.isna(value):
		return None
	return value


def _base_error(customer_id: int | None, reason: str, authenticated: bool) -> dict:
	return {
		"authenticated": authenticated,
		"customer_id": customer_id,
		"order_id": None,
		"status": {},
		"shipments": [],
		"tracking_timeline": [],
		"reason": reason,
	}


def _norm_status(value: object) -> str:
	return str(value or "").strip().lower()


@tool
def get_order_shipping_status(order_id: int | None = None) -> dict:
	"""
	Return shipping status, shipment records, and tracking timeline for one customer order.

	Use this tool for delivery progress questions (tracking, ETA, delivered vs in-transit).
	If order_id is omitted, the most recent customer order is used.

	Args:
		order_id: Optional order identifier.
			- If provided, it must belong to the authenticated customer.
			- If omitted, resolves the most recent customer order.

	Returns:
		dict: Stable response payload with keys:
			- authenticated, customer_id, order_id
			- status: normalized shipping summary (booleans, dates, ETA, tracking refs)
			- shipments: list[dict] from shipments.csv for the order
			- tracking_timeline: ordered list[dict] from tracking.csv
			- reason: str | None
			When auth/data validation fails, shipments/timeline are empty and reason is populated.
	"""
	input_data = {"order_id": order_id}

	session_customer = session_context.get_session_customer()
	if session_context._SESSION_CUSTOMER is None or session_customer is None:
		output = _base_error(None, "customer not authenticated", False)
		session_context.add_tool_trace("get_order_shipping_status", input_data, output)
		return output

	customer_id = _to_int(session_customer.get("customer_id"))
	if customer_id is None:
		output = _base_error(None, "invalid session customer_id", False)
		session_context.add_tool_trace("get_order_shipping_status", input_data, output)
		return output

	orders_data = load_s3_data("orders.csv")
	if not isinstance(orders_data, pd.DataFrame):
		output = _base_error(customer_id, "orders table unavailable", True)
		session_context.add_tool_trace("get_order_shipping_status", input_data, output)
		return output

	orders_df = orders_data.copy()
	required_order_columns = {"order_id", "customer_id", "status"}
	if not required_order_columns.issubset(orders_df.columns):
		output = _base_error(customer_id, "orders schema mismatch", True)
		session_context.add_tool_trace("get_order_shipping_status", input_data, output)
		return output

	orders_df["_customer_id"] = orders_df["customer_id"].map(_to_int)
	orders_df["_order_id"] = orders_df["order_id"].map(_to_int)
	orders_df = orders_df[orders_df["_customer_id"] == customer_id].copy()

	if orders_df.empty:
		output = _base_error(customer_id, "customer has no orders", True)
		session_context.add_tool_trace("get_order_shipping_status", input_data, output)
		return output

	if "order_date" in orders_df.columns:
		orders_df["_order_date"] = pd.to_datetime(orders_df["order_date"], errors="coerce")
		orders_df = orders_df.sort_values(by="_order_date", ascending=False, na_position="last")

	target_order_id = _to_int(order_id)
	if target_order_id is not None:
		target = orders_df[orders_df["_order_id"] == target_order_id].copy().head(1)
		if target.empty:
			output = _base_error(customer_id, "order not found for authenticated customer", True)
			session_context.add_tool_trace("get_order_shipping_status", input_data, output)
			return output
	else:
		target = orders_df.head(1).copy()

	order_row = target.iloc[0]
	resolved_order_id = _to_int(order_row["order_id"])
	if resolved_order_id is None:
		output = _base_error(customer_id, "invalid order_id in data", True)
		session_context.add_tool_trace("get_order_shipping_status", input_data, output)
		return output

	shipment_payload: list[dict] = []
	shipments_data = load_s3_data("shipments.csv")
	if isinstance(shipments_data, pd.DataFrame) and not shipments_data.empty:
		shipments_df = shipments_data.copy()
		if "order_id" in shipments_df.columns:
			shipments_df["_order_id"] = shipments_df["order_id"].map(_to_int)
			shipments_df = shipments_df[shipments_df["_order_id"] == resolved_order_id].copy()

			if "shipped_date" in shipments_df.columns:
				shipments_df["_shipped_date"] = pd.to_datetime(shipments_df["shipped_date"], errors="coerce")
				shipments_df = shipments_df.sort_values(by="_shipped_date", ascending=False, na_position="last")

			for _, row in shipments_df.iterrows():
				shipment_payload.append(
					{
						col: _clean_value(row[col])
						for col in shipments_df.columns
						if not col.startswith("_")
					}
				)

	tracking_payload: list[dict] = []
	latest_tracking_status = None
	latest_tracking_location = None
	latest_tracking_timestamp = None

	tracking_data = load_s3_data("tracking.csv")
	if isinstance(tracking_data, pd.DataFrame) and not tracking_data.empty:
		tracking_df = tracking_data.copy()
		if "order_id" in tracking_df.columns:
			tracking_df["_order_id"] = tracking_df["order_id"].map(_to_int)
			tracking_df = tracking_df[tracking_df["_order_id"] == resolved_order_id].copy()

			if "timestamp" in tracking_df.columns:
				tracking_df["_event_ts"] = pd.to_datetime(tracking_df["timestamp"], errors="coerce")
				tracking_df = tracking_df.sort_values(by="_event_ts", ascending=True, na_position="last")

			for _, row in tracking_df.iterrows():
				tracking_payload.append(
					{
						col: _clean_value(row[col])
						for col in tracking_df.columns
						if not col.startswith("_")
					}
				)

			if not tracking_df.empty:
				last_row = tracking_df.iloc[-1]
				latest_tracking_status = _clean_value(last_row.get("status"))
				latest_tracking_location = _clean_value(last_row.get("location"))
				latest_tracking_timestamp = _clean_value(last_row.get("timestamp"))

	order_status = _norm_status(order_row.get("status"))
	shipment_statuses = {_norm_status(s.get("shipment_status")) for s in shipment_payload}

	is_delivered = (
		order_status == "delivered"
		or _clean_value(order_row.get("delivered_at")) is not None
		or "delivered" in shipment_statuses
	)
	is_cancelled = order_status == "cancelled" or _clean_value(order_row.get("cancelled_at")) is not None
	is_returned = order_status == "returned"
	is_shipped = (
		_clean_value(order_row.get("shipped_at")) is not None
		or len(shipment_payload) > 0
		or "shipped" in shipment_statuses
		or "in_transit" in shipment_statuses
	)

	out_for_delivery_statuses = {"out_for_delivery", "ready_for_delivery"}
	is_out_for_delivery = any(status in out_for_delivery_statuses for status in shipment_statuses)

	if isinstance(latest_tracking_status, str):
		is_out_for_delivery = is_out_for_delivery or _norm_status(latest_tracking_status) in out_for_delivery_statuses

	tracking_refs = []
	seen_tracking = set()
	for s in shipment_payload:
		tracking_number = s.get("tracking_number")
		tracking_url = s.get("tracking_url")
		if tracking_number and tracking_number not in seen_tracking:
			tracking_refs.append(
				{
					"tracking_number": tracking_number,
					"tracking_url": tracking_url,
					"carrier": s.get("carrier"),
				}
			)
			seen_tracking.add(tracking_number)

	eta_candidates = [s.get("estimated_delivery_date") for s in shipment_payload if s.get("estimated_delivery_date")]
	eta_date = min(eta_candidates) if eta_candidates else None

	status_payload = {
		"order_status": _clean_value(order_row.get("status")),
		"is_shipped": is_shipped,
		"is_out_for_delivery": is_out_for_delivery,
		"is_delivered": is_delivered,
		"is_cancelled": is_cancelled,
		"is_returned": is_returned,
		"shipped_at": _clean_value(order_row.get("shipped_at")),
		"delivered_at": _clean_value(order_row.get("delivered_at")),
		"cancelled_at": _clean_value(order_row.get("cancelled_at")),
		"current_city": latest_tracking_location,
		"last_tracking_status": latest_tracking_status,
		"last_tracking_timestamp": latest_tracking_timestamp,
		"eta_date": eta_date,
		"tracking": tracking_refs,
	}

	output = {
		"authenticated": True,
		"customer_id": customer_id,
		"order_id": resolved_order_id,
		"status": status_payload,
		"shipments": shipment_payload,
		"tracking_timeline": tracking_payload,
		"reason": None,
	}
	session_context.add_tool_trace("get_order_shipping_status", input_data, output)
	return output


if __name__ == "__main__":
	import time

	session_context.set_session_customer(1234, "test_user")

	s = time.perf_counter()
	result = get_order_shipping_status(order_id=74)
	print(result)
	f = time.perf_counter()
	print(f"Time: {f - s}")

	s = time.perf_counter()
	result = get_order_shipping_status(order_id=74)
	print(result)
	f = time.perf_counter()
	print(f"Time (warm): {f - s}")
