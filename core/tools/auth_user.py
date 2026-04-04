from strands import tool
from core.data_store import load_s3_data
from core.session_context import set_session_customer, add_tool_trace

@tool
def auth_user(identifier: str, identifier_type: str):
    """
    Verify a customer's identity using one allowed identifier.
    Args:
      identifier: The identifier value extracted from the user's message.
      identifier_type: The type of identifier to verify. Must be either
            "dni" or "phone".
            -format for phones: "+57 [999] [999] [9999]"

    Returns:
        A dictionary with a fixed structure:
        {
            "authenticated": bool,
            "customer_id": int | None,
            "display_name": str | None,
            "matched_by": "dni" | "phone" | None,
            "reason": str | None
        }
    """
    #objects for tool tracing
    input = {
        "identifier": identifier,
        "identifier_type": identifier_type
    }
    output = None

    if identifier_type not in {"dni", "phone"}:
        #tool tracing
        output = {
            "authenticated": False, 
            "customer_id": None, 
            "display_name": None, 
            "matched_by": None, 
            "reason": "invalid identifier_type"
        }
        add_tool_trace("auth_user", input, output)
        return output

    customers = load_s3_data("customers")
    customer = customers.loc[customers[identifier_type] == f"{identifier}"]
    
    if customer.empty:
        output = {
            "authenticated": False,
            "customer_id": None,
            "display_name": None,
            "matched_by": None,
            "reason": "customer not found"
        }
        add_tool_trace("auth_user", input, output)
        return output
    
    customer_id = customer.iloc[0]["customer_id"]
    customer_name = customer.iloc[0]["name"]

    output = {
            "authenticated": True,
            "customer_id": customer_id,
            "display_name": customer_name,
            "matched_by": identifier_type,
            "reason": None
        }
    
    set_session_customer(customer_id, customer_name)
    add_tool_trace("auth_user", input, output)

    return output
    

  


