from strands import tool

@tool
def auth_user(identifier: str, identifier_type: str):
  """
  Verify a customer's identity using one allowed identifier.
  Args:
      identifier: The identifier value extracted from the user's message.
      identifier_type: The type of identifier to verify. Must be either
            "dni" or "phone".

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
  


