from core.routers.public_routing import classify_public_intent, inspect_public_routing, passes_delta
from colorama import init, Fore, Style

#Colorama init
init(autoreset=True)

#Faq cases
faq_cases = [
    "¿Qué métodos de pago manejan?",
    "¿Hacen envíos a todo Colombia?",
    "¿Cuáles son sus canales de atención?",
    "¿Cómo hago seguimiento a un pedido en general?",
    "¿Dónde puedo descargar la factura?",
    "¿Cuánto tarda un envío a Pasto?",
    "¿Cómo reporto un problema con mi pedido?",
    "¿Puedo contactarlos por WhatsApp?"
]

#Policy 
policy_cases = [
    "¿Cuál es la política de devoluciones?",
    "¿Qué cubre la garantía?",
    "¿Qué productos no tienen devolución?",
    "¿Cuánto tiempo tengo para devolver un producto?",
    "¿Cuánto tarda un reembolso?",
    "¿Puedo cancelar una compra antes del despacho?",
    "¿Puedo cambiar la dirección antes del envío?",
    "¿Qué pasa si el producto fue comprado en promoción?"
]

#Inventory 
#It has more cases due to we have to focus on inventory routing (it's the risky case)
inventory_cases = [
    "¿Cuánto cuesta el iPhone 13?",
    "¿Tienen stock del Samsung Galaxy A55?",
    "¿Cuál es el precio de una air fryer Oster?",
    "¿Hay unidades disponibles de este producto?",
    "¿Este producto tiene envío gratis?",
    "¿Cuántas unidades quedan?",
    "¿El producto está agotado?",
    "¿Qué productos tienen envío gratis?",
    "¿Cuánto vale este producto?",
    "¿Qué precio tiene este artículo?",
    "¿Hay disponibilidad de este producto?",
    "¿Tienen existencias?",
    "¿Está disponible en inventario?",
    "¿Quedan unidades?",
    "¿Está agotado?",
    "¿Cuánto cuesta y si tiene stock?",
    "¿Este producto tiene envío gratis?",
    "¿Qué productos tienen envío gratis?",
    "¿Cuánto cuesta el Samsung Galaxy A55?",
    "¿Hay stock del iPhone 13?",
    "¿Cuál es el precio de esta referencia?",
    "¿Puedo comprar este producto todavía?",
    "¿Sigue disponible este artículo?",
    "¿Hay unidades reservadas o disponibles?",
    "¿Cuál es el valor del producto?",
    "¿Tienen en existencia la air fryer Oster?",
    "¿Está en inventario?",
    "¿Hay disponibilidad?",
    "¿Se puede comprar todavía?",
    "¿Sigue disponible este artículo?",
    "¿Este producto aún está a la venta?",
    "¿Lo tienen disponible?",
    "¿Está en existencia?",
    "¿Se encuentra disponible actualmente?",
    "¿Puedo conseguir este producto?",
    "¿Está habilitada la compra de este producto?",
    "¿Cuánto cuesta y si hay stock?",
    "¿Cuál es el precio y disponibilidad?",
    "¿Tiene stock y cuánto vale?",
    "¿Hay unidades disponibles y precio?",
    "¿Cuánto cuesta y si está disponible?",
    "¿Precio y existencia del producto?",
    "Estoy interesado en este producto, ¿lo tienen?",
    "Quiero comprar esto, ¿todavía se puede?",
    "Estoy viendo este producto, ¿está disponible?",
    "Antes de comprar, ¿hay unidades?",
    "Estoy pensando en comprarlo, ¿hay stock?",
    "¿Crees que aún haya unidades disponibles?",
    "¿Este producto tiene envío gratis?",
    "¿Este producto está en promoción?",
    "¿Cuánto cuesta con envío?",
    "¿El precio incluye envío?",
    "¿Este producto tiene descuento?",
    "¿Está en oferta este producto?",
    "¿Cuál es el precio con promoción?",
    "Hola, quería saber si tienen stock del producto",
    "Buenas, me interesa este artículo, ¿está disponible?",
    "Oye, ¿todavía venden este producto?",
    "Disculpa, ¿cuánto cuesta esto?",
    "Hey, ¿queda algo en inventario?",
    "Quisiera saber si aún tienen unidades",
    "Quiero saber si puedo comprar este producto",
    "Necesito saber si está disponible",
    "Estoy viendo si hay unidades",
    "Quisiera saber sobre disponibilidad",
    "Me interesa este producto, ¿hay?",
    "¿Este producto tiene envío gratis y está disponible?",
    "¿Está en promoción y hay stock?",
    "¿Cuánto cuesta y si aplica descuento?",
    "¿El producto tiene garantía y cuánto vale?",
    "¿Se puede comprar y cuánto cuesta?",
    "Cuanto cuesta esta vaina?"
]

#Frontier 
boundary_cases = [
    "¿Cómo funciona el seguimiento del pedido?",
    "¿Puedo cancelar una compra?",
    "¿Qué pasa si mi pedido se demora?",
    "¿Puedo devolver este producto?",
    "¿El envío tiene costo?",
    "¿Este producto tiene envío gratis?",
    "¿Qué pasa si el producto llega dañado?",
    "¿Cuánto tarda el reembolso?"
]
 
#s_cases: faq | pol | inv | bound
s_case = "inv"
match s_case:
    case "faq":
        print("FAQ cases\n")
        for i, query in enumerate(faq_cases):
            output = classify_public_intent(query)
            delta_info = passes_delta(query)
            print(f"{i}. Query: {query}")
            if str(output.name) != "FAQ" or delta_info["passes"] is False:
                print(f"{Fore.RED}[Failed]{Style.RESET_ALL}")
            print(f"Predicted Route: {output.name}")
            print(f"CS: {output.similarity_score}")
            print(f"Delta: {delta_info['delta']}")
            output_meta = inspect_public_routing(query)
            print(output_meta["passed_routes"])
            print("\n")  
    case "pol":
        print("Policy cases\n")
        for i, query in enumerate(policy_cases):
            output = classify_public_intent(query)
            delta_info = passes_delta(query)
            print(f"{i}. Query: {query}")
            if str(output.name) != "Politicas" or delta_info["passes"] is False:
                print(f"{Fore.RED}[Failed]{Style.RESET_ALL}")
            print(f"Predicted Route: {output.name}")            
            print(f"CS: {output.similarity_score}")
            print(f"Delta: {delta_info['delta']}")
            output_meta = inspect_public_routing(query)
            print(output_meta["passed_routes"])
            print("\n")
    case "inv":
        print("Inventory cases\n")
        for i, query in enumerate(inventory_cases):
            output = classify_public_intent(query)
            delta_info = passes_delta(query)
            print(f"{i}. Query: {query}")
            if str(output.name) != "Inventario" or delta_info["passes"] is False:
                print(f"{Fore.RED}[Failed]{Style.RESET_ALL}")
            print(f"Predicted Route: {output.name}")
            print(f"CS: {output.similarity_score}")
            print(f"Delta: {delta_info['delta']}")
            output_meta = inspect_public_routing(query)
            print(output_meta["passed_routes"])
            print("\n")
    case "bound":
        print("Boundary cases\n")
        for i, query in enumerate(boundary_cases):
            output = classify_public_intent(query)
            print(f"Query: {query}")
            print(f"{i}: Predicted Route: {output.name}")
            print(f"CS: {output.similarity_score}")
            output_meta = inspect_public_routing(query)
            print(output_meta["passed_routes"])
            print("\n")
    case _:
        print("Another query...\n")
        query = "¿Cuánto tarda un envío a Pasto?"
        output = classify_public_intent(query)
        output_meta = inspect_public_routing(query)
        print(output_meta["passed_routes"])
        print("\n")

        