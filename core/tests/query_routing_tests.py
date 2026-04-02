from core.routers.query_router import classify_query_route


RUN_AUTH_COMPARISON = True
RUN_PUBLIC_COMPARISON = False


# -----------------------------
# AUTH CASES (from auth router tests)
# -----------------------------
auth_public_cases = [
    "¿Qué métodos de pago manejan?",
    "¿Hacen envíos a todo Colombia?",
    "¿Cuánto tarda un envío a una ciudad principal?",
    "¿Cuál es la política de devoluciones?",
    "¿Cómo funciona la garantía?",
    "¿Puedo cambiar un producto en general?",
    "¿Cuánto cuesta el producto Samsung Galaxy A55?",
    "¿Tienen stock del producto Air Fryer Oster?",
    "¿Cuáles son sus canales de atención?",
    "¿Cómo hago seguimiento a un pedido?",
    "¿Se puede cambiar la dirección antes del despacho?",
    "¿Qué cubre la garantía de electrónica?"
]

auth_private_cases = [
    "¿Cuál es el total de mi pedido?",
    "¿Cuánto pagué de IVA en mi compra?",
    "¿Dónde está mi pedido?",
    "Muéstrame el historial de envío del pedido 1234.",
    "Quiero saber si mi compra ya fue entregada.",
    "¿Mi producto todavía tiene garantía?",
    "¿Mi pedido ya fue despachado?",
    "¿Cuándo llegará mi pedido?",
    "Quiero devolver el pedido 8472.",
    "¿Cuál fue el subtotal de mi orden?",
    "¿Cuántos intentos de entrega tuvo mi pedido?",
    "Quiero saber el estado de mi reembolso.",
    "Quiero revisar una devolución.",
    "Necesito ayuda con una garantía.",
    "Quiero saber sobre mi envío.",
    "Necesito el estado del pedido.",
    "Quiero consultar un reembolso.",
    "Ayúdame con una compra que hice.",
    "Consulta el pedido 100245.",
    "Revisa el tracking de la orden 8472.",
    "¿Cuál fue el total del pedido 9321?",
    "Quiero devolver el producto del pedido 4567.",
    "¿El pedido 2211 ya fue entregado?"
]

auth_ambiguous_cases = [
    "¿Puedo devolver este producto?",
    "Quiero saber sobre la garantía de mi producto.",
    "¿Cómo es el proceso de reembolso?",
    "¿Qué pasa si mi pedido se demora?",
    "Quiero saber sobre cambios de productos.",
    "¿Qué cubre la garantía en mi caso?",
    "Necesito información sobre una devolución.",
    "¿Puedo cancelar una compra?",
    "¿Cómo funciona el seguimiento del pedido?",
    "¿Qué pasa si el producto llega dañado?"
]

auth_conversational_cases = [
    "Hola, quiero saber el estado de las políticas de devolución actuales.",
    "Necesito información sobre el precio y las políticas de garantía.",
    "Quiero consultar las condiciones actuales de envío y devoluciones.",
    "¿Cuál es la política actual sobre productos en promoción y devoluciones?",
    "Envío este mensaje para preguntar por las políticas de precios.",
    "¿Cómo están manejando actualmente las garantías y los cambios?",
    "Hola, buenas, quería saber si hacen envíos a Pasto.",
    "Buenas tardes, me gustaría saber qué medios de pago aceptan.",
    "Hola, quisiera preguntar cuánto tarda normalmente un envío rural.",
    "Estoy viendo un producto y quería saber si tienen envío gratis.",
    "Antes de comprar, ¿puedo saber si manejan cambios?"
]

auth_special_cases = [
    "Quiero entender mejor mis opciones de devolución en general.",
    "Quiero conocer mis derechos de garantía como cliente.",
    "¿Cómo puedo saber si una compra tiene envío gratis?",
    "Quiero revisar mis posibilidades de cambio antes de comprar."
]

auth_custom_cases = [" Quiero saber mas acerca de este producto "]

selected_auth_cases = auth_private_cases
expected_auth_route = "PRIVATE"


# -----------------------------
# PUBLIC CASES (from public router tests)
# -----------------------------
public_faq_cases = [
    "¿Que metodos de pago manejan?",
    "¿Que medios de pago aceptan?",
    "¿Aceptan Nequi o Daviplata?",
    "¿Se puede pagar por PSE?",
    "¿Manejan pago contraentrega?",
    "¿Hacen envios a todo Colombia?",
    "¿Tienen cobertura nacional?",
    "¿Hacen envios a mi zona?",
    "¿Mandan pedidos a Pasto?",
    "¿Envian hasta municipios o solo ciudades principales?",
    "¿Cuanto tarda un envio a una ciudad principal?",
    "¿Cuanto demora un envio a una ciudad intermedia?",
    "¿Cuanto se demora un envio a zona rural?",
    "¿Los fines de semana hacen envios?",
    "¿Tambien entregan festivos o solo dias habiles?",
    "¿Por donde puedo comprar?",
    "¿Se puede comprar por WhatsApp?",
    "¿Cuales son los canales de atencion?",
    "¿Los puedo contactar por WhatsApp Business?",
    "¿Tienen correo de soporte?",
    "¿Donde descargo la factura?",
    "¿Como bajo la factura electronica?",
    "¿Como hago seguimiento a un pedido en general?",
    "¿Donde miro el tracking de un pedido?",
    "¿Donde veo el numero de guia?",
    "¿Que significa que un pedido este en preparacion?",
    "¿Que significa el estado 'En camino'?",
    "¿Que significa 'Listo para entrega'?",
    "¿Por que un pedido puede llegar en varios paquetes?",
    "¿Cuantos intentos de entrega hacen normalmente?",
    "¿Como reporto un problema con mi pedido?",
    "¿Donde se reportan productos faltantes o incorrectos?",
    "¿Como hago un reclamo por error de cobro?",
    "Buenas, ¿por que medio los contacto si tengo un problema?",
    "Oe, ¿ustedes si hacen envios a toda Colombia o que?",
    "Parcero, ¿como veo el seguimiento del pedido en general?",
    "¿Donde reviso la factura, pues?",
    "¿Que estados puede tener un pedido?",
    "¿Como funciona el seguimiento del pedido en terminos generales?",
    "¿Cuanto puede tardar en actualizarse el tracking despues del despacho?",
]

public_policy_cases = [
    "¿Cual es la politica de devoluciones?",
    "¿Cuanto tiempo tengo para devolver un producto?",
    "¿Cuales son las condiciones para devolver un producto?",
    "¿Que opciones tengo para una devolucion?",
    "¿Puedo pedir cambio por otro producto?",
    "¿Puedo pedir credito en tienda en vez de reembolso?",
    "¿El envio original se reembolsa tambien?",
    "¿Que productos no tienen devolucion?",
    "¿Los productos en promocion tienen cambio o devolucion?",
    "¿Que pasa con un producto marcado como venta final?",
    "¿Aceptan devoluciones de productos usados?",
    "¿Aceptan devolucion si ya no tengo el empaque original?",
    "¿Como es el proceso para solicitar una devolucion?",
    "¿En cuanto tiempo procesan una devolucion?",
    "¿Cuanto tarda el reembolso a tarjeta de credito?",
    "¿Cuanto tarda el reembolso a tarjeta debito?",
    "¿Cuanto tarda el reembolso por PSE?",
    "¿Puedo cancelar una compra antes del despacho?",
    "¿Se puede cancelar despues de que ya va en camino?",
    "¿Puedo cambiar la direccion antes del despacho?",
    "¿Puedo cambiar la direccion si el pedido ya fue despachado?",
    "¿Que cubre la garantia?",
    "¿Que no cubre la garantia?",
    "¿La garantia cubre danos por golpes o caidas?",
    "¿La garantia cubre danos por agua?",
    "¿La garantia cubre desgaste normal?",
    "¿Que pasa si un tecnico no autorizado toca el producto?",
    "¿Cuanto dura la garantia de ropa y calzado?",
    "¿Cuanto dura la garantia de electronica?",
    "¿Como funciona el proceso de garantia?",
    "¿Que resoluciones pueden dar por garantia?",
    "¿Cuanto tarda una reparacion por garantia?",
    "¿Cuanto tarda un reemplazo por garantia?",
    "¿Cuanto tarda un reembolso por garantia?",
    "¿La revision de garantia tiene costo?",
    "¿Necesito factura para reclamar garantia?",
    "¿Donde se solicita la garantia?",
    "¿Si el producto llego incompleto eso entra por garantia?",
    "¿Que pasa si rechazo el paquete al momento de la entrega?",
    "¿Las demoras por clima generan reembolso del envio?",
    "¿Que pasa si la direccion esta mal escrita?",
    "¿Se puede reprogramar la entrega?",
    "¿Puedo redirigir a punto de retiro despues de comprar?",
    "¿Cuando aplica envio gratis?",
    "¿Como calculan el costo de envio en general?",
    "¿Quien responde por el producto mientras esta en transito?",
    "¿Despues de entregado responden por robo o perdida?",
    "¿Cuantos intentos de entrega hacen antes de devolver el paquete?",
    "¿Que pasa si un producto comprado en descuento sale defectuoso?",
    "Parce, si compre algo en promo, ¿igual lo puedo devolver?",
    "Oiga, ¿si el producto ya va en camino todavia lo puedo cancelar o paila?",
    "¿La direccion se puede cambiar despues del despacho o ya no?",
]

public_inventory_cases = [
    "¿Cuanto cuesta este producto?",
    "¿Que precio tiene este producto?",
    "¿Cual es el valor de este producto?",
    "¿Cuanto vale este articulo?",
    "¿Me regalas el precio de este producto?",
    "¿Cuanto cuesta el iPhone 13?",
    "¿Que precio tiene la air fryer Oster?",
    "¿Cuanto vale este Samsung Galaxy A55?",
    "¿Hay stock de este producto?",
    "¿Tienen stock disponible?",
    "¿Hay existencias de este producto?",
    "¿Tienen existencias?",
    "¿Esta disponible este articulo?",
    "¿Sigue disponible este producto?",
    "¿Todavia lo tienen disponible?",
    "¿Se puede comprar todavia?",
    "¿Este producto aun esta a la venta?",
    "¿Lo tienen en inventario?",
    "¿Esta en existencia?",
    "¿Hay disponibilidad de este producto?",
    "¿Quedan unidades?",
    "¿Cuantas unidades quedan?",
    "¿El producto esta agotado?",
    "¿Hay unidades reservadas o disponibles?",
    "¿Tienen en existencia la licuadora Oster?",
    "¿El producto tiene envio gratis?",
    "¿Este articulo tiene envio gratis?",
    "¿Que productos tienen envio gratis?",
    "¿Este producto esta en promocion?",
    "¿Esta en descuento este producto?",
    "¿Esta en oferta este articulo?",
    "¿Cual es el precio con promocion?",
    "¿Cuanto cuesta con descuento?",
    "¿Cuanto cuesta y si hay stock?",
    "¿Cual es el precio y disponibilidad?",
    "¿Tiene stock y cuanto vale?",
    "¿Hay unidades disponibles y precio?",
    "¿Cuanto cuesta y si esta disponible?",
    "¿Precio y existencia del producto?",
    "Buenas, ¿si tienen este producto o ya se acabo?",
    "Oe, ¿todavia venden este producto o ya no?",
    "Parce, ¿queda stock de esa referencia?",
    "¿Hay de este producto todavia, pues?",
    "¿Cuanto vale esa vaina?",
    "¿A como esta este producto?",
    "¿En cuanto sale este articulo?",
    "¿Si hay unidades de este producto o paila?",
    "¿Me confirmas si este producto tiene promo y stock?",
    "¿Este producto aplica para envio gratis o no?",
    "¿Que precio tiene y si lo puedo comprar ya mismo?",
]

public_custom_cases = [" holaa, hacen envios a popa? "]

selected_public_cases = public_policy_cases
expected_public_route = "POLICY"


if RUN_AUTH_COMPARISON:
    for i, case in enumerate(selected_auth_cases):
        response = classify_query_route(case)
        summary = response["response_data"].metrics.get_summary()
        last_usage = summary["agent_invocations"][-1]["usage"]
        print(str(case))
        print(f"Case {i}: expected_auth={expected_auth_route} auth={response['auth_route']} final={response['final_route']}")
        print(f"Query route: {response['query_route']}")
        print(f"Avg cycle (s): {summary['average_cycle_time']}")
        print(f"Per-call usage: {last_usage}")
        print("-------------------------------------")


if RUN_PUBLIC_COMPARISON:
    for i, case in enumerate(selected_public_cases):
        response = classify_query_route(case)
        summary = response["response_data"].metrics.get_summary()
        last_usage = summary["agent_invocations"][-1]["usage"]
        print(str(case))
        print(f"Case {i}: expected_public={expected_public_route} query={response['query_route']} final={response['final_route']} auth={response['auth_route']}")
        print(f"Avg cycle (s): {summary['average_cycle_time']}")
        print(f"Per-call usage: {last_usage}")
        print("-------------------------------------")
