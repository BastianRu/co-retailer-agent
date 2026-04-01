from core.session_context import add_tool_trace, get_tool_trace, set_session_customer
from core.routers.auth_routing import classify_auth_route
from core.routers.public_routing import classify_public_route
from core.routers.context_routing import classify_context_route
from core.routers.auth_routing import classify_auth_route
from core.routers.public_routing import classify_public_route
import json
import time

def warmup_routers():
    print("Precalentando routers...")
    _start = time.perf_counter()
    
    try:
        classify_auth_route("ok")  # throwaway agent instance
    except:
        pass
    
    try:
        classify_public_route("ok")  # throwaway agent instance
    except:
        pass

    try:
        classify_context_route("ok") # throwaway agent instance
    except:
        pass    

    elapsed = time.perf_counter() - _start
    print(f"✓ Warmup completado en {elapsed:.3f}s\n")

#warm-up
warmup_routers()

for i in range(4):
    messsage = input("Mensaje: ")
    script_start = time.perf_counter()
    first_response_elapsed = None

    context = classify_context_route(messsage)
    match context["route"]:
        case "AUTO":
          auth = classify_auth_route(messsage)
          match auth["route"]:
            case "PUBLIC":
                public = classify_public_route(messsage)
                match public["route"]:
                    case "FAQ":
                        print("La pregunta es acerca de FAQ! (3 routings)")
                    case "POLICY":
                        print("La pregunta es acerca de Politicas!  (3 routings)")
                    case "INVENTORY":
                        print("La pregunta es acerca del Inventario! (3 routings)")
            case "PRIVATE":
                print("La pregunta es privada (2 routings)")
        case "FOLLOW":
            print("Se necesita memoria! (1 routing)")
        case "TALK":
            print("El mensaje es puramente conversacional")
                    
    if first_response_elapsed is None:
        first_response_elapsed = time.perf_counter() - script_start
        total_elapsed = time.perf_counter() - script_start
    print(f"TIME_TO_FIRST_RESPONSE_SECONDS: {first_response_elapsed:.3f}")
    print(f"TOTAL_PROCESS_SECONDS: {total_elapsed:.3f}")



  


  
