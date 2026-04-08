from core.session_context import (
    get_tool_trace, 
    set_session_customer, reset_session,
    get_session_customer)
from core.routers.query_router import classify_query_route
from core.agents.query_agent import rewrite_query
from core.agents.rag_agent import solve_rag_query
from core.agents.inventory_agent import solve_inventory_query
from core.agents.auth_agent import auth_agent_loop
from core.data_store import load_all_s3_data
import time

def warmup_routers():
    print("Precalentando routers...")
    _start = time.perf_counter()
    
    try:
        classify_query_route("ok")  # throwaway agent instance
    except:
        pass
    
    try:
        rewrite_query("ok") # throwaway agent instance
    except:
        pass    
    try:
        load_all_s3_data() #load all S3 dataset
    except:
        pass


    elapsed = time.perf_counter() - _start
    print(f"✓ Warmup completado en {elapsed:.3f}s\n")

#warm-up
warmup_routers()

for i in range(2):
    messsage = input("Mensaje: ")
    script_start = time.perf_counter()

    rewrited = rewrite_query(messsage)
    summary = rewrited["response_data"].metrics.get_summary()
    last_usage = summary["agent_invocations"][-1]["usage"]
    #print(f"Per-call usage: {last_usage}")
    print("1st routing OK")
    match rewrited["route"]:
        case "QUERY_REWRITE":
          query_route = classify_query_route(rewrited["message"])

          summary = query_route["response_data"].metrics.get_summary()
          last_usage = summary["agent_invocations"][-1]["usage"]

          
          print("2nd routing OK")

          print(query_route["auth_route"])  
          match query_route["auth_route"]:
            case "PUBLIC":
                match query_route["query_route"]:
                    case "FAQ":
                        response = solve_rag_query(rewrited["message"])
                        print(response["message"])
                    case "POLICY":
                        response = solve_rag_query(rewrited["message"])
                        print(response["message"])
                    case "INVENTORY":
                        response = solve_inventory_query(rewrited["message"])
                        print(response["message"])
                    case "AMBIGUOUS":
                        print("No pude clasificar con suficiente certeza la consulta publica.")
                    case _:
                        print("No pude determinar el tipo de consulta publica.")
            case "PRIVATE":
                if get_session_customer() is None:
                    response = auth_agent_loop(rewrited["message"])
                    print(response["message"])
                    for _ in range(2):
                        auth_message = input("Mensaje: ")
                        response = auth_agent_loop(auth_message)
                        print(response["message"])
                        if response["stop"] is True:
                            break
                    if response["authenticated"] is False:
                        print("No fue posible autenticarte, intentalo de nuevo mas tarde.")
                        reset_session()
                        break

                response = solve_inventory_query(rewrited["message"])
                print(response["message"])
                print(get_tool_trace())
                
            case "AMBIGUOUS":
                print("pregunta ambigua")
            case _:
                print("No pude determinar si la consulta es publica o privada.")
        case "AUTH_ATTEMPT":
            response = auth_agent_loop(rewrited["message"])
            print(response["message"])
        case "DIRECT_ANSWER":
            print(rewrited["message"])
        case "BLOCK":
            print(rewrited["message"])
        case _:
            print("No pude procesar la solicitud en este momento.")
                    
total_elapsed = time.perf_counter() - script_start
print(f"TOTAL_PROCESS_SECONDS: {total_elapsed:.3f}")
print("\n\n")
reset_session()



  


  
