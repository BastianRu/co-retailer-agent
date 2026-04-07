from core.session_context import add_tool_trace, get_tool_trace, set_session_customer
from core.routers.query_router import classify_query_route
from core.agents.query_agent import rewrite_query
from core.agents.rag_agent import solve_query
import json
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

    elapsed = time.perf_counter() - _start
    print(f"✓ Warmup completado en {elapsed:.3f}s\n")

#warm-up
warmup_routers()

for i in range(2):
    messsage = input("Mensaje: ")
    script_start = time.perf_counter()
    first_response_elapsed = None

    rewrited = rewrite_query(messsage)
    summary = rewrited["response_data"].metrics.get_summary()
    last_usage = summary["agent_invocations"][-1]["usage"]
    #print(f"Per-call usage: {last_usage}")
    print("1st routing OK")
    match rewrited["route"]:
        case "QUERY_REWRITE":
          query_route = classify_query_route(messsage)

          summary = query_route["response_data"].metrics.get_summary()
          last_usage = summary["agent_invocations"][-1]["usage"]
          #print(f"Per-call usage: {last_usage}")
          print("2nd routing OK")
          print(query_route["auth_route"])  
          match query_route["auth_route"]:
            case "PUBLIC":
                match query_route["query_route"]:
                    case "FAQ":
                        print("La pregunta es acerca de FAQ! (2 routings)")
                    case "POLICY":
                        response = solve_query(messsage)
                        print(response["message"])
                    case "INVENTORY":
                        print("La pregunta es acerca del Inventario! (2 routings)")
            case "PRIVATE":
                print("La pregunta es privada (1 routings)")
            case "AMBIGUOUS":
                print("pregunta ambigua")
        case "DIRECT_ANSWER":
            print(rewrited["message"])
        case "BLOCK":
            print(rewrited["message"])
                    
    if first_response_elapsed is None:
        first_response_elapsed = time.perf_counter() - script_start
        total_elapsed = time.perf_counter() - script_start
    #print(f"TIME_TO_FIRST_RESPONSE_SECONDS: {first_response_elapsed:.3f}")
    print(f"TOTAL_PROCESS_SECONDS: {total_elapsed:.3f}")
    print("\n\n")



  


  
