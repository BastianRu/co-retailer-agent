from core.agent import create_agent
from core.session_context import reset_session
import time

agent = create_agent()


while 1:
  user_message = input("Mensaje: ")

  if (user_message in ["salir"]):
    break

  s = time.perf_counter()
  response = agent(user_message)
  f = time.perf_counter()
  print(str(response))
  print( f"{f - s:.3f}s")
    

