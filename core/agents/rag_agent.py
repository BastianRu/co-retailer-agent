import json
import os 
import boto3
from dotenv import load_dotenv
import time

test_text="""


## **Vigencia de la Garantía por Categoría**

La duración de la garantía varía según la categoría del producto:

| Categoría | Garantía | Cobertura |
| ----- | ----- | ----- |
| **Electrónica** | 6 a 36 meses | Defectos de fabricación, fallas de software y hardware |
| **Electrodomésticos** | 6 a 36 meses | Defectos de fabricación, fallas mecánicas y eléctricas |
| **Hogar y Muebles** | 6 a 36 meses | Defectos estructurales, fallas en mecanismos |
| **Ropa y Calzado** | 90 días | Defectos de confección y fallas en materiales |
| **Deportes y Fitness** | 6 a 24 meses | Defectos de fabricación, fallas mecánicas |
| **Belleza y Cuidado Personal** | 6 a 36 meses | Defectos en equipos eléctricos |
| **Libros y Papelería** | 6 a 36 meses | Defectos en dispositivos electrónicos |
| **Juguetes y Bebés** | 6 a 36 meses | Defectos de fabricación, seguridad del producto |

**Nota:** El periodo exacto de garantía de cada producto se encuentra indicado en la ficha del producto.

---
"""

s = time.perf_counter()
client = boto3.client(
    'bedrock-runtime', 
    region_name= "us-east-2"
    )

for i in range(10):
  response = client.invoke_model(
                  modelId="amazon.titan-embed-text-v2:0",
                  body=json.dumps({"inputText": test_text})
              )


response = json.loads(response['body'].read())["embedding"]

f = time.perf_counter()
print(response)
print(f - s)
