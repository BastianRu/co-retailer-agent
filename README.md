# Co-Retailer Agent

Agente conversacional multi-agente para e-commerce que integra:
- Datos estructurados (CSV en S3)
- Documentos no estructurados (Markdown)
- Orquestación con LLM y tools (Strands + Bedrock)

---

## Configuración de entorno

Crear un archivo `.env` con las siguientes variables:

### AWS / Bedrock

```env
AWS_REGION=us-east-2 #o sa-east-1 segun lo que proporcione la mejor latencia
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_super_secret_key
AWS_SESSION_TOKEN=
```

**Notas:**
- `AWS_SESSION_TOKEN` es opcional. No es necesario si no se usan credenciales temporales.

### AWS / S3

```env
BUCKET_NAME=co-retailer-agent-2026
PREFIX=
```

**Notas:**
- `PREFIX` debe dejarse vacío si los archivos están en el root del bucket.

Se adjunta un archivo `.env.example` como guia adicional.

---

## Carga de datos en S3

El agente espera los datasets y documentos (**exactamente** como fueron suministrados en el reto) en un bucket S3.

### Archivos requeridos

**CSV (datos estructurados):**
- `customers.csv`
- `products.csv`
- `orders.csv`
- y demás tablas del challenge

**Markdown (políticas):**
- Política de envío.md
- Política de garantía.md
- Política de devoluciones.md

Estos **16** archivos se deben subir a un bucket regular en S3, puede tener cualquier nombre, y estar ubicado en cualquier sub-carpeta, siempre y cuando se proporcione ese nombre (`BUCKET_NAME`) y la ruta (`PREFIX`) en el archivo `.env`.

Usando AWS CLI:

```bash
aws s3 cp ./data s3://co-retailer-agent-2026/ --recursive
```

Si se usa una carpeta dentro del bucket:

```env
PREFIX=data/
```

---

## Warm-up del sistema

Al ejecutar `create_agent()`, se realiza un proceso de inicialización.

### ¿Qué hace?
- Carga datos desde S3
- Inicializa estructuras internas
- Prepara contexto compartido
- Configura agentes y herramientas

### Impacto
- La primera ejecución tiene una latencia aproximada de 5 segundos
- Ocurre **una sola vez** por cada ejecución de `create_agent()`
- Está controlado por una flag interna como `_WARMED_UP`

### Beneficio
- Reduce significativamente la latencia en ejecuciones posteriores
- Evita recargar datos innecesariamente

---

## Manejo de inconsistencias en datos

El sistema incluye una flag (parámetro) en `create_agent()` para tolerar ruido en el dataset:

```python
create_agent(..., handle_dataset_inconsistencies: bool = False)
```

### ¿Qué hace?
Permite manejar casos como:
- nombres duplicados o mal escritos
- productos inconsistentes
- variaciones como `"iPhone 15 Pro Max Max"` o `"samsumg ultra ultra"`

Dado que usa un agente adicional, añade un poco de latencia a las respuestas relacionadas con inventario.

### Importancia
El dataset contiene errores intencionales, por lo que:
- no se puede depender de coincidencias exactas
- el agente debe ser robusto ante ambigüedad

---

## Consideraciones clave

### Anti-alucinación

El agente:
- siempre usa tools antes de responder información de fuentes de verdad.

Esto se valida mediante:
- `tool_trace` en `session_context`

### Autenticación

Para consultas sensibles como:
- pedidos
- estado de envíos
- devoluciones

se requiere:
- DNI o teléfono

Si no hay autenticación:
- el agente solicita identificación
- no accede a datos sensibles

Esto se valida mediante:
- `get_session_customer` en `session_context`

### Memoria y sesión

El sistema separa:

- `reset_session()` : limpia autenticación y `tool_trace`
- `reset_memory()` : limpia contexto conversacional

Esto permite:
- mantener identidad entre turnos
- resetear conversación sin perder sesión, si es necesario

### Contrato de create_agent()

El entrypoint principal del proyecto es:

```python
create_agent(streaming: bool = False, handle_dataset_inconsistencies: bool = False)
```
- `streaming`: se conserva por compatibilidad con el contrato técnico del challenge pero en la versión actual no modifica el comportamiento interno de respuesta.


### **Esto es todo lo que se necesita para el uso del agente.** 

---

## Arquitectura del sistema 

El sistema sigue una arquitectura multi-agente orquestada.

### Flujo general

```text
User → Input Agent → Query Router → Agents → Tools/Data → Response
```

### Componentes

#### Input Agent
- Clasifica el mensaje inicial
- Detecta queries nuevas, follow-ups, intentos de autenticación y mensajes inválidos

#### Query Router
Decide a qué agente enviar la consulta:
- FAQ
- Policies (RAG)
- Inventory (público)
- Private (requiere auth)

#### Auth Agent
- Verifica identidad mediante DNI o teléfono
- Bloquea acceso si no está autenticado

#### Agentes de ejecución

**FAQ Agent (planeado)**
- Preguntas generales
- No requiere tools

**RAG Agent**
- Consulta documentos Markdown
- Responde políticas

**Inventory Agent**
- Productos
- Stock
- Pedidos
- Tracking
- Garantías
- Devoluciones

#### Session Context

Módulo global compartido.

Contiene:
- `tool_trace`
- `customer_id`
- estado de autenticación
- Otras variables internas de context.

#### Data Layer

- S3: almacenamiento de datos
- CSV: datos estructurados
- Markdown: políticas
- Bedrock: modelo LLM
- Strands: orquestación y tools

---

## Uso básico

```python
agent = create_agent()

response = agent("¿Dónde está mi pedido?")
print(response)
```

---

## Notas finales

- El sistema está optimizado para datos ruidosos

- Prioriza trazabilidad, seguridad y control de flujo
- La latencia inicial es esperada por el warm-up
