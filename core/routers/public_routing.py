import statistics
import time

from semantic_router import Route
from semantic_router.encoders import OllamaEncoder
from semantic_router.routers import SemanticRouter

routes = [
    Route(name="saludo", utterances=["hola", "buenas", "que tal"]),
    Route(name="despedida", utterances=["adios", "nos vemos", "hasta luego"]),
]


encoder = OllamaEncoder(
    name="nomic-embed-text",
    base_url="http://localhost:11434",
)

router = SemanticRouter(
    encoder=encoder,
    routes=routes,
    auto_sync="local",
)

def sec(start, end):
    return end - start

def bench_latency(query: str, n: int = 10, warmup: int = 2):
    total_t = []
    embed_t = []
    route_t = []

    bench_start = time.perf_counter()

    for _ in range(warmup):
        _ = router(query)

    for _ in range(n):
        t0 = time.perf_counter()
        out = router(query)
        t1 = time.perf_counter()
        total_t.append(sec(t0, t1))

        t2 = time.perf_counter()
        vector = encoder([query])
        t3 = time.perf_counter()
        embed_t.append(sec(t2, t3))

        t4 = time.perf_counter()
        out_vec = router(vector=vector)
        t5 = time.perf_counter()
        route_t.append(sec(t4, t5))

    bench_end = time.perf_counter()

    def p95(vals):
        vals_sorted = sorted(vals)
        idx = int(0.95 * (len(vals_sorted) - 1))
        return vals_sorted[idx]

    print("Query:", query)
    print("Ruta detectada (texto):", out.name)
    print("Ruta detectada (vector):", out_vec.name)
    print("--- Tiempos de ejecución (s) ---")
    print("Total por query     avg={:.4f}s p95={:.4f}s".format(statistics.mean(total_t), p95(total_t)))
    print("Embedding por query avg={:.4f}s p95={:.4f}s".format(statistics.mean(embed_t), p95(embed_t)))
    print("Routing por query   avg={:.4f}s p95={:.4f}s".format(statistics.mean(route_t), p95(route_t)))
    print("Tiempo total benchmark: {:.4f}s".format(sec(bench_start, bench_end)))

TEST_INPUTS = [
"hola amigo",
"como estamos??",
"que onda!",
"nos vemos!",
"hasta la proxima!",
"chaito!",
]

if __name__ == "__main__":
    for query in TEST_INPUTS:
        bench_latency(query=query, n=10, warmup=2)
    
