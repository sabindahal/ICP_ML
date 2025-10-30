import os, time, random, asyncio
from typing import Dict, List

import grpc
import inference_pb2 as pb
import inference_pb2_grpc as pbg

# env: BACKENDS="model2=host.docker.internal:50051,model4=host.docker.internal:50052"
BACKENDS = os.getenv("BACKENDS", "model2=localhost:50051,model4=localhost:50052")
ROUTER_PORT = int(os.getenv("ROUTER_PORT", 50050))

class Worker:
    def __init__(self, variant, target):
        self.variant = variant
        self.target = target
        self.channel = grpc.insecure_channel(target)   # sync
        self.stub = pbg.InferenceStub(self.channel)
        self.inflight = 0

def parse_backends(s: str) -> Dict[str, List[Worker]]:
    pools = {}
    for entry in s.split(","):
        if "=" not in entry: continue
        variant, hosts = entry.split("=")
        pools[variant] = [Worker(variant, h) for h in hosts.split(";") if h]
    return pools

POOLS = parse_backends(BACKENDS)
LOCK = asyncio.Lock()

def pick_random_variant():
    keys = list(POOLS.keys())
    return random.choice(keys) if keys else None

def pick_least_busy(workers):
    return min(workers, key=lambda w: w.inflight)

class Router(pbg.InferenceServicer):
    async def Predict(self, request, context):
        variant = pick_random_variant()
        if not variant or not POOLS.get(variant):
            context.abort(grpc.StatusCode.UNAVAILABLE, "No backend available")

        async with LOCK:
            worker = pick_least_busy(POOLS[variant])
            worker.inflight += 1

        loop = asyncio.get_running_loop()

        def call_worker():
            t0 = time.perf_counter()
            resp = worker.stub.Predict(request)
            net = (time.perf_counter() - t0) * 1000
            return resp, net

        try:
            resp, net = await loop.run_in_executor(None, call_worker)

            context.set_trailing_metadata((
                ("x-variant", variant),
                ("x-target", worker.target),
                ("x-router-net-ms", f"{net:.2f}")
            ))

            return resp

        except grpc.RpcError as e:
            context.abort(grpc.StatusCode.UNAVAILABLE, f"backend error {e}")

        finally:
            async with LOCK:
                worker.inflight -= 1

async def serve():
    server = grpc.aio.server()
    pbg.add_InferenceServicer_to_server(Router(), server)
    server.add_insecure_port(f"[::]:{ROUTER_PORT}")
    await server.start()
    print(f"[ROUTER READY] :{ROUTER_PORT} backends={ {v: [w.target for w in ws] for v,ws in POOLS.items()} }")
    await server.wait_for_termination()

if __name__ == "__main__":
    asyncio.run(serve())
