import os, time, random, socket, threading
from typing import Dict, List, Tuple, Optional

import grpc
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

import inference_pb2 as pb
import inference_pb2_grpc as pbg

ROUTER_HTTP_PORT = int(os.getenv("ROUTER_HTTP_PORT", "8080"))

# In-cluster DNS names (headless Services) for per-pod endpoint discovery
MODEL1_DNS = os.getenv("MODEL1_DNS", "model1-headless.default.svc.cluster.local")
MODEL2_DNS = os.getenv("MODEL2_DNS", "model2-headless.default.svc.cluster.local")
MODEL_GRPC_PORT = int(os.getenv("MODEL_GRPC_PORT", "50051"))

MIN_RECHECK_SEC = float(os.getenv("ENDPOINT_RECHECK_SEC", "2.0"))

app = FastAPI(title="OCR Monitoring Router")

class Worker:
    def __init__(self, model: str, target: str):
        self.model = model
        self.target = target  # "ip:port"
        self.channel = grpc.insecure_channel(target)
        self.stub = pbg.InferenceStub(self.channel)
        self.inflight = 0
        self.last_seen = time.time()

def resolve_ips(host: str) -> List[str]:
    # Return a stable, deduped list of IPs for the headless service
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
        ips = sorted({i[4][0] for i in infos})
        return ips
    except Exception:
        return []

class PoolManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.pools: Dict[str, Dict[str, Worker]] = {"model1": {}, "model2": {}}
        self.last_refresh = 0.0
        self.last_sizes = {"model1": 0, "model2": 0}
        self.seen_pods = set()

    def refresh(self):
        now = time.time()
        if now - self.last_refresh < MIN_RECHECK_SEC:
            return

        with self.lock:
            self.last_refresh = now

            mapping = {"model1": MODEL1_DNS, "model2": MODEL2_DNS}
            for model, dns in mapping.items():
                ips = resolve_ips(dns)
                targets = [f"{ip}:{MODEL_GRPC_PORT}" for ip in ips]

                # Add new workers
                for tgt in targets:
                    if tgt not in self.pools[model]:
                        self.pools[model][tgt] = Worker(model, tgt)

                # Mark unseen workers (optional pruning)
                for tgt, w in self.pools[model].items():
                    if tgt in targets:
                        w.last_seen = now

                # Track size changes for "scaled up" signal
                self.last_sizes[model] = len(self.pools[model])

    def pick_model(self) -> str:
        return random.choice(["model1", "model2"])

    def pick_worker(self, model: str) -> Optional[Worker]:
        with self.lock:
            workers = list(self.pools.get(model, {}).values())
            if not workers:
                return None
            # least inflight
            return min(workers, key=lambda w: w.inflight)

    def incr(self, w: Worker):
        with self.lock:
            w.inflight += 1

    def decr(self, w: Worker):
        with self.lock:
            w.inflight = max(0, w.inflight - 1)

    def pool_size(self, model: str) -> int:
        with self.lock:
            return len(self.pools.get(model, {}))

pm = PoolManager()

INDEX_HTML = r"""
<!doctype html>
<html>
  <head>
    <meta charset="utf-8"/>
    <title>OCR Monitoring Router</title>
    <style>
      body { font-family: Arial, sans-serif; margin: 24px; }
      .box { border: 1px solid #ddd; border-radius: 10px; padding: 16px; max-width: 720px; }
      .row { margin: 12px 0; }
      pre { background: #111; color: #eee; padding: 12px; border-radius: 10px; overflow: auto; }
      button { padding: 10px 14px; border-radius: 10px; border: 1px solid #ccc; cursor: pointer; }
    </style>
  </head>
  <body>
    <div class="box">
      <h2>OCR Monitoring Router</h2>
      <div class="row">
        <form id="f">
          <input type="file" id="files" name="files" multiple />
          <button type="submit">Upload & Predict</button>
        </form>
      </div>
      <div class="row">
        <b>Result</b>
        <pre id="out">{}</pre>
      </div>
    </div>

    <script>
      const form = document.getElementById("f");
      form.addEventListener("submit", async (e) => {
        e.preventDefault();
        const input = document.getElementById("files");
        const fd = new FormData();
        for (const f of input.files) fd.append("files", f, f.name);

        const r = await fetch("/predict", { method: "POST", body: fd });
        const j = await r.json();
        document.getElementById("out").textContent = JSON.stringify(j, null, 2);
      });
    </script>
  </body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML

@app.post("/predict")
async def predict(files: List[UploadFile] = File(...)):
    pm.refresh()

    results = []
    for f in files:
        img = await f.read()
        model = pm.pick_model()

        pm.refresh()
        worker = pm.pick_worker(model)
        if worker is None:
            return JSONResponse(
                status_code=503,
                content={"error": f"No backends available for {model}. Check DNS/service/endpoints."}
            )

        pm.incr(worker)

        t0 = time.perf_counter()
        try:
            resp = worker.stub.Predict(pb.PredictRequest(image=img, filename=f.filename))
        except grpc.RpcError as e:
            pm.decr(worker)
            return JSONResponse(status_code=503, content={"error": f"Backend error: {e}"})
        finally:
            pm.decr(worker)

        router_ms = (time.perf_counter() - t0) * 1000.0
        pool_sz = pm.pool_size(model)

        # "Was another pod made active?" Best-effort signal:
        # If pool size > 1, we're scaled beyond the always-on replica.
        # Also report pod start age as a "fresh pod" hint.
        pod_age_s = max(0, int(time.time()) - int(resp.started_at_unix))
        scaled_up = pool_sz > 1
        likely_new_pod = pod_age_s < 120 and scaled_up  # heuristic

        results.append({
            "filename": f.filename,
            "selected_model": model,
            "backend_target": worker.target,
            "backend_pod": resp.pod_name,
            "backend_pod_ip": resp.pod_ip,
            "backend_pod_age_s": pod_age_s,
            "scaled_up_active": scaled_up,
            "likely_new_pod_used": likely_new_pod,
            "prediction_text": resp.text,
            "model_infer_ms": resp.infer_ms,
            "router_total_ms": router_ms
        })

    return {"count": len(results), "results": results}

def main():
    # warm endpoint discovery
    pm.refresh()
    uvicorn.run(app, host="0.0.0.0", port=ROUTER_HTTP_PORT)

if __name__ == "__main__":
    main()
