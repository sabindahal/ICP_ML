import os, time, random, socket, threading, math
from typing import Dict, List, Optional, Deque, Tuple
from collections import deque

import grpc
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

import inference_pb2 as pb
import inference_pb2_grpc as pbg


ROUTER_HTTP_PORT = int(os.getenv("ROUTER_HTTP_PORT", "8080"))

# Headless service DNS names (per-pod discovery)
MODEL1_DNS = os.getenv("MODEL1_DNS", "model1-headless.default.svc.cluster.local")
MODEL2_DNS = os.getenv("MODEL2_DNS", "model2-headless.default.svc.cluster.local")
MODEL_GRPC_PORT = int(os.getenv("MODEL_GRPC_PORT", "50051"))

ENDPOINT_RECHECK_SEC = float(os.getenv("ENDPOINT_RECHECK_SEC", "2.0"))

# Policy knobs
DEADLINE_MS = float(os.getenv("DEADLINE_MS", "300"))
WINDOW_SEC = float(os.getenv("WINDOW_SEC", "30"))
MIN_SAMPLES = int(os.getenv("MIN_SAMPLES", "10"))
SWITCH_COOLDOWN_SEC = float(os.getenv("SWITCH_COOLDOWN_SEC", "10"))
LAT_PCTL = float(os.getenv("LAT_PCTL", "95"))

ACCURATE_MODEL = os.getenv("ACCURATE_MODEL", "model1")
FAST_MODEL = os.getenv("FAST_MODEL", "model2")

app = FastAPI(title="OCR Monitoring Router")


def json_safe_float(x: float) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return float(x)


def resolve_ips(host: str) -> List[str]:
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
        ips = sorted({i[4][0] for i in infos})
        return ips
    except Exception:
        return []


class Worker:
    def __init__(self, model: str, target: str):
        self.model = model
        self.target = target  # "ip:port"
        self.channel = grpc.insecure_channel(target)
        self.stub = pbg.InferenceStub(self.channel)
        self.inflight = 0
        self.last_seen = time.time()


class RollingStats:
    """
    Stores recent request outcomes per model:
    (timestamp, router_total_ms, ok)
    """
    def __init__(self, window_sec: float):
        self.window_sec = window_sec
        self.data: Dict[str, Deque[Tuple[float, float, bool]]] = {
            "model1": deque(),
            "model2": deque(),
        }

    def _prune(self, model: str, now: float):
        dq = self.data.get(model)
        if dq is None:
            return
        cutoff = now - self.window_sec
        while dq and dq[0][0] < cutoff:
            dq.popleft()

    def add(self, model: str, router_ms: float, ok: bool):
        now = time.time()
        if model not in self.data:
            self.data[model] = deque()
        self.data[model].append((now, float(router_ms), bool(ok)))
        self._prune(model, now)

    def summary(self, model: str) -> Dict[str, Optional[float]]:
        now = time.time()
        if model not in self.data:
            return {"n": 0.0, "pctl_ms": None, "err_rate": 1.0}

        self._prune(model, now)
        dq = self.data[model]
        n = len(dq)
        if n == 0:
            return {"n": 0.0, "pctl_ms": None, "err_rate": 1.0}

        lat = np.array([x[1] for x in dq], dtype=float)
        ok = np.array([x[2] for x in dq], dtype=bool)

        pctl_ms = float(np.percentile(lat, LAT_PCTL)) if n > 0 else None
        err_rate = float(1.0 - ok.mean()) if n > 0 else 1.0

        return {
            "n": float(n),
            "pctl_ms": json_safe_float(pctl_ms),
            "err_rate": json_safe_float(err_rate),
        }


class PoolManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.pools: Dict[str, Dict[str, Worker]] = {"model1": {}, "model2": {}}
        self.last_refresh = 0.0

        self.stats = RollingStats(window_sec=WINDOW_SEC)

        # hysteresis
        self.last_choice = FAST_MODEL
        self.last_switch_at = 0.0

    def refresh(self):
        now = time.time()
        if now - self.last_refresh < ENDPOINT_RECHECK_SEC:
            return

        with self.lock:
            self.last_refresh = now
            mapping = {"model1": MODEL1_DNS, "model2": MODEL2_DNS}

            for model, dns in mapping.items():
                ips = resolve_ips(dns)
                targets = [f"{ip}:{MODEL_GRPC_PORT}" for ip in ips]

                for tgt in targets:
                    if tgt not in self.pools[model]:
                        self.pools[model][tgt] = Worker(model, tgt)

                for tgt, w in self.pools[model].items():
                    if tgt in targets:
                        w.last_seen = now

    def pool_size(self, model: str) -> int:
        with self.lock:
            return len(self.pools.get(model, {}))

    def pick_worker(self, model: str) -> Optional[Worker]:
        with self.lock:
            workers = list(self.pools.get(model, {}).values())
            if not workers:
                return None
            return min(workers, key=lambda w: w.inflight)

    def incr(self, w: Worker):
        with self.lock:
            w.inflight += 1

    def decr(self, w: Worker):
        with self.lock:
            w.inflight = max(0, w.inflight - 1)

    def choose_model(self) -> Tuple[str, Dict[str, Dict[str, Optional[float]]]]:
        """
        Heuristic model switching:
        Prefer ACCURATE_MODEL if:
          - enough samples
          - error rate acceptable
          - p{LAT_PCTL} latency <= DEADLINE_MS
        Else use FAST_MODEL.
        Includes cooldown to avoid flip-flopping.
        """
        a = ACCURATE_MODEL
        f = FAST_MODEL

        sa = self.stats.summary(a)
        sf = self.stats.summary(f)

        now = time.time()
        can_switch = (now - self.last_switch_at) >= SWITCH_COOLDOWN_SEC

        enough_a = (sa["n"] or 0.0) >= MIN_SAMPLES
        enough_f = (sf["n"] or 0.0) >= MIN_SAMPLES

        a_err = sa["err_rate"] if sa["err_rate"] is not None else 1.0
        f_err = sf["err_rate"] if sf["err_rate"] is not None else 1.0

        a_p = sa["pctl_ms"]
        f_p = sf["pctl_ms"]

        a_healthy = a_err <= 0.10
        f_healthy = f_err <= 0.20

        accurate_meets_deadline = bool(enough_a and a_healthy and (a_p is not None) and (a_p <= DEADLINE_MS))

        desired = a if accurate_meets_deadline else f

        # If fast has no samples yet but accurate does and is healthy, allow accurate
        if (not enough_f) and enough_a and a_healthy:
            desired = a

        # cooldown hysteresis
        if not can_switch and desired != self.last_choice:
            desired = self.last_choice

        if desired != self.last_choice:
            self.last_choice = desired
            self.last_switch_at = now

        return desired, {"accurate": sa, "fast": sf}


pm = PoolManager()


@app.get("/")
def index():
    # Keep single-file UI outside python if you want
    return FileResponse("web/index.html")


@app.post("/predict")
async def predict(files: List[UploadFile] = File(...)):
    pm.refresh()

    results = []
    for f in files:
        img = await f.read()

        chosen_model, policy_stats = pm.choose_model()

        pm.refresh()
        worker = pm.pick_worker(chosen_model)
        if worker is None:
            return JSONResponse(
                status_code=503,
                content={"error": f"No backends available for {chosen_model}. Check services/endpoints."}
            )

        pm.incr(worker)

        t0 = time.perf_counter()
        ok = True
        try:
            resp = worker.stub.Predict(pb.PredictRequest(image=img, filename=f.filename))
        except grpc.RpcError as e:
            ok = False
            router_ms = (time.perf_counter() - t0) * 1000.0
            pm.stats.add(chosen_model, router_ms=router_ms, ok=False)
            pm.decr(worker)
            return JSONResponse(status_code=503, content={"error": f"Backend error: {e}"})
        finally:
            pm.decr(worker)

        router_ms = (time.perf_counter() - t0) * 1000.0
        pm.stats.add(chosen_model, router_ms=router_ms, ok=ok)

        pool_sz = pm.pool_size(chosen_model)
        scaled_up = pool_sz > 1

        pod_age_s = max(0, int(time.time()) - int(resp.started_at_unix))
        likely_new_pod = bool(scaled_up and pod_age_s < 120)

        results.append({
            "filename": f.filename,
            "selected_model": chosen_model,
            "policy": {
                "deadline_ms": DEADLINE_MS,
                "window_sec": WINDOW_SEC,
                "min_samples": MIN_SAMPLES,
                "cooldown_sec": SWITCH_COOLDOWN_SEC,
                "pctl": LAT_PCTL,
                "accurate_model": ACCURATE_MODEL,
                "fast_model": FAST_MODEL,
                "stats": policy_stats,  # already JSON-safe (no inf/NaN)
            },
            "backend_target": worker.target,
            "backend_pod": resp.pod_name,
            "backend_pod_ip": resp.pod_ip,
            "backend_pod_age_s": pod_age_s,
            "scaled_up_active": scaled_up,
            "likely_new_pod_used": likely_new_pod,
            "prediction_text": resp.text,
            "model_infer_ms": float(resp.infer_ms),
            "router_total_ms": float(router_ms),
        })

    return {"count": len(results), "results": results}


def main():
    pm.refresh()
    uvicorn.run(app, host="0.0.0.0", port=ROUTER_HTTP_PORT)


if __name__ == "__main__":
    main()
