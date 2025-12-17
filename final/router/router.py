import os, time, socket, threading, math, csv
from datetime import datetime
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
DEADLINE_MS = float(os.getenv("DEADLINE_MS", "3000"))
WINDOW_SEC = float(os.getenv("WINDOW_SEC", "20"))
MIN_SAMPLES = int(os.getenv("MIN_SAMPLES", "5"))  # warmup REQUESTS per model
SWITCH_COOLDOWN_SEC = float(os.getenv("SWITCH_COOLDOWN_SEC", "3"))
LAT_PCTL = float(os.getenv("LAT_PCTL", "95"))

ACCURATE_MODEL = os.getenv("ACCURATE_MODEL", "model2")
FAST_MODEL = os.getenv("FAST_MODEL", "model1")

# Monitoring log
MONITOR_CSV_PATH = os.getenv("MONITOR_CSV_PATH", "router_monitor.csv")
MONITOR_SORT_BY = os.getenv("MONITOR_SORT_BY", "router_total_ms")  # router_total_ms | ts
MONITOR_LOCK = threading.Lock()

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
    Stores PER-IMAGE latency samples (ms) per model in a rolling time window.
    We then use pctl_ms_per_img * num_images to predict request tail latency.
    """

    def __init__(self, window_sec: float, models: List[str]):
        self.window_sec = window_sec
        self.data: Dict[str, Deque[Tuple[float, float, bool]]] = {m: deque() for m in models}

    def _prune(self, model: str, now: float):
        dq = self.data.get(model)
        if dq is None:
            return
        cutoff = now - self.window_sec
        while dq and dq[0][0] < cutoff:
            dq.popleft()

    def add(self, model: str, per_img_ms: float, ok: bool):
        now = time.time()
        if model not in self.data:
            self.data[model] = deque()
        self.data[model].append((now, float(per_img_ms), bool(ok)))
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
            "pctl_ms": json_safe_float(pctl_ms),      # NOTE: per-image pctl
            "err_rate": json_safe_float(err_rate),
        }


class PoolManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.pools: Dict[str, Dict[str, Worker]] = {"model1": {}, "model2": {}}
        self.last_refresh = 0.0

        self.models = sorted({ACCURATE_MODEL, FAST_MODEL})
        self.stats = RollingStats(window_sec=WINDOW_SEC, models=self.models)

        # hysteresis
        self.last_choice = ACCURATE_MODEL
        self.last_switch_at = 0.0

        # warmup (by REQUEST count)
        self.warmup_counts: Dict[str, int] = {ACCURATE_MODEL: 0, FAST_MODEL: 0}

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

    def choose_model(self, num_images: int) -> Tuple[str, Dict[str, Dict[str, Optional[float]]]]:

        a = ACCURATE_MODEL
        f = FAST_MODEL
        k = max(1, int(num_images))

        sa = self.stats.summary(a)  # per-image windowed stats
        sf = self.stats.summary(f)

        now = time.time()
        can_switch = (now - self.last_switch_at) >= SWITCH_COOLDOWN_SEC

        # Health (windowed)
        a_err = sa["err_rate"] if sa.get("err_rate") is not None else 1.0
        f_err = sf["err_rate"] if sf.get("err_rate") is not None else 1.0
        a_healthy = a_err <= 0.10
        f_healthy = f_err <= 0.10

        # Tail per-image latency
        a_p_img = sa.get("pctl_ms")
        f_p_img = sf.get("pctl_ms")

        # Predict tail total latency for this request size
        a_pred = (a_p_img * k) if (a_p_img is not None) else None
        f_pred = (f_p_img * k) if (f_p_img is not None) else None

        accurate_meets_deadline = bool(
            a_healthy and (a_pred is not None) and (a_pred <= DEADLINE_MS)
        )

        warm_a = int(self.warmup_counts.get(a, 0))
        warm_f = int(self.warmup_counts.get(f, 0))

        # Warmup phases (by request count)
        if warm_a < MIN_SAMPLES:
            desired = a
        elif warm_f < MIN_SAMPLES:
            desired = f
        else:
            na = int(sa.get("n") or 0)
            nf = int(sf.get("n") or 0)

            if na == 0 and nf == 0:
                desired = a
            elif k > 6:
                desired = f
            else:
                desired = a if accurate_meets_deadline else f

                if desired == a and (not a_healthy) and f_healthy:
                    desired = f
                elif desired == f and (not f_healthy) and a_healthy:
                    desired = a


        # Cooldown hysteresis
        if (not can_switch) and desired != self.last_choice:
            desired = self.last_choice

        if desired != self.last_choice:
            self.last_choice = desired
            self.last_switch_at = now

        # Add useful debug fields for UI
        sa2 = dict(sa)
        sf2 = dict(sf)
        sa2["pred_ms_for_req"] = json_safe_float(a_pred)
        sf2["pred_ms_for_req"] = json_safe_float(f_pred)
        sa2["warmup_req_n"] = float(warm_a)
        sf2["warmup_req_n"] = float(warm_f)
        sa2["req_images"] = float(k)
        sf2["req_images"] = float(k)

        return desired, {"accurate": sa2, "fast": sf2}


pm = PoolManager()


def append_monitor_row(row: Dict[str, object]) -> None:
    fieldnames = ["ts", "num_images", "selected_model", "router_total_ms"]

    with MONITOR_LOCK:
        file_exists = os.path.exists(MONITOR_CSV_PATH)
        with open(MONITOR_CSV_PATH, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                w.writeheader()

            safe_row = {k: row.get(k) for k in fieldnames}
            w.writerow(safe_row)


def read_monitor_rows() -> List[Dict[str, object]]:
    if not os.path.exists(MONITOR_CSV_PATH):
        return []

    with MONITOR_LOCK:
        with open(MONITOR_CSV_PATH, "r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

    def to_int(x, default=0):
        try:
            return int(float(x))
        except Exception:
            return default

    def to_float(x, default=0.0):
        try:
            return float(x)
        except Exception:
            return default

    out: List[Dict[str, object]] = []
    for r in rows:
        rr: Dict[str, object] = dict(r)
        rr["num_images"] = to_int(r.get("num_images"))
        rr["router_total_ms"] = to_float(r.get("router_total_ms"))
        out.append(rr)

    return out


@app.get("/")
def index():
    return FileResponse("web/index.html")


@app.get("/monitor/ui")
def monitor_ui():
    return FileResponse("web/monitor.html")


@app.get("/monitor")
def monitor(limit: int = 300, sort_by: str = ""):
    rows = read_monitor_rows()
    if not rows:
        return {"count": 0, "rows": [], "csv_path": MONITOR_CSV_PATH}

    key = (sort_by or MONITOR_SORT_BY).strip()
    allowed = {"ts", "router_total_ms", "num_images", "selected_model"}
    if key not in allowed:
        key = "router_total_ms"

    rows.sort(
        key=lambda x: x.get(key, 0) if x.get(key, 0) is not None else 0,
        reverse=True,
    )

    if limit > 0:
        rows = rows[:limit]

    return {"count": len(rows), "sort_by": key, "csv_path": MONITOR_CSV_PATH, "rows": rows}


@app.post("/predict")
async def predict(files: List[UploadFile] = File(...)):
    pm.refresh()

    req_t0 = time.perf_counter()
    num_images = len(files)

    # Choose ONCE per request (batch-aware)
    chosen_model, policy_stats = pm.choose_model(num_images=num_images)

    # Pick backend once per request as well (keeps all images consistent)
    pm.refresh()
    worker = pm.pick_worker(chosen_model)
    if worker is None:
        return JSONResponse(
            status_code=503,
            content={"error": f"No backends available for {chosen_model}. Check services/endpoints."},
        )

    results = []
    ok_req = True

    # Run each image, collect per-image stats
    for f in files:
        img = await f.read()

        pm.incr(worker)
        t0 = time.perf_counter()
        ok_img = True
        try:
            resp = worker.stub.Predict(pb.PredictRequest(image=img, filename=f.filename))
        except grpc.RpcError as e:
            ok_img = False
            ok_req = False
            per_img_ms = (time.perf_counter() - t0) * 1000.0
            pm.stats.add(chosen_model, per_img_ms=per_img_ms, ok=False)
            pm.decr(worker)
            return JSONResponse(status_code=503, content={"error": f"Backend error: {e}"})
        finally:
            pm.decr(worker)

        per_img_ms = (time.perf_counter() - t0) * 1000.0
        pm.stats.add(chosen_model, per_img_ms=per_img_ms, ok=ok_img)

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
                "stats": policy_stats,
            },
            "backend_target": worker.target,
            "backend_pod": resp.pod_name,
            "backend_pod_ip": resp.pod_ip,
            "backend_pod_age_s": pod_age_s,
            "scaled_up_active": scaled_up,
            "likely_new_pod_used": likely_new_pod,
            "prediction_text": resp.text,
            "model_infer_ms": float(resp.infer_ms),
            "router_total_ms": float(per_img_ms),  # per-image router time
        })

    # Count warmup by REQUESTS (one increment per /predict call)
    pm.warmup_counts[chosen_model] = int(pm.warmup_counts.get(chosen_model, 0)) + 1

    req_ms = (time.perf_counter() - req_t0) * 1000.0

    # Log one row per request (batch upload)
    append_monitor_row({
        "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "num_images": num_images,
        "selected_model": chosen_model,
        "router_total_ms": float(req_ms),
    })

    return {"count": len(results), "selected_model": chosen_model, "request_total_ms": float(req_ms), "results": results}


def main():
    pm.refresh()
    uvicorn.run(app, host="0.0.0.0", port=ROUTER_HTTP_PORT)


if __name__ == "__main__":
    main()
