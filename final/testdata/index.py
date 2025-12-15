#!/usr/bin/env python3
import json
import urllib.request
from pathlib import Path

DATASET = "lansinuote/ocr_id_card"
CONFIG = "default"
SPLIT = "train"
OFFSET = 0
LENGTH = 30  # max 100

OUT_DIR = Path("ocr_id_card_30")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ROWS_URL = (
    "https://datasets-server.huggingface.co/rows"
    f"?dataset={DATASET}&config={CONFIG}&split={SPLIT}&offset={OFFSET}&length={LENGTH}"
)

UA_HDRS = {"User-Agent": "python-stdlib"}

def http_get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers=UA_HDRS)
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read().decode("utf-8"))

def download(url: str, out_path: Path):
    req = urllib.request.Request(url, headers=UA_HDRS)
    with urllib.request.urlopen(req) as r, open(out_path, "wb") as f:
        f.write(r.read())

def main():
    data = http_get_json(ROWS_URL)

    # Find which column is the Image feature (don’t assume it's named "image")
    image_cols = []
    for feat in data.get("features", []):
        t = feat.get("type", {})
        if isinstance(t, dict) and t.get("_type") == "Image":
            image_cols.append(feat["name"])

    if not image_cols:
        raise SystemExit("No Image column found in /rows response.")
    img_col = image_cols[0]

    rows = data.get("rows", [])
    if not rows:
        raise SystemExit("No rows returned. Try changing OFFSET or CONFIG/SPLIT.")

    for i, item in enumerate(rows):
        row = item.get("row", {})
        img_obj = row.get(img_col)

        # Image objects have {"src": "...signed url...", "width":..., "height":...} :contentReference[oaicite:2]{index=2}
        if not isinstance(img_obj, dict) or "src" not in img_obj:
            print(f"[skip] row {item.get('row_idx')} missing image")
            continue

        img_url = img_obj["src"]  # signed URL (expires), so download immediately :contentReference[oaicite:3]{index=3}
        out_path = OUT_DIR / f"img_{i:04d}.jpg"

        print(f"[{i+1:02d}/{len(rows)}] downloading -> {out_path.name}")
        download(img_url, out_path)

    print(f"\nDone. Saved into: {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
