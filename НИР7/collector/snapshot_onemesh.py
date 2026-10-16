"""Snapshots of the ONEmesh map (map.onemesh.ru) for one city topic.

* cache snapshot (cheap, 3 requests): node cache with hardware, role, firmware,
  position, last telemetry and neighbour lists -> data/raw/onemesh/cache/<ts>/
* full snapshot (~1 h for Moscow): additionally, per node, the history the map keeps:
  traceroutes with per-hop SNR in both directions, neighbour reports, device metrics,
  position history -> data/raw/onemesh/full/<ts>/ ; the file DONE marks a finished one.
  An interrupted full snapshot can be resumed (nodes already saved are skipped).

Polite: one request at a time with a pause between requests.

    python -m collector.snapshot_onemesh --mode cache
    python -m collector.snapshot_onemesh --mode full
"""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "data" / "raw" / "onemesh"

BASE = "https://map.onemesh.ru"
TOPIC = "msh/RU/MSK"
CACHE_NAME = "msk"  # cache/nodes_<name>_optimized_v2.json

# Column order of cache/nodes_*_optimized_v2.json (taken from the map's front-end code).
NODE_COLUMNS = [
    "id", "node_id", "long_name", "short_name", "hardware_model", "role", "firmware_version",
    "region", "modem_preset", "has_default_channel", "position_precision", "num_online_local_nodes",
    "latitude", "longitude", "fixed_latitude", "fixed_longitude", "altitude", "position_updated_at",
    "battery_level", "voltage", "channel_utilization", "air_util_tx", "uptime_seconds", "temperature",
    "relative_humidity", "barometric_pressure", "neighbours", "neighbours_updated_at",
    "mqtt_connection_state_updated_at", "created_at", "updated_at", "gateway_id", "node_type",
    "root_topic", "public_key", "username", "radiation", "default_channel_id",
]
PRIVATE_COLUMNS = {"long_name", "short_name", "public_key", "username"}


class Api:
    def __init__(self, base=BASE, pause=0.5):
        self.base, self.pause = base, pause
        self.s = requests.Session()
        self.s.headers["User-Agent"] = "NIR7-research-snapshot (MISIS student project)"

    def get(self, path, **params):
        for attempt in range(4):
            try:
                r = self.s.get(self.base + path, params=params, timeout=60)
                if r.status_code == 200:
                    return r.json()
                if r.status_code in (429, 502, 503, 504):
                    time.sleep(5 * (attempt + 1))
                    continue
                return {"_http_status": r.status_code}
            except (requests.RequestException, ValueError):
                time.sleep(5 * (attempt + 1))
            finally:
                time.sleep(self.pause)
        return {"_error": "gave up"}


def _dump(obj, path):
    tmp = path.with_suffix(path.suffix + ".part")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False)
    tmp.replace(path)


def _stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")


def stamp_time(path):
    """Unix time encoded in a snapshot directory name (YYYYMMDD_HHMM, UTC)."""
    return datetime.strptime(Path(path).name, "%Y%m%d_%H%M").replace(tzinfo=timezone.utc).timestamp()


def latest_snapshot(out_root, kind):
    """Newest snapshot directory of `kind` ("cache" or "full"), or None."""
    dirs = sorted(p for p in (Path(out_root) / kind).glob("????????_????") if p.is_dir())
    return dirs[-1] if dirs else None


def snapshot_cache(out_root=DEFAULT_OUT, api=None, topic=TOPIC, cache_name=CACHE_NAME):
    """Node cache + topics + hardware stats. Returns (out_dir, nodes)."""
    api = api or Api()
    rows = api.get(f"/cache/nodes_{cache_name}_optimized_v2.json")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"node cache unavailable: {str(rows)[:200]}")
    nodes = [{k: v for k, v in zip(NODE_COLUMNS, row) if k not in PRIVATE_COLUMNS} for row in rows]
    out = Path(out_root) / "cache" / _stamp()
    out.mkdir(parents=True, exist_ok=True)
    _dump(api.get("/api/v1/topics", show_all="true"), out / "topics.json")
    _dump(nodes, out / "nodes.json")
    _dump(api.get("/api/v1/stats/hardware-models", root_topic=topic), out / "hardware_models.json")
    return out, nodes


def snapshot_full(out_root=DEFAULT_OUT, api=None, topic=TOPIC, cache_name=CACHE_NAME,
                  max_age_days=30, history_days=30, traceroutes=200, progress=print, resume=None,
                  max_failed_in_row=20):
    """Cache snapshot plus per-node history files. Returns out_dir.

    resume: an unfinished full/<ts> directory to complete instead of starting a new one.
    A node whose requests all gave up is not saved (a resumed run retries it); after
    `max_failed_in_row` such nodes in a row the site is considered down and we raise.
    """
    api = api or Api()
    if resume:
        out = Path(resume)
        nodes = json.loads((out / "nodes.json").read_text(encoding="utf-8"))
        now = datetime.fromtimestamp(stamp_time(out), timezone.utc)
    else:
        cache_dir, nodes = snapshot_cache(out_root, api, topic, cache_name)
        out = Path(out_root) / "full" / cache_dir.name
        (out / "nodes").mkdir(parents=True, exist_ok=True)
        for name in ("topics.json", "nodes.json", "hardware_models.json"):
            (out / name).write_bytes((cache_dir / name).read_bytes())
        now = datetime.now(timezone.utc)

    time_from = int((now.timestamp() - history_days * 86400) * 1000)
    fresh = [n for n in nodes if n["updated_at"] and
             (now - datetime.fromisoformat(n["updated_at"].replace("Z", "+00:00"))).days <= max_age_days]
    progress(f"full snapshot{' (resumed)' if resume else ''}: {len(fresh)}/{len(nodes)} nodes -> {out}")
    saved = failed = in_row = 0
    for i, n in enumerate(fresh, 1):
        nid = n["node_id"]
        if (out / "nodes" / f"{nid}.json").exists():
            saved += 1
            continue
        rec = {
            "node_id": nid,
            "traceroutes": api.get(f"/api/v1/nodes/{nid}/traceroutes", root_topic=topic, count=traceroutes),
            "neighbours": api.get(f"/api/v1/nodes/{nid}/neighbours", root_topic=topic),
        }
        if n["channel_utilization"] is not None or n["voltage"] is not None:
            rec["device_metrics"] = api.get(f"/api/v1/nodes/{nid}/device-metrics",
                                            root_topic=topic, time_from=time_from)
        if n["latitude"] is not None:
            rec["position_history"] = api.get(f"/api/v1/nodes/{nid}/position-history", root_topic=topic,
                                              time_from=time_from, time_to=int(now.timestamp() * 1000))
        if all(isinstance(v, dict) and "_error" in v for k, v in rec.items() if k != "node_id"):
            failed += 1
            in_row += 1
            if in_row >= max_failed_in_row:
                raise RuntimeError(f"ONEmesh API unavailable: {in_row} nodes in a row failed")
            continue
        in_row = 0
        _dump(rec, out / "nodes" / f"{nid}.json")
        saved += 1
        if i % 100 == 0:
            progress(f"full snapshot: {i}/{len(fresh)}")
    _dump({"finished": datetime.now(timezone.utc).isoformat(), "nodes": len(fresh),
           "saved": saved, "failed": failed}, out / "DONE")
    progress(f"full snapshot done: {saved}/{len(fresh)} nodes saved, {failed} failed")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("cache", "full"), default="full")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--pause", type=float, default=0.5, help="seconds between requests")
    args = ap.parse_args()
    api = Api(pause=args.pause)
    if args.mode == "cache":
        out, nodes = snapshot_cache(args.out, api)
        print(f"{len(nodes)} nodes -> {out}")
    else:
        snapshot_full(args.out, api)


if __name__ == "__main__":
    main()
