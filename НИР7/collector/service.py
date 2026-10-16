"""Always-on data collection service for НИР7 (run on the machine that never sleeps).

One process does everything:
  * MQTT collector, continuously (2-hour chunk files, fsync every 5 s);
  * ONEmesh node-cache dump every 2 hours;
  * full ONEmesh snapshot (per-node history) once a day, in a background thread;
  * data/raw/status.json refreshed every minute (counters, last message time).

    python -m collector.service
"""
import argparse
import json
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from .collect_mqtt import MqttCollector, log
from .snapshot_onemesh import Api, snapshot_cache, snapshot_full

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-every-h", type=float, default=2)
    ap.add_argument("--full-every-h", type=float, default=24)
    ap.add_argument("--no-full-at-start", action="store_true")
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    col = MqttCollector(out_dir=RAW / "mqtt")
    col.start()
    log("service started")

    state = {"cache_ok": None, "full_ok": None, "full_running": False, "errors": 0}
    next_cache = time.time()
    next_full = time.time() + (args.full_every_h * 3600 if args.no_full_at_start else 0)

    def run_full():
        state["full_running"] = True
        try:
            snapshot_full(RAW / "onemesh", Api(), progress=log)
            state["full_ok"] = datetime.now(timezone.utc).isoformat()
        except Exception:
            state["errors"] += 1
            log("full snapshot failed:\n" + traceback.format_exc())
        finally:
            state["full_running"] = False

    try:
        while True:
            now = time.time()
            if now >= next_cache:
                next_cache = now + args.cache_every_h * 3600
                try:
                    out, nodes = snapshot_cache(RAW / "onemesh", Api())
                    state["cache_ok"] = datetime.now(timezone.utc).isoformat()
                    log(f"cache snapshot: {len(nodes)} nodes -> {out.name}")
                except Exception:
                    state["errors"] += 1
                    log("cache snapshot failed:\n" + traceback.format_exc())
            if now >= next_full and not state["full_running"]:
                next_full = now + args.full_every_h * 3600
                threading.Thread(target=run_full, daemon=True).start()

            col.writer.flush()
            status = {
                "updated": datetime.now(timezone.utc).isoformat(),
                "mqtt": dict(col.stats.most_common()),
                "last_mqtt_message": (datetime.fromtimestamp(col.last_message_at, timezone.utc).isoformat()
                                      if col.last_message_at else None),
                **state,
            }
            tmp = RAW / "status.json.part"
            tmp.write_text(json.dumps(status, ensure_ascii=False, indent=1), encoding="utf-8")
            tmp.replace(RAW / "status.json")
            time.sleep(60)
    except KeyboardInterrupt:
        pass
    finally:
        col.stop()
        log("service stopped", dict(col.stats))


if __name__ == "__main__":
    main()
