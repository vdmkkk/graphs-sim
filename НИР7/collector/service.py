"""Always-on data collection service for НИР7 (run on the machine that never sleeps).

One process does everything:
  * MQTT collector, continuously (2-hour chunk files, fsync every 5 s);
  * ONEmesh node-cache dump every 2 hours;
  * full ONEmesh snapshot (per-node history) once a day, in a background thread;
  * data/raw/status.json refreshed every minute (counters, last message time).

Restart-safe: the schedule continues from the snapshots already on disk, an interrupted
full snapshot is resumed, and a second instance refuses to start (records would double).
If no MQTT message arrives for --stall-min minutes the process exits with code 1, so the
supervisor (systemd / run_service.*) restarts it.

    python -m collector.service
"""
import argparse
import json
import os
import signal
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from .collect_mqtt import MqttCollector, log
from .snapshot_onemesh import Api, latest_snapshot, snapshot_cache, snapshot_full, stamp_time

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
TICK = 5  # s: flush/fsync period of the MQTT file


def _single_instance(path):
    """Hold an exclusive lock on `path` for the life of the process, or exit."""
    fh = open(path, "a+")
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        log(f"another collector already holds {path}, exiting")
        sys.exit(2)
    return fh


def _iso(ts):
    return datetime.fromtimestamp(ts, timezone.utc).isoformat() if ts else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-every-h", type=float, default=2)
    ap.add_argument("--full-every-h", type=float, default=24)
    ap.add_argument("--no-full-at-start", action="store_true",
                    help="do not start a new full snapshot right away (an unfinished one is still resumed)")
    ap.add_argument("--stall-min", type=float, default=10, help="exit if MQTT is silent this long")
    args = ap.parse_args()
    cache_period, full_period = args.cache_every_h * 3600, args.full_every_h * 3600

    RAW.mkdir(parents=True, exist_ok=True)
    lock = _single_instance(RAW / "service.lock")  # noqa: F841 (kept open on purpose)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))  # systemctl stop -> run `finally`

    col = MqttCollector(out_dir=RAW / "mqtt")
    col.start()
    started = time.time()
    onemesh = RAW / "onemesh"

    # Continue the schedule from what is already on disk.
    sched = {"cache": 0.0, "full": 0.0, "resume": None}
    last = latest_snapshot(onemesh, "cache")
    if last:
        sched["cache"] = stamp_time(last) + cache_period
    last = latest_snapshot(onemesh, "full")
    if last and (last / "DONE").exists():
        sched["full"] = stamp_time(last) + full_period
    elif last and (last / "nodes.json").exists() and started - stamp_time(last) < full_period:
        sched["resume"] = last
    elif args.no_full_at_start:
        sched["full"] = started + full_period
    log("service started; next cache", _iso(max(sched["cache"], started)),
        "| next full", "resume " + last.name if sched["resume"] else _iso(max(sched["full"], started)))

    state = {"started": _iso(started), "cache_ok": None, "full_ok": None, "full_running": False,
             "full_progress": None, "errors": 0}

    def progress(msg):
        log(msg)
        state["full_progress"] = msg

    def run_full(resume):
        try:
            out = snapshot_full(onemesh, Api(), progress=progress, resume=resume)
            state["full_ok"] = datetime.now(timezone.utc).isoformat()
            state["full_last"] = out.name
        except Exception:
            state["errors"] += 1
            log("full snapshot failed, will resume in 1 h:\n" + traceback.format_exc())
            unfinished = latest_snapshot(onemesh, "full")
            if unfinished and (unfinished / "nodes.json").exists() and not (unfinished / "DONE").exists():
                sched["resume"] = unfinished
            sched["full"] = time.time() + 3600
        finally:
            state["full_running"] = False

    exit_code = 0
    next_status = 0.0
    try:
        while True:
            now = time.time()
            if now >= sched["cache"]:
                sched["cache"] = now + cache_period
                try:
                    out, nodes = snapshot_cache(onemesh, Api())
                    state["cache_ok"] = datetime.now(timezone.utc).isoformat()
                    log(f"cache snapshot: {len(nodes)} nodes -> {out.name}")
                except Exception:
                    state["errors"] += 1
                    sched["cache"] = now + 600
                    log("cache snapshot failed, retry in 10 min:\n" + traceback.format_exc())
            if now >= sched["full"] and not state["full_running"]:
                resume, sched["resume"] = sched["resume"], None
                sched["full"] = (stamp_time(resume) if resume else now) + full_period
                state["full_running"] = True
                threading.Thread(target=run_full, args=(resume,), daemon=True).start()

            col.writer.flush()
            if now >= next_status:
                next_status = now + 60
                status = {
                    "updated": datetime.now(timezone.utc).isoformat(),
                    "mqtt": dict(col.stats.most_common()),
                    "last_mqtt_message": _iso(col.last_message_at),
                    **state,
                }
                tmp = RAW / "status.json.part"
                tmp.write_text(json.dumps(status, ensure_ascii=False, indent=1), encoding="utf-8")
                tmp.replace(RAW / "status.json")
                silent = now - (col.last_message_at or started)
                if silent > args.stall_min * 60:
                    log(f"no MQTT messages for {silent / 60:.0f} min, exiting for a restart")
                    exit_code = 1
                    break
            time.sleep(TICK)
    except KeyboardInterrupt:
        pass
    finally:
        col.stop()
        log("service stopped", dict(col.stats))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
