"""Read-only MQTT collector for a city segment of the Meshtastic network.

Every received ServiceEnvelope becomes one JSON line in a 2-hour chunk file
data/raw/mqtt/YYYY-MM-DD/HH.jsonl (UTC, HH = 00, 02, ..., 22). Lines are flushed and
fsync'ed every few seconds, so a crash loses seconds, and a damaged file at most two
hours. Never publishes anything. Reconnects automatically.

    python -m collector.collect_mqtt                      # Moscow, ONEmesh broker
    python -m collector.collect_mqtt --duration 60        # short test run
"""
import argparse
import json
import os
import ssl
import sys
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import paho.mqtt.client as mqtt

from .meshtastic_decode import decode_envelope

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "data" / "raw" / "mqtt"

BROKER = dict(
    host="mqtt.onemesh.ru",
    port=8883,
    # ONEmesh: "onemesh" gets no downlink at all, "onemeshz" has hop_limit zeroed by the
    # server; "onemeshd" delivers unmodified headers. We only subscribe, never publish.
    user="onemeshd",
    password="onecat",
    topic="msh/RU/MSK/#",
)


def log(*parts):
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S}", *parts, flush=True)


class ChunkedJsonlWriter:
    """Append-only JSONL writer that starts a new file every `chunk_hours` (UTC)."""

    def __init__(self, out_dir, chunk_hours=2):
        self.out_dir = Path(out_dir)
        self.chunk_hours = chunk_hours
        self._key, self._fh = None, None
        self._lock = threading.Lock()

    def _path_for(self, now):
        hh = now.hour - now.hour % self.chunk_hours
        return self.out_dir / now.strftime("%Y-%m-%d") / f"{hh:02d}.jsonl"

    def write(self, rec):
        path = self._path_for(datetime.now(timezone.utc))
        line = json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n"
        with self._lock:
            if path != self._key:
                self._close()
                path.parent.mkdir(parents=True, exist_ok=True)
                self._fh, self._key = open(path, "a", encoding="utf-8"), path
            self._fh.write(line)

    def flush(self):
        with self._lock:
            if self._fh:
                self._fh.flush()
                os.fsync(self._fh.fileno())

    def _close(self):
        if self._fh:
            self._fh.flush()
            os.fsync(self._fh.fileno())
            self._fh.close()
            self._fh = None

    def close(self):
        with self._lock:
            self._close()


class MqttCollector:
    def __init__(self, out_dir=DEFAULT_OUT, host=BROKER["host"], port=BROKER["port"], tls=True,
                 user=BROKER["user"], password=BROKER["password"], topic=BROKER["topic"]):
        self.writer = ChunkedJsonlWriter(out_dir)
        self.topic = topic
        self.stats = Counter()
        self.last_message_at = None
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                                  client_id=f"nir7-research-{int(time.time())}", clean_session=True)
        self.client.username_pw_set(user, password)
        if tls:
            self.client.tls_set(cert_reqs=ssl.CERT_REQUIRED)
        self.client.reconnect_delay_set(min_delay=2, max_delay=120)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message
        self._host, self._port = host, port

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        log("mqtt connected", reason_code)
        client.subscribe(self.topic, qos=0)

    def _on_disconnect(self, client, userdata, flags, reason_code, properties):
        log("mqtt disconnected", reason_code)

    def _on_message(self, client, userdata, msg):
        self.stats["mqtt"] += 1
        self.last_message_at = time.time()
        try:
            rec = decode_envelope(msg.topic, msg.payload)
        except Exception as exc:  # one bad packet must not stop the collection
            self.stats["error"] += 1
            print("decode error:", repr(exc), file=sys.stderr, flush=True)
            return
        if rec is None:
            self.stats["non_envelope"] += 1
            return
        rec["t"] = round(time.time(), 3)
        self.writer.write(rec)
        self.stats["saved"] += 1
        self.stats[rec.get("portnum", rec.get("enc"))] += 1

    def start(self):
        self.client.connect_async(self._host, self._port, keepalive=60)
        self.client.loop_start()

    def stop(self):
        self.client.loop_stop()
        self.client.disconnect()
        self.writer.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--topic", default=BROKER["topic"])
    ap.add_argument("--user", default=BROKER["user"])
    ap.add_argument("--report-every", type=int, default=600, help="status line period, s")
    ap.add_argument("--duration", type=float, default=0, help="stop after N seconds (0 = forever)")
    args = ap.parse_args()

    col = MqttCollector(out_dir=args.out, topic=args.topic, user=args.user)
    col.start()
    started = last = time.time()
    try:
        while not args.duration or time.time() - started < args.duration:
            time.sleep(5)
            col.writer.flush()
            if time.time() - last >= args.report_every:
                last = time.time()
                log(", ".join(f"{k}={v}" for k, v in col.stats.most_common(8)))
    except KeyboardInterrupt:
        pass
    finally:
        col.stop()
        log("final:", dict(col.stats))


if __name__ == "__main__":
    main()
