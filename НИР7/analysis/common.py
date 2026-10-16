"""Shared loaders and helpers for the НИР7 analysis.

Sources (see docs/данные.md):
  * the newest finished full ONEmesh snapshot (data/raw/onemesh/full/<ts>/ with DONE);
  * MQTT records data/raw/mqtt/*/*.jsonl;
  * terrain height (SRTM GL1 30 m via OpenTopoData, cached in data/env/) for height above ground.
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from meshtastic.protobuf import config_pb2, mesh_pb2

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
ENV_DIR = ROOT / "data" / "env"
RESULTS = ROOT / "results"

CENTER = (55.7520, 37.6175)  # Kremlin: distance to it is the urban-density proxy
BROADCAST = 0xFFFFFFFF
MAX_LINK_KM = 100.0  # longer "direct" links on 868 MHz ground paths are stale/wrong coordinates
UNKNOWN_SNR = -32.0  # int8 -128 / 4: "SNR unknown" in RouteDiscovery

# Thermal noise in 250 kHz (MEDIUM_FAST): -174 + 10 lg(250e3) = -120.0 dBm, plus receiver NF.
THERMAL_250K = -174 + 10 * np.log10(250e3)

# Antenna/form-factor classes for Г3. Pocket = enclosed handhelds with built-in antennas.
POCKET = {"HELTEC_MESH_POCKET", "TRACKER_T1000_E", "T_ECHO", "T_ECHO_PLUS", "WISMESH_TAG", "THINKNODE_M1"}
BOARD = {"HELTEC_V3", "HELTEC_V4", "HELTEC_MESH_NODE_T114", "HELTEC_WSL_V3", "NRF52_PROMICRO_DIY",
         "TBEAM", "TBEAM_1_WATT", "TLORA_V2_1_1P6", "TLORA_T3_S3", "DIY_V1", "RAK4631",
         "EBYTE_ESP32_S3", "SEEED_XIAO_S3", "HELTEC_WIRELESS_PAPER", "HELTEC_VISION_MASTER_E213",
         "HELTEC_VISION_MASTER_E290", "HELTEC_WIRELESS_TRACKER"}
# Models with enough nodes to get their own coefficient; the rest -> OTHER.
MAIN_MODELS = ["HELTEC_V4", "HELTEC_V3", "HELTEC_MESH_NODE_T114", "HELTEC_MESH_POCKET",
               "NRF52_PROMICRO_DIY", "TBEAM", "T_ECHO"]
SHORT = {"HELTEC_V4": "Heltec V4", "HELTEC_V3": "Heltec V3", "HELTEC_MESH_NODE_T114": "T114",
         "HELTEC_MESH_POCKET": "MeshPocket", "NRF52_PROMICRO_DIY": "ProMicro DIY", "TBEAM": "T-Beam",
         "T_ECHO": "T-Echo", "OTHER": "прочие", "UNKNOWN": "неизв."}

# Position precision (bits) -> size of the truncation cell, m (Meshtastic position_precision).
def precision_m(bits):
    if bits is None or bits >= 32 or bits <= 0:
        return 0.0 if bits and bits >= 32 else np.nan
    return 23300.0 / 2 ** (bits - 10)  # 10 bits = 23.3 km, each bit halves the cell


def enum_name(enum, value):
    try:
        return enum.Name(int(value))
    except (ValueError, TypeError):
        return None


def anon(node_id):
    """Stable pseudonym for a node id (results are published, raw ids are not)."""
    return "N" + hashlib.sha256(f"nir7:{int(node_id)}".encode()).hexdigest()[:6]


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def latest_full():
    dirs = sorted(p for p in (RAW / "onemesh" / "full").glob("????????_????") if (p / "DONE").exists())
    if not dirs:
        raise SystemExit("no finished full snapshot (DONE) in data/raw/onemesh/full")
    return dirs[-1]


# ---------------------------------------------------------------- nodes

def dem_heights(lat, lon, dataset="srtm30m"):
    """Terrain (surface) height, m, for arrays of coordinates; NaN where unknown.

    SRTM GL1 30 m via the public OpenTopoData API (AWS S3 with Copernicus GLO-30 is not
    reachable from the collection machine). Answers are cached in data/env/ground_<dataset>.json,
    so reruns only ask for new coordinates (API limit: 100 points per call, 1 call/s, 1000/day).
    """
    import requests
    cache_file = ENV_DIR / f"ground_{dataset}.json"
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    keys = [f"{a:.5f},{b:.5f}" for a, b in zip(lat, lon)]
    todo = sorted(set(keys) - cache.keys())
    for i in range(0, len(todo), 100):
        chunk = todo[i:i + 100]
        r = requests.get(f"https://api.opentopodata.org/v1/{dataset}",
                         params={"locations": "|".join(chunk)}, timeout=60)
        r.raise_for_status()
        for k, res in zip(chunk, r.json()["results"]):
            cache[k] = res["elevation"]
        ENV_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(cache))
        time.sleep(1.1)
    return np.array([np.nan if cache.get(k) is None else cache[k] for k in keys], float)


def load_nodes(full_dir):
    raw = json.loads((full_dir / "nodes.json").read_text(encoding="utf-8"))
    df = pd.DataFrame(raw)
    df["node_id"] = df["node_id"].astype("int64")
    df = df.drop_duplicates("node_id", keep="last")
    df["hw"] = [enum_name(mesh_pb2.HardwareModel, v) or "UNKNOWN" for v in df["hardware_model"]]
    df["model"] = np.where(df["hw"].isin(MAIN_MODELS), df["hw"], "OTHER")
    df["hw_class"] = np.select([df["hw"].isin(POCKET), df["hw"].isin(BOARD)], ["pocket", "board"], "other")
    df["role_name"] = [enum_name(config_pb2.Config.DeviceConfig.Role, v) for v in df["role"]]
    df["region_name"] = [enum_name(config_pb2.Config.LoRaConfig.RegionCode, v) if v is not None else None
                         for v in df["region"]]
    df["lat"] = df["latitude"] / 1e7
    df["lon"] = df["longitude"] / 1e7
    df.loc[(df["lat"] == 0) & (df["lon"] == 0), ["lat", "lon"]] = np.nan
    df["pos_err_m"] = [precision_m(b) if pd.notna(la) else np.nan
                       for b, la in zip(df["position_precision"], df["lat"])]
    df["dist_center_km"] = haversine_km(df["lat"], df["lon"], *CENTER)
    has = df["lat"].notna().to_numpy()
    df["ground_m"] = np.nan
    df.loc[has, "ground_m"] = dem_heights(df.loc[has, "lat"].to_numpy(), df.loc[has, "lon"].to_numpy())
    alt = pd.to_numeric(df["altitude"], errors="coerce")
    hag = alt - df["ground_m"]
    # GPS/manual altitude is MSL with tens of metres of error; keep physically plausible values only.
    df["hag_m"] = hag.where((hag > -30) & (hag < 350))
    df = df.set_index("node_id", drop=False)
    # Mobility from the position history: the cache keeps only the latest position, while links
    # span the whole week, so nodes that moved get wrong distances.
    mob = position_spread(full_dir)
    df = df.join(mob, how="left")
    df["mobile"] = df["spread_km"] > np.maximum(1.0, 2 * df["pos_err_m"].fillna(0) / 1000)
    return df


def position_spread(full_dir):
    """Per node: number of positions in the map history and their max distance from the median."""
    rows = []
    for rec in iter_node_files(full_dir):
        ph = rec.get("position_history")
        ph = ph.get("position_history", []) if isinstance(ph, dict) else []
        la = np.array([p["latitude"] / 1e7 for p in ph if p.get("latitude") and p.get("longitude")])
        lo = np.array([p["longitude"] / 1e7 for p in ph if p.get("latitude") and p.get("longitude")])
        spread = haversine_km(la, lo, np.median(la), np.median(lo)).max() if len(la) else np.nan
        rows.append((int(rec["node_id"]), len(la), spread))
    return pd.DataFrame(rows, columns=["node_id", "n_pos", "spread_km"]).set_index("node_id")


# ---------------------------------------------------------------- per-node history (full snapshot)

def iter_node_files(full_dir):
    for f in sorted((full_dir / "nodes").glob("*.json")):
        yield json.loads(f.read_text(encoding="utf-8"))


def _ts(s):
    return pd.Timestamp(s).timestamp() if s else np.nan


def traceroute_links(full_dir):
    """Directed one-hop links with SNR from traceroute replies.

    Reply record: `to` = requester, `from` = responder. Forward chain
    [to] + route + [from] with snr_towards; return chain [from] + route_back
    (+ [to] when snr_back has one more element) with snr_back.
    Returns (links, routes): links tx, rx, snr, t, key; routes = relay lists per traceroute.
    """
    rows, routes, seen_routes = [], [], set()
    for rec in iter_node_files(full_dir):
        trs = rec.get("traceroutes")
        trs = trs.get("traceroutes", []) if isinstance(trs, dict) else []
        for tr in trs:
            req, resp = int(tr["to"]), int(tr["from"])
            pid, t = tr.get("packet_id"), _ts(tr.get("created_at"))
            route, back = tr.get("route") or [], tr.get("route_back") or []
            st, sb = tr.get("snr_towards") or [], tr.get("snr_back") or []
            fwd = [req] + route + [resp]
            for k, snr in enumerate(st[:len(fwd) - 1]):
                rows.append((fwd[k], fwd[k + 1], snr, t, f"{pid}:f:{k}"))
            bch = [resp] + back + ([req] if len(sb) == len(back) + 1 else [])
            for k, snr in enumerate(sb[:len(bch) - 1]):
                rows.append((bch[k], bch[k + 1], snr, t, f"{pid}:b:{k}"))
            if (pid, "f") not in seen_routes and len(st) == len(route) + 1:
                seen_routes.add((pid, "f"))
                routes.append({"packet_id": pid, "dir": "f", "src": req, "dst": resp, "relays": route})
            if len(sb) == len(back) + 1 and (pid, "b") not in seen_routes:  # complete return path
                seen_routes.add((pid, "b"))
                routes.append({"packet_id": pid, "dir": "b", "src": resp, "dst": req, "relays": back})
    df = pd.DataFrame(rows, columns=["tx", "rx", "snr", "t", "key"]).drop_duplicates("key")
    ok = ((df["tx"] != BROADCAST) & (df["rx"] != BROADCAST) & (df["tx"] != df["rx"])
          & (df["snr"] > UNKNOWN_SNR) & df["snr"].notna())
    return df[ok].assign(source="traceroute"), routes


def neighbour_links(full_dir):
    """Directed links from NeighborInfo: the reporting node heard `node_id` with `snr`."""
    rows = []
    for rec in iter_node_files(full_dir):
        nb = rec.get("neighbours")
        if not isinstance(nb, dict):
            continue
        me = int(rec["node_id"])
        for e in nb.get("nodes_that_we_heard", []) or []:
            rows.append((int(e["node_id"]), me, e.get("snr"), _ts(e.get("updated_at"))))
        for e in nb.get("nodes_that_heard_us", []) or []:
            rows.append((me, int(e["node_id"]), e.get("snr"), _ts(e.get("updated_at"))))
    df = pd.DataFrame(rows, columns=["tx", "rx", "snr", "t"]).dropna(subset=["snr"])
    df = df.drop_duplicates(["tx", "rx", "t"])
    df = df[(df["tx"] != df["rx"]) & (df["tx"] != BROADCAST)]
    return df.assign(source="neighbourinfo")


# ---------------------------------------------------------------- MQTT

MQTT_DTYPES = {"t": "float64", "gw": "int64", "from": "int64", "id": "int64", "hop_start": "int8",
               "hop_limit": "int8", "rx_snr": "float32", "rx_rssi": "int16", "relay_node": "int16"}


def _read_mqtt_file(path):
    cols = {k: [] for k in MQTT_DTYPES}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
                gw = int(r["gateway_id"][1:], 16)
            except (ValueError, KeyError, TypeError):
                continue  # a line cut by a crash or an envelope without a gateway
            cols["gw"].append(gw)
            for k in ("t", "from", "id", "hop_start", "hop_limit", "rx_snr", "rx_rssi", "relay_node"):
                cols[k].append(r.get(k) or 0)
    return pd.DataFrame({k: np.asarray(v, dtype=MQTT_DTYPES[k]) for k, v in cols.items()})


def load_mqtt(since=None):
    """All MQTT records (radio header only, compact dtypes), deduplicated by (gateway, from, id)."""
    frames = [_read_mqtt_file(f) for f in sorted((RAW / "mqtt").glob("*/*.jsonl"))]
    df = pd.concat(frames, ignore_index=True)
    if since is not None:
        df = df[df["t"] >= since]
    df = df.drop_duplicates(["gw", "from", "id"])
    df["self"] = df["gw"] == df["from"]
    df["hops"] = np.where(df["hop_start"] > 0, df["hop_start"] - df["hop_limit"], np.nan)
    df["radio_ok"] = ~((df["rx_snr"] == 0) & (df["rx_rssi"] == 0)) & ~df["self"]
    return df.reset_index(drop=True)


def noise_floor(rssi, snr):
    """Noise power at the receiver, dBm: RSSI is signal+noise, so N = RSSI - 10 lg(1 + 10^(SNR/10))."""
    return rssi - 10 * np.log10(1 + 10 ** (snr / 10))


def resolve_relays(mq, neighbours_of):
    """Map 1-byte relay_node to a full node id using the gateway's known direct neighbours.

    neighbours_of: dict gateway -> set of node ids linked to it directly (any source).
    Unique candidate among them -> resolved; otherwise NaN.
    """
    out = np.full(len(mq), np.nan)
    by_byte_cache = {}
    for i, (gw, rb) in enumerate(zip(mq["gw"].to_numpy(), mq["relay_node"].to_numpy())):
        if not rb:
            continue
        key = (gw, rb)
        if key not in by_byte_cache:
            cand = [n for n in neighbours_of.get(gw, ()) if n & 0xFF == rb]
            by_byte_cache[key] = cand[0] if len(cand) == 1 else np.nan
        out[i] = by_byte_cache[key]
    return out


def mqtt_links(mq, neighbours_of):
    """Directed last-hop links (tx -> gateway) with SNR and RSSI from MQTT.

    Zero-hop packets give tx = sender; multi-hop packets give tx = last relayer when its
    1-byte id is resolved uniquely among the gateway's direct neighbours.
    """
    m = mq[mq["radio_ok"]].copy()
    m["relay_full"] = resolve_relays(m, neighbours_of)
    zero = m["hops"] == 0
    m["tx"] = np.where(zero, m["from"], m["relay_full"])
    m["how"] = np.where(zero, "zero_hop", "relay")
    m = m[m["tx"].notna() & (m["tx"] != m["gw"])]
    return pd.DataFrame({"tx": m["tx"].astype("int64"), "rx": m["gw"], "snr": m["rx_snr"],
                         "rssi": m["rx_rssi"], "t": m["t"], "how": m["how"], "source": "mqtt"})


# ---------------------------------------------------------------- link table

def attach(g, nodes, min_pos_factor=3.0):
    """Add attributes of both ends and the geometry to a frame with tx, rx columns.

    dist_ok: both ends have coordinates, neither moved during the history window, the distance
    exceeds `min_pos_factor` x the combined coordinate truncation (and 300 m) and is <= MAX_LINK_KM.
    """
    g = g.copy()
    for side in ("tx", "rx"):
        a = nodes.reindex(g[side])
        for col in ("lat", "lon", "pos_err_m", "hag_m", "model", "hw_class", "hw", "dist_center_km",
                    "role_name", "region_name", "mobile"):
            g[f"{side}_{col}"] = a[col].to_numpy()
    g["d_km"] = haversine_km(g["tx_lat"], g["tx_lon"], g["rx_lat"], g["rx_lon"])
    err_km = (g["tx_pos_err_m"] + g["rx_pos_err_m"]) / 1000
    g["dist_ok"] = (g["d_km"].notna() & (g["d_km"] >= np.maximum(min_pos_factor * err_km, 0.3))
                    & (g["d_km"] <= MAX_LINK_KM) & ~g["tx_mobile"].eq(True) & ~g["rx_mobile"].eq(True))
    mid_lat, mid_lon = (g["tx_lat"] + g["rx_lat"]) / 2, (g["tx_lon"] + g["rx_lon"]) / 2
    g["mid_center_km"] = haversine_km(mid_lat, mid_lon, *CENTER)
    return g


def link_table(obs, nodes, min_pos_factor=3.0):
    """Aggregate observations to directed links (median SNR/RSSI) and attach geometry (see attach)."""
    agg = {"snr": "median", "t": "size"}
    if "rssi" in obs:
        agg["rssi"] = "median"
    g = obs.groupby(["tx", "rx"]).agg(agg).rename(columns={"t": "n_obs"}).reset_index()
    return attach(g, nodes, min_pos_factor)


def mqtt_packets(mq, nodes):
    """Per (sender, gateway, packet): was it received directly (zero hops) and with what SNR.

    A gateway uploads the first copy it decodes; the original transmission precedes every
    rebroadcast, so an upload with hops > 0 means the direct copy was not decoded. This gives
    the non-detections that opportunistic link lists lack.
    """
    m = mq[~mq["self"] & mq["hops"].notna() & (mq["hops"] >= 0)].copy()
    m["direct"] = m["hops"] == 0
    m = m[~(m["direct"] & ~m["radio_ok"])]  # direct, but the gateway reported no SNR
    m = m.rename(columns={"from": "tx", "gw": "rx", "rx_snr": "snr", "rx_rssi": "rssi"})
    return attach(m[["tx", "rx", "t", "direct", "snr", "rssi", "hops"]], nodes)


def zone(dist_center_km):
    return pd.cut(dist_center_km, [0, 7, 15, np.inf], labels=["центр (<7 км)", "середина (7–15 км)",
                                                              "периферия (>15 км)"], right=False)
