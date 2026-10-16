"""Decoding of Meshtastic MQTT ServiceEnvelopes into flat, privacy-reduced records.

Only service metadata is kept: radio header of the packet as heard by the MQTT
gateway (SNR, RSSI, hops, relay) plus decoded service payloads (node info,
position, telemetry, neighbour info, traceroute, map report). Message texts,
user names and keys are dropped.
"""
import base64

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from google.protobuf.json_format import MessageToDict
from google.protobuf.message import DecodeError
from meshtastic.protobuf import mesh_pb2, mqtt_pb2, portnums_pb2, telemetry_pb2

# Key of the default channel (PSK "AQ==", index 1), see Meshtastic channel docs.
DEFAULT_KEY = base64.b64decode("1PG7OiApB1nwvP+rz05pAQ==")

HEADER_FIELDS = ("hop_start", "hop_limit", "rx_snr", "rx_rssi", "rx_time", "relay_node",
                 "next_hop", "via_mqtt", "want_ack", "priority", "transport_mechanism")

# Payload fields that carry personal data rather than radio/network information.
DROP_FIELDS = {"longName", "shortName", "publicKey", "macaddr"}

PAYLOAD_TYPES = {
    portnums_pb2.NODEINFO_APP: mesh_pb2.User,
    portnums_pb2.POSITION_APP: mesh_pb2.Position,
    portnums_pb2.TELEMETRY_APP: telemetry_pb2.Telemetry,
    portnums_pb2.NEIGHBORINFO_APP: mesh_pb2.NeighborInfo,
    portnums_pb2.TRACEROUTE_APP: mesh_pb2.RouteDiscovery,
    portnums_pb2.ROUTING_APP: mesh_pb2.Routing,
}


def decrypt(packet, key=DEFAULT_KEY):
    """AES-CTR with nonce = packet id (u64 LE) | sender (u32 LE) | 0 (u32)."""
    nonce = packet.id.to_bytes(8, "little") + getattr(packet, "from").to_bytes(4, "little") + bytes(4)
    decryptor = Cipher(algorithms.AES(key), modes.CTR(nonce)).decryptor()
    return decryptor.update(packet.encrypted) + decryptor.finalize()


def _clean(msg):
    d = MessageToDict(msg, preserving_proto_field_name=False)
    return {k: v for k, v in d.items() if k not in DROP_FIELDS}


def decode_envelope(topic, payload):
    """Return one flat dict for an MQTT message, or None if it is not a ServiceEnvelope."""
    env = mqtt_pb2.ServiceEnvelope()
    try:
        env.ParseFromString(payload)
    except DecodeError:
        return None
    if not env.HasField("packet"):
        return None
    p = env.packet
    rec = {
        "topic": topic,
        "channel_id": env.channel_id,
        "gateway_id": env.gateway_id,
        "from": getattr(p, "from"),
        "to": p.to,
        "id": p.id,
        "channel": p.channel,
    }
    for f in HEADER_FIELDS:
        rec[f] = getattr(p, f)

    data = None
    if p.HasField("decoded"):
        data, rec["enc"] = p.decoded, "plain"
    elif p.encrypted:
        rec["enc"] = "pki" if p.pki_encrypted else "channel"
        if not p.pki_encrypted:
            candidate = mesh_pb2.Data()
            try:
                candidate.ParseFromString(decrypt(p))
                if candidate.portnum in portnums_pb2.PortNum.values():
                    data, rec["enc"] = candidate, "default_key"
            except DecodeError:
                pass
    if data is None:
        return rec

    rec["portnum"] = portnums_pb2.PortNum.Name(data.portnum)
    rec["request_id"] = data.request_id
    rec["want_response"] = data.want_response
    if data.portnum in (portnums_pb2.TEXT_MESSAGE_APP, portnums_pb2.TEXT_MESSAGE_COMPRESSED_APP):
        rec["text_len"] = len(data.payload)  # length only, never the content
    elif data.portnum == portnums_pb2.MAP_REPORT_APP:
        rec["payload"] = _parse(mqtt_pb2.MapReport, data.payload)
    elif data.portnum in PAYLOAD_TYPES:
        rec["payload"] = _parse(PAYLOAD_TYPES[data.portnum], data.payload)
    return rec


def _parse(cls, raw):
    msg = cls()
    try:
        msg.ParseFromString(raw)
    except DecodeError:
        return {"_decode_error": True}
    return _clean(msg)
