# pd/ipc_protocol.py
import json
import socket
import struct
from dataclasses import is_dataclass, asdict


def _to_jsonable(obj):
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(x) for x in obj]
    return obj


def send_obj(sock: socket.socket, obj: dict):
    payload = json.dumps(_to_jsonable(obj), ensure_ascii=False).encode("utf-8")
    header = struct.pack("!I", len(payload))
    sock.sendall(header + payload)


def recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("socket closed while receiving")
        buf += chunk
    return buf


def recv_obj(sock: socket.socket) -> dict:
    header = recv_exact(sock, 4)
    (size,) = struct.unpack("!I", header)
    payload = recv_exact(sock, size)
    return json.loads(payload.decode("utf-8"))


def request_reply(host: str, port: int, obj: dict, timeout: float = 30.0) -> dict:
    with socket.create_connection((host, port), timeout=timeout) as sock:
        send_obj(sock, obj)
        return recv_obj(sock)