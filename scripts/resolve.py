#!/usr/bin/env python3
"""Minimal stub DNS resolver (no external deps).

This machine has an empty /etc/resolv.conf (no nameserver), so glibc name
resolution fails. Routing works, however, so we query a reachable public DNS
server directly over UDP/53 and print A records. Used to feed `curl --resolve`
for hosts that are not on the Cursor sandbox allowlist.
"""
import socket
import struct
import sys


def query(server: str, name: str, timeout: float = 5.0):
    tid = 0x1234
    header = struct.pack(">HHHHHH", tid, 0x0100, 1, 0, 0, 0)
    q = b"".join(struct.pack("B", len(l)) + l.encode() for l in name.split(".")) + b"\x00"
    q += struct.pack(">HH", 1, 1)  # QTYPE=A, QCLASS=IN
    pkt = header + q
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(timeout)
    try:
        s.sendto(pkt, (server, 53))
        data, _ = s.recvfrom(2048)
    except Exception as e:  # noqa: BLE001
        return f"ERR {e}"
    finally:
        s.close()
    ancount = struct.unpack(">H", data[6:8])[0]
    idx = 12
    while data[idx] != 0:
        idx += 1 + data[idx]
    idx += 5  # null + QTYPE + QCLASS
    ips = []
    for _ in range(ancount):
        if data[idx] & 0xC0 == 0xC0:
            idx += 2
        else:
            while data[idx] != 0:
                idx += 1 + data[idx]
            idx += 1
        rtype, _rclass, _ttl, rdlen = struct.unpack(">HHIH", data[idx:idx + 10])
        idx += 10
        rdata = data[idx:idx + rdlen]
        idx += rdlen
        if rtype == 1 and rdlen == 4:
            ips.append(".".join(map(str, rdata)))
    return ips or "no A record"


DEFAULT_SERVER = "223.5.5.5"  # AliDNS: the only reachable resolver on this box


def first_ip(name: str, server: str = DEFAULT_SERVER):
    res = query(server, name)
    if isinstance(res, list) and res:
        return res[0]
    return None


if __name__ == "__main__":
    # Usage:
    #   resolve.py <host> [host2 ...]            -> print first IP per host (for scripts)
    #   resolve.py --all <host> [host2 ...]      -> probe all servers (diagnostics)
    args = sys.argv[1:]
    if args and args[0] == "--all":
        servers = ["8.8.8.8", "1.1.1.1", "10.7.0.1", "223.5.5.5"]
        for name in args[1:]:
            for srv in servers:
                print(f"{srv:15s} {name:35s} -> {query(srv, name)}")
    else:
        for name in (args or ["download.pytorch.org"]):
            ip = first_ip(name)
            print(ip if ip else f"# FAILED to resolve {name}")

