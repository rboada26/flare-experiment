#!/usr/bin/env python3
"""
MIST proxy — bidirectional TCP proxy implementing constant-rate packet obfuscation.

Wire format: every packet is exactly P_fixed bytes.
  Header (9 bytes): [type:1B][seq:4B][plen:4B]
  type=0 → DECOY  (plen=0, discarded by receiver)
  type=1 → REAL   (plen>0, forwarded to application)
  Padding: random bytes to reach P_fixed

Deployment:
  Server proxy — shares FL server's network namespace (--network container:fl_<arch>_server)
                 listens :9090, forwards decoded stream to localhost:8080 (Flower server)
  Client proxy — shares FL client's network namespace (--network container:fl_<arch>_client)
                 listens :9091, FL client connects here instead of the real server

Traffic seen by eavesdropper: constant-rate, fixed-size packets in both directions.
Round boundaries, gradient sizes, and model identity are hidden.
"""
import argparse
import asyncio
import os
import struct
import sys

HEADER_FMT  = '>BII'                        # big-endian: uint8 type + uint32 seq + uint32 plen
HEADER_SIZE = struct.calcsize(HEADER_FMT)   # 9 bytes
TYPE_DECOY  = 0
TYPE_REAL   = 1

CONNECT_RETRIES = 30
CONNECT_DELAY   = 1.0    # seconds between retries


class MISTConfig:
    __slots__ = ('p_fixed', 'payload_size', 'interval', 'session_duration')

    def __init__(self, p_fixed: int, rate_pps: int, session_duration: float = 0.0):
        if p_fixed <= HEADER_SIZE:
            raise ValueError(f'p_fixed must be > {HEADER_SIZE} bytes')
        self.p_fixed          = p_fixed
        self.payload_size     = p_fixed - HEADER_SIZE
        self.interval         = 1.0 / rate_pps
        self.session_duration = session_duration   # seconds; 0 = stop when FL done


async def _read_exact(reader: asyncio.StreamReader, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = await reader.read(n - len(buf))
        if not chunk:
            raise EOFError('connection closed')
        buf.extend(chunk)
    return bytes(buf)


async def _obfuscate(app_r: asyncio.StreamReader,
                     wire_w: asyncio.StreamWriter,
                     cfg: MISTConfig) -> None:
    """Read raw bytes from app, send constant-rate MIST-framed packets on wire.

    Two concurrent tasks:
      _feeder  — reads app bytes, enqueues REAL packets (never blocks on full queue)
      _pacer   — every 1/R seconds, sends next REAL packet or a DECOY if queue is empty
    """
    queue: asyncio.Queue = asyncio.Queue()   # unbounded; backlog if rate < app throughput
    seq   = [0]
    done  = asyncio.Event()

    def _make(ptype: int, payload: bytes) -> bytes:
        s = seq[0]
        seq[0] = (s + 1) & 0xFFFFFFFF
        hdr = struct.pack(HEADER_FMT, ptype, s, len(payload))
        pad = os.urandom(cfg.p_fixed - HEADER_SIZE - len(payload))
        return hdr + payload + pad

    async def _feeder() -> None:
        try:
            while True:
                chunk = await app_r.read(cfg.payload_size)
                if not chunk:
                    break
                queue.put_nowait(_make(TYPE_REAL, chunk))
        except (ConnectionResetError, BrokenPipeError, OSError):
            pass
        finally:
            done.set()

    async def _pacer() -> None:
        loop = asyncio.get_running_loop()
        start    = loop.time()
        deadline = start
        session_end = (start + cfg.session_duration) if cfg.session_duration > 0 else None
        try:
            while True:
                now = loop.time()
                sleep_for = deadline - now
                if sleep_for > 0:
                    await asyncio.sleep(sleep_for)
                deadline += cfg.interval   # drift-free: advance absolute deadline

                try:
                    pkt = queue.get_nowait()
                except asyncio.QueueEmpty:
                    if done.is_set():
                        # FL session over — pad with DECOYs until session_end
                        if session_end is None or loop.time() >= session_end:
                            break
                    pkt = _make(TYPE_DECOY, b'')

                wire_w.write(pkt)
                # drain outside the deadline arithmetic to avoid eating into sleep window
                try:
                    await wire_w.drain()
                except (ConnectionResetError, BrokenPipeError, OSError):
                    break
        except (ConnectionResetError, BrokenPipeError, OSError):
            pass
        finally:
            done.set()

    await asyncio.gather(_feeder(), _pacer(), return_exceptions=True)


async def _deobfuscate(wire_r: asyncio.StreamReader,
                       app_w: asyncio.StreamWriter,
                       cfg: MISTConfig) -> None:
    """Read MIST packets from wire; forward only REAL payloads to app."""
    try:
        while True:
            raw = await _read_exact(wire_r, cfg.p_fixed)
            ptype, _seq, plen = struct.unpack(HEADER_FMT, raw[:HEADER_SIZE])
            if ptype == TYPE_REAL and plen > 0:
                app_w.write(raw[HEADER_SIZE: HEADER_SIZE + plen])
                await app_w.drain()
    except (EOFError, ConnectionResetError, BrokenPipeError, OSError):
        pass


async def _connect_with_retry(host: str, port: int) -> tuple:
    for attempt in range(CONNECT_RETRIES):
        try:
            return await asyncio.open_connection(host, port)
        except OSError:
            if attempt < CONNECT_RETRIES - 1:
                await asyncio.sleep(CONNECT_DELAY)
    raise OSError(f'could not connect to {host}:{port} after {CONNECT_RETRIES} attempts')


async def _handle_server_proxy(wire_r: asyncio.StreamReader,
                                wire_w: asyncio.StreamWriter,
                                fl_host: str, fl_port: int,
                                cfg: MISTConfig) -> None:
    """Server proxy: accept MIST-encoded stream from client proxy, forward raw to FL server."""
    try:
        fl_r, fl_w = await _connect_with_retry(fl_host, fl_port)
    except OSError as e:
        print(f'[MIST-server] {e}', file=sys.stderr, flush=True)
        wire_w.close()
        return
    print(f'[MIST-server] tunnel up → {fl_host}:{fl_port}', flush=True)
    try:
        await asyncio.gather(
            _deobfuscate(wire_r, fl_w, cfg),    # client proxy → FL server (decode)
            _obfuscate(fl_r, wire_w, cfg),       # FL server → client proxy (encode)
            return_exceptions=True,
        )
    finally:
        for w in (fl_w, wire_w):
            try: w.close()
            except: pass


async def _handle_client_proxy(fl_r: asyncio.StreamReader,
                                fl_w: asyncio.StreamWriter,
                                server_host: str, server_port: int,
                                cfg: MISTConfig) -> None:
    """Client proxy: accept raw FL client connection, forward MIST-encoded to server proxy."""
    try:
        wire_r, wire_w = await _connect_with_retry(server_host, server_port)
    except OSError as e:
        print(f'[MIST-client] {e}', file=sys.stderr, flush=True)
        fl_w.close()
        return
    print(f'[MIST-client] tunnel up → {server_host}:{server_port}', flush=True)
    try:
        await asyncio.gather(
            _obfuscate(fl_r, wire_w, cfg),       # FL client → server proxy (encode)
            _deobfuscate(wire_r, fl_w, cfg),     # server proxy → FL client (decode)
            return_exceptions=True,
        )
    finally:
        for w in (wire_w, fl_w):
            try: w.close()
            except: pass


async def run_server_proxy(bind_port: int, fl_host: str, fl_port: int,
                            cfg: MISTConfig) -> None:
    server = await asyncio.start_server(
        lambda r, w: _handle_server_proxy(r, w, fl_host, fl_port, cfg),
        '0.0.0.0', bind_port,
    )
    kb = cfg.p_fixed // 1024
    print(
        f'[MIST-server] listening :9090 → {fl_host}:{fl_port}'
        f'  P_fixed={kb}KB  R={round(1 / cfg.interval)}pps',
        flush=True,
    )
    async with server:
        await server.serve_forever()


async def run_client_proxy(bind_port: int, server_host: str, server_port: int,
                            cfg: MISTConfig) -> None:
    server = await asyncio.start_server(
        lambda r, w: _handle_client_proxy(r, w, server_host, server_port, cfg),
        '0.0.0.0', bind_port,
    )
    kb = cfg.p_fixed // 1024
    print(
        f'[MIST-client] listening :9091 → {server_host}:{server_port}'
        f'  P_fixed={kb}KB  R={round(1 / cfg.interval)}pps',
        flush=True,
    )
    async with server:
        await server.serve_forever()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='MIST TCP proxy')
    ap.add_argument('role',      choices=['server', 'client'],
                    help='server = in FL server netns; client = in FL client netns')
    ap.add_argument('--host',    default='127.0.0.1',
                    help='server proxy IP (client mode: IP of FL server container)')
    ap.add_argument('--p-fixed', type=int, default=262144,
                    help='fixed packet size in bytes (default: 256 KB)')
    ap.add_argument('--rate',    type=int, default=10,
                    help='send rate in packets/sec (default: 10)')
    ap.add_argument('--session-duration', type=float, default=0.0,
                    help='pad with DECOYs for this many seconds after FL ends (0 = no padding)')
    args = ap.parse_args()

    cfg = MISTConfig(args.p_fixed, args.rate, args.session_duration)
    if args.role == 'server':
        # Shares FL server netns; forwards decoded traffic to Flower at localhost:8080
        asyncio.run(run_server_proxy(9090, 'localhost', 8080, cfg))
    else:
        # Shares FL client netns; FL client connects here (localhost:9091) instead of real server
        asyncio.run(run_client_proxy(9091, args.host, 9090, cfg))
