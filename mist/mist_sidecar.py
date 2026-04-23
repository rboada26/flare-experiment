#!/usr/bin/env python3
"""
MIST sidecar — constant-rate cover-traffic generator.

Runs as 'server' (inside the FL server's network namespace) or 'client'
(on the arch's Docker bridge network).  Both sides independently send
P_fixed-byte cryptographically-random packets at rate R pps, creating
a continuous bidirectional stream that hides FL round boundaries from
a passive network observer.

The sidecar never inspects, modifies, or proxies real FL traffic —
it only adds cover traffic alongside the existing Flower gRPC stream.
"""
import argparse, os, socket, sys, threading, time

MIST_PORT = 9090


def _sender(sock: socket.socket, p_fixed: int, rate_pps: int) -> None:
    """Send P_fixed random bytes at constant rate R until the socket closes."""
    interval = 1.0 / rate_pps
    while True:
        t0 = time.perf_counter()
        try:
            sock.sendall(os.urandom(p_fixed))
        except OSError:
            return
        wait = interval - (time.perf_counter() - t0)
        if wait > 0.0:
            time.sleep(wait)


def _drain(sock: socket.socket) -> None:
    """Discard all incoming bytes — we only care about generating traffic."""
    try:
        while sock.recv(65536):
            pass
    except OSError:
        pass


def run_server(p_fixed: int, rate_pps: int) -> None:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(('0.0.0.0', MIST_PORT))
    srv.listen(8)
    kb = p_fixed // 1024
    print(f'[MIST-server] :9090  P_fixed={kb}KB  R={rate_pps}pps', flush=True)
    while True:
        try:
            conn, addr = srv.accept()
        except OSError:
            return
        print(f'[MIST-server] client connected from {addr}', flush=True)
        threading.Thread(target=_sender, args=(conn, p_fixed, rate_pps), daemon=True).start()
        threading.Thread(target=_drain,  args=(conn,), daemon=True).start()


def run_client(host: str, p_fixed: int, rate_pps: int) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    kb = p_fixed // 1024
    for _ in range(60):
        try:
            sock.connect((host, MIST_PORT))
            break
        except OSError:
            time.sleep(1)
    else:
        print('[MIST-client] failed to connect after 60s', file=sys.stderr, flush=True)
        return
    print(f'[MIST-client] connected → {host}:{MIST_PORT}  P_fixed={kb}KB  R={rate_pps}pps', flush=True)
    t_send = threading.Thread(target=_sender, args=(sock, p_fixed, rate_pps), daemon=True)
    t_recv = threading.Thread(target=_drain,  args=(sock,), daemon=True)
    t_send.start()
    t_recv.start()
    t_send.join()  # block until sender exits (socket closed)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='MIST cover-traffic sidecar')
    ap.add_argument('role',      choices=['server', 'client'])
    ap.add_argument('--host',    default='127.0.0.1', help='server IP (client mode only)')
    ap.add_argument('--p-fixed', type=int, default=262144, help='packet size in bytes')
    ap.add_argument('--rate',    type=int, default=10,     help='send rate in packets/sec')
    args = ap.parse_args()
    if args.role == 'server':
        run_server(args.p_fixed, args.rate)
    else:
        run_client(args.host, args.p_fixed, args.rate)
