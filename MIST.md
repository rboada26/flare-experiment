# MIST — Masking via Interleaved Scheduled Transmissions

A Traffic Flow Confidentiality (TFC) defense layer for Federated Learning,
designed to defeat passive side-channel fingerprinting attacks such as FLARE.

---

## 1. Motivation

The FLARE attack identifies which FL model architecture a client is training
by passively observing encrypted network traffic. Two signals drive it:

- **Packet size** — gradient update payloads are proportional to parameter
  count, making each architecture emit a characteristic size distribution.
- **Timing** — round boundaries and training durations are visible in
  inter-arrival patterns, further separating architectures.

MIST eliminates both signals at the network layer without modifying the FL
framework, client code, or model weights.

---

## 2. Threat Model

The attacker is a **passive eavesdropper** (e.g., a compromised wireless AP
or an on-path observer) who can see all packets flowing between FL clients
and the FL server. The attacker:

- Knows P_fixed and the nominal packet rate (these are public protocol
  parameters, like TLS cipher suites).
- Cannot decrypt packet payloads.
- Classifies architecture by traffic metadata alone (sizes, timing, volume).

MIST provides **traffic unobservability**: under MIST, all architectures
produce an indistinguishable wire trace for a fixed observation window.

---

## 3. Architecture

MIST is deployed as an **in-process proxy thread** inside each FL container.
No separate sidecar or external process is required.

```
┌─────────────────────────────┐        ┌─────────────────────────────┐
│     FL Client Container     │        │     FL Server Container     │
│                             │        │                             │
│  Flower client              │        │  Flower server              │
│    ↓ connects to            │        │    ↑ binds to               │
│  localhost:9091             │        │  localhost:8080             │
│    ↑                        │        │    ↑                        │
│  MIST client proxy          │        │  MIST server proxy          │
│  (daemon thread)            │        │  (daemon thread)            │
│    ↓ encodes → sends        │        │    ↓ receives → decodes     │
└──────────────┬──────────────┘        └──────────────┬──────────────┘
               │                                      │
               │   ◄─── eavesdropper sees this ───►   │
               └──────────── eth0  :9090 ─────────────┘
                    constant-rate, fixed-size packets
```

**Client proxy** (`run_client_proxy`): binds `localhost:9091`. The Flower
client connects here instead of the real server. The proxy encodes the raw
FL byte stream into MIST packets and forwards them to the server proxy at
`<server_host>:9090`.

**Server proxy** (`run_server_proxy`): binds `0.0.0.0:9090`. Accepts the
MIST-encoded connection, decodes it, and forwards the raw byte stream to
the Flower server at `localhost:8080`. Flower itself binds only to
localhost and is never directly reachable from the network.

---

## 4. Wire Format

Every packet on the wire is exactly **P_fixed bytes** — no exceptions.

```
 0       1       2       3       4       5       6       7       8
 ┌───────┬───────────────────────┬───────────────────────┐
 │ type  │         seq           │         plen          │  ← header (9 B)
 │  1 B  │         4 B           │         4 B           │
 └───────┴───────────────────────┴───────────────────────┘
 ┌──────────────────────────────────────────────────────────── ─ ─
 │  payload  (plen bytes of real application data)
 │                                                              ─ ─
 ├────────────────────────────────────────────────────────────────
 │  padding  (P_fixed − 9 − plen bytes of cryptographic noise)
 └────────────────────────────────────────────────────────────────
```

### 4.1 Header Fields

| Field  | Size   | Encoding         | Values |
|--------|--------|------------------|--------|
| `type` | 1 byte | unsigned int     | `0` = DECOY, `1` = REAL |
| `seq`  | 4 bytes | big-endian uint32 | monotonically increasing per direction, wraps at 2³²−1 |
| `plen` | 4 bytes | big-endian uint32 | number of real payload bytes in this packet; `0` for DECOY |

The header is packed with the format `>BII` (big-endian, no alignment
padding), for a fixed size of **9 bytes**.

### 4.2 Packet Types

**REAL** (`type=1`): carries application data. `plen` bytes immediately
following the header are forwarded to the application; the remaining
`P_fixed − 9 − plen` bytes are cryptographic padding and are discarded.

**DECOY** (`type=0`): carries no application data. `plen` is always `0`.
The entire body after the header is cryptographic padding. DECOYs are
sent whenever the application has nothing to transmit, preserving the
constant-rate schedule.

---

## 5. Encoding — The Sender Side

The sender runs two concurrent asyncio tasks: `_feeder` and `_pacer`.

### 5.1 Feeder

```
_feeder:
  loop:
    chunk ← app_stream.read(P_fixed − 9)   # up to one packet's worth
    frame ← make_real_packet(chunk)
    enqueue(frame)
  on EOF: signal done
```

`read(n)` returns **up to** `n` bytes. For large FL payloads (e.g., a
ResNet18 gradient of 44.6 MB), this loop runs hundreds of times — once per
P_fixed-sized chunk — automatically segmenting the stream without any
application-level awareness of packet boundaries. Each chunk, regardless of
size, becomes exactly one P_fixed-byte wire packet.

### 5.2 Pacer

```
_pacer:
  deadline ← now
  loop:
    sleep until deadline
    deadline += 1/R          # drift-free absolute scheduling
    if queue non-empty:
      send dequeue()         # REAL packet
    elif done and session_end reached:
      break
    else:
      send make_decoy()      # DECOY packet
```

The pacer advances its deadline by exactly `1/R` seconds each tick,
independent of how long the send took. This prevents clock drift from
accumulating and ensures the inter-packet interval is stable at `1/R`
seconds for the full session duration.

If the application produces data faster than the pacer can send it
(rate < throughput), frames queue up. The queue is unbounded — MIST
never drops real data, it just increases send-side latency.

### 5.3 Packet Construction

```python
hdr = struct.pack('>BII', type, seq, len(payload))
pad = os.urandom(P_fixed - 9 - len(payload))
wire_packet = hdr + payload + pad
```

**Padding is generated by `os.urandom`** — a cryptographically secure
random byte source (backed by `/dev/urandom` or `getrandom(2)` on Linux).
This means an eavesdropper cannot distinguish a REAL packet's padded tail
from a DECOY packet's body, and cannot use padding patterns to infer
payload length or content.

---

## 6. Decoding — The Receiver Side

```
_deobfuscate:
  loop:
    raw ← read_exact(wire_stream, P_fixed)   # always read full frame
    type, seq, plen ← unpack header (raw[0:9])
    if type == REAL and plen > 0:
      app_stream.write(raw[9 : 9 + plen])    # forward only real bytes
    # DECOY frames are silently dropped
```

`read_exact` loops on `asyncio.StreamReader.read()` until exactly P_fixed
bytes have been accumulated, reassembling any TCP segmentation transparently.
The application-layer byte stream is reconstructed perfectly regardless of
how the OS fragmented the 256 KB frames into TCP segments.

This is the key property: the eavesdropper sees TCP segments whose sizes
are determined by the OS network stack (typically 1–9 KB depending on MTU
and offloading), but the **application-layer framing is always P_fixed**.
Both sides agree on P_fixed and advance through the stream in lockstep,
one P_fixed-byte frame at a time.

---

## 7. Session Duration Padding

After the FL session ends (Flower client disconnects), the sender does not
immediately close the wire connection. Instead, it continues sending DECOY
packets at the same constant rate until `session_duration` seconds have
elapsed since the session start.

```
session_end = start + session_duration   # absolute time
while now < session_end:
    send DECOY
```

This hides session length — and therefore training duration and the number
of FL rounds — from the eavesdropper. Without this, an attacker could
distinguish architectures by the time window during which real traffic
flows, even if per-packet sizes are uniform.

In the lab, `session_duration` is set to `NUM_ROUNDS × 120` seconds so
that all architectures, regardless of training speed, pad out to the same
maximum possible session length.

---

## 8. Parameters

| Parameter         | Default    | Effect |
|-------------------|------------|--------|
| `P_fixed`         | 256 KB     | Wire packet size. Must exceed the largest expected single gRPC message chunk. Larger values reduce per-packet overhead but increase DECOY bandwidth cost. |
| `R` (rate)        | 10 pps     | Packets per second in each direction. Total wire bandwidth = `P_fixed × R × 2` bytes/sec. |
| `session_duration`| `rounds × 120 s` | How long to pad with DECOYs after FL ends. Set to the maximum expected session length across all architectures. |

---

## 9. What MIST Hides vs. What Remains Visible

| Observable           | Without MIST                        | With MIST |
|----------------------|-------------------------------------|-----------|
| Per-packet size      | Varies by model (leaked)            | Always P_fixed |
| Inter-packet interval| Varies by round/training speed      | Always 1/R seconds |
| Session duration     | Varies by architecture + rounds     | Fixed at `session_duration` |
| Total byte volume    | Proportional to model size (leaked) | Fixed: `P_fixed × R × session_duration` |
| Payload content      | Encrypted (not visible)             | Encrypted + indistinguishable padding |

Under correctly configured MIST, all six architectures produce an
**identical wire trace** for a fixed observation window. The classifier
cannot extract any architecture-discriminating features.
