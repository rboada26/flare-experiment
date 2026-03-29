import os
import glob
import numpy as np
import pandas as pd
from scapy.all import rdpcap, IP
from scipy import stats

# ── Constants ────────────────────────────────────────────────────────────────
PCAP_DIR    = "./captures"
OUTPUT_CSV  = "./features.csv"
WINDOW_SIZE = 30       # seconds per window (paper uses 250-300s, we use 60s to match our shorter sessions)
MIN_PACKETS = 50        # discard windows with too few packets
SIZE_BINS   = 20        # number of bins for packet length histogram

# ── Helpers ──────────────────────────────────────────────────────────────────

def load_pcap(path):
    """Load pcap and return list of (timestamp, size, direction) tuples.
    Direction: 1 = uplink (client→server), -1 = downlink (server→client).
    Server IPs end in .10 in both networks."""
    packets = []
    try:
        raw = rdpcap(path)
    except Exception as e:
        print(f"  ERROR reading {path}: {e}")
        return packets

    for pkt in raw:
        if IP not in pkt:
            continue
        ts   = float(pkt.time)
        size = len(pkt)
        src  = pkt[IP].src
        # server always ends in .10
        direction = -1 if src.endswith(".10") else 1
        packets.append((ts, size, direction))

    packets.sort(key=lambda x: x[0])
    return packets


def segment_packets(packets, window_size):
    """Split packet list into fixed-duration windows."""
    if not packets:
        return []
    windows = []
    start   = packets[0][0]
    current = []
    for pkt in packets:
        if pkt[0] - start <= window_size:
            current.append(pkt)
        else:
            if len(current) >= MIN_PACKETS:
                windows.append(current)
            current = [pkt]
            start   = pkt[0]
    if len(current) >= MIN_PACKETS:
        windows.append(current)
    return windows


# ── Flow-level features ───────────────────────────────────────────────────────

def extract_flow_features(window):
    """Coarse-grained temporal and directional statistics."""
    timestamps  = np.array([p[0] for p in window])
    sizes       = np.array([p[1] for p in window])
    directions  = np.array([p[2] for p in window])

    duration    = timestamps[-1] - timestamps[0] + 1e-9
    iats        = np.diff(timestamps)

    uplink      = sizes[directions == 1]
    downlink    = sizes[directions == -1]

    def safe_stats(arr):
        if len(arr) == 0:
            return [0] * 9
        return [
            np.mean(arr), np.max(arr), np.min(arr),
            np.var(arr),  np.std(arr),
            np.median(arr),
            stats.median_abs_deviation(arr) if len(arr) > 1 else 0,
            stats.skew(arr)     if len(arr) > 2 else 0,
            stats.kurtosis(arr) if len(arr) > 3 else 0,
        ]

    # packet rate per second
    pkt_rate = len(window) / duration

    # up/down ratio
    ud_ratio = len(uplink) / (len(downlink) + 1e-9)

    feats = (
        [pkt_rate, ud_ratio] +
        safe_stats(uplink) +
        safe_stats(downlink) +
        safe_stats(iats) if len(iats) > 0 else safe_stats(uplink) + safe_stats(downlink) + [0]*9
    )

    # flatten properly
    flow_feats = [pkt_rate, ud_ratio]
    flow_feats += safe_stats(uplink)
    flow_feats += safe_stats(downlink)
    flow_feats += safe_stats(iats) if len(iats) > 0 else [0] * 9

    return np.array(flow_feats, dtype=np.float32)


# ── Packet-level features ─────────────────────────────────────────────────────

def extract_packet_features(window):
    """Fine-grained structural features: size histogram + edge packets."""
    sizes = np.array([p[1] for p in window])
    iats  = np.diff([p[0] for p in window])

    # Packet length histogram
    hist, _ = np.histogram(sizes, bins=SIZE_BINS, range=(0, 65536))
    hist     = hist.astype(np.float32) / (len(sizes) + 1e-9)  # normalize

    # Edge features: first 5 and last 5 packets
    edge_n     = 5
    first_pkts = sizes[:edge_n]  if len(sizes) >= edge_n else np.pad(sizes,  (0, edge_n - len(sizes)))
    last_pkts  = sizes[-edge_n:] if len(sizes) >= edge_n else np.pad(sizes,  (0, edge_n - len(sizes)))
    first_iats = iats[:edge_n]   if len(iats)  >= edge_n else np.pad(iats,   (0, edge_n - len(iats)))
    last_iats  = iats[-edge_n:]  if len(iats)  >= edge_n else np.pad(iats,   (0, edge_n - len(iats)))

    pkt_feats = np.concatenate([
        hist,
        first_pkts.astype(np.float32),
        last_pkts.astype(np.float32),
        first_iats.astype(np.float32),
        last_iats.astype(np.float32),
    ])

    return pkt_feats


# ── Main extraction loop ──────────────────────────────────────────────────────

def process_pcap(path, label):
    """Extract all windows from one pcap file, return list of feature dicts."""
    print(f"  Processing: {os.path.basename(path)}")
    packets = load_pcap(path)
    if not packets:
        return []

    windows = segment_packets(packets, WINDOW_SIZE)
    print(f"    {len(packets)} packets → {len(windows)} windows")

    rows = []
    for i, window in enumerate(windows):
        flow_feats = extract_flow_features(window)
        pkt_feats  = extract_packet_features(window)

        row = {
            "label":    label,
            "pcap":     os.path.basename(path),
            "window":   i,
            "n_packets": len(window),
        }

        # Add flow features with names
        flow_names = (
            ["pkt_rate", "up_down_ratio"] +
            [f"up_{s}"   for s in ["mean","max","min","var","std","median","mad","skew","kurt"]] +
            [f"down_{s}" for s in ["mean","max","min","var","std","median","mad","skew","kurt"]] +
            [f"iat_{s}"  for s in ["mean","max","min","var","std","median","mad","skew","kurt"]]
        )
        for name, val in zip(flow_names, flow_feats):
            row[name] = val

        # Add packet features with names
        pkt_names = (
            [f"hist_{i}" for i in range(SIZE_BINS)] +
            [f"first_size_{i}" for i in range(5)] +
            [f"last_size_{i}"  for i in range(5)] +
            [f"first_iat_{i}"  for i in range(5)] +
            [f"last_iat_{i}"   for i in range(5)]
        )
        for name, val in zip(pkt_names, pkt_feats):
            row[name] = val

        rows.append(row)

    return rows


def main():
    all_rows = []

    # CNN pcaps → label 0
    cnn_files = sorted(glob.glob(os.path.join(PCAP_DIR, "*_cnn_*.pcap")))
    print(f"\nFound {len(cnn_files)} CNN pcap files")
    for path in cnn_files:
        all_rows.extend(process_pcap(path, label=0))

    # RNN pcaps → label 1
    rnn_files = sorted(glob.glob(os.path.join(PCAP_DIR, "*_rnn_*.pcap")))
    print(f"\nFound {len(rnn_files)} RNN pcap files")
    for path in rnn_files:
        all_rows.extend(process_pcap(path, label=1))

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nDone. Feature matrix saved to {OUTPUT_CSV}")
    print(f"Shape: {df.shape}")
    print(f"\nLabel distribution:")
    print(df["label"].value_counts())
    print(f"\nFeature columns: {len(df.columns) - 4}")  # minus label/pcap/window/n_packets
    print(f"\nSample (first 3 rows):")
    print(df[["label", "n_packets", "pkt_rate", "up_down_ratio", "iat_mean"]].head(3))


if __name__ == "__main__":
    main()
