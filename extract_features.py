import os
import glob
import argparse
import numpy as np
import pandas as pd
from scapy.all import rdpcap, IP
from scipy import stats

PCAP_DIR    = "./captures"
OUTPUT_CSV  = "./features.csv"
WINDOW_SIZE = 30
MIN_PACKETS = 50
MIN_SIZE    = 0       # drop packets smaller than this (bytes); 66 matches paper
SIZE_BINS   = 20

# Label map — order matters for classifier interpretation
ARCH_LABELS = {
    "simplecnn":  0,
    "resnet":     1,
    "mobilenet":  2,
    "gru":        3,
    "lstm":       4,
    "bilstm":     5,
}

def get_label_from_filename(filename):
    base = os.path.basename(filename)
    for arch, label in ARCH_LABELS.items():
        if f"_{arch}_" in base:
            return label, arch
    return None, None

def load_pcap(path):
    packets = []
    try:
        raw = rdpcap(path)
    except Exception as e:
        print(f"  ERROR reading {path}: {e}")
        return packets
    for pkt in raw:
        if IP not in pkt:
            continue
        size = len(pkt)
        if size < MIN_SIZE:
            continue
        ts        = float(pkt.time)
        direction = -1 if pkt[IP].src.endswith(".10") else 1
        packets.append((ts, size, direction))
    packets.sort(key=lambda x: x[0])
    return packets

def segment_packets(packets, window_size):
    if not packets:
        return []
    windows, current = [], []
    start = packets[0][0]
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

def extract_flow_features(window):
    timestamps = np.array([p[0] for p in window])
    sizes      = np.array([p[1] for p in window])
    directions = np.array([p[2] for p in window])
    duration   = timestamps[-1] - timestamps[0] + 1e-9
    iats       = np.diff(timestamps)
    uplink     = sizes[directions ==  1]
    downlink   = sizes[directions == -1]

    def safe_stats(arr):
        if len(arr) == 0:
            return [0] * 9
        return [
            np.mean(arr), np.max(arr), np.min(arr),
            np.var(arr),  np.std(arr), np.median(arr),
            stats.median_abs_deviation(arr) if len(arr) > 1 else 0,
            stats.skew(arr)     if len(arr) > 2 else 0,
            stats.kurtosis(arr) if len(arr) > 3 else 0,
        ]

    feats  = [len(window) / duration, len(uplink) / (len(downlink) + 1e-9)]
    feats += safe_stats(uplink)
    feats += safe_stats(downlink)
    feats += safe_stats(iats) if len(iats) > 0 else [0] * 9
    return np.array(feats, dtype=np.float32)

def extract_packet_features(window):
    sizes = np.array([p[1] for p in window])
    iats  = np.diff([p[0] for p in window])
    hist, _ = np.histogram(sizes, bins=SIZE_BINS, range=(0, 65536))
    hist    = hist.astype(np.float32) / (len(sizes) + 1e-9)
    edge_n  = 5
    pad = lambda a, n: a[:n] if len(a) >= n else np.pad(a, (0, n - len(a)))
    return np.concatenate([
        hist,
        pad(sizes, edge_n).astype(np.float32),
        pad(sizes[::-1], edge_n).astype(np.float32),
        pad(iats,  edge_n).astype(np.float32),
        pad(iats[::-1],  edge_n).astype(np.float32),
    ])

def process_pcap(path, label, arch):
    print(f"  [{arch:>10}] {os.path.basename(path)}")
    packets = load_pcap(path)
    if not packets:
        return []
    windows = segment_packets(packets, WINDOW_SIZE)
    print(f"              {len(packets)} packets → {len(windows)} windows")

    flow_names = (
        ["pkt_rate", "up_down_ratio"] +
        [f"up_{s}"   for s in ["mean","max","min","var","std","median","mad","skew","kurt"]] +
        [f"down_{s}" for s in ["mean","max","min","var","std","median","mad","skew","kurt"]] +
        [f"iat_{s}"  for s in ["mean","max","min","var","std","median","mad","skew","kurt"]]
    )
    pkt_names = (
        [f"hist_{i}"       for i in range(SIZE_BINS)] +
        [f"first_size_{i}" for i in range(5)] +
        [f"last_size_{i}"  for i in range(5)] +
        [f"first_iat_{i}"  for i in range(5)] +
        [f"last_iat_{i}"   for i in range(5)]
    )

    rows = []
    for i, window in enumerate(windows):
        row = {
            "label":     label,
            "arch":      arch,
            "pcap":      os.path.basename(path),
            "window":    i,
            "n_packets": len(window),
        }
        for name, val in zip(flow_names, extract_flow_features(window)):
            row[name] = val
        for name, val in zip(pkt_names, extract_packet_features(window)):
            row[name] = val
        rows.append(row)
    return rows

def main():
    global WINDOW_SIZE, MIN_PACKETS, MIN_SIZE

    parser = argparse.ArgumentParser(description="Extract features from PCAP captures")
    parser.add_argument("--window",      type=int, default=WINDOW_SIZE,
                        help="Time window size in seconds (default: 30)")
    parser.add_argument("--min-packets", type=int, default=MIN_PACKETS,
                        help="Minimum packets per window (default: 50)")
    parser.add_argument("--min-size",    type=int, default=MIN_SIZE,
                        help="Drop packets smaller than this many bytes (default: 0 = keep all; paper uses 66)")
    args = parser.parse_args()

    WINDOW_SIZE = args.window
    MIN_PACKETS = args.min_packets
    MIN_SIZE    = args.min_size

    print(f"[extract_features] Config: window={WINDOW_SIZE}s, min_packets={MIN_PACKETS}, min_size={MIN_SIZE}b")
    print(f"[extract_features] Input: {PCAP_DIR}")
    print(f"[extract_features] Output: {OUTPUT_CSV}")
    print()

    all_rows = []
    all_files = sorted(glob.glob(os.path.join(PCAP_DIR, "*.pcap")))
    print(f"Found {len(all_files)} pcap files\n")

    for path in all_files:
        label, arch = get_label_from_filename(path)
        if label is None:
            print(f"  SKIPPING (unknown arch): {path}")
            continue
        all_rows.extend(process_pcap(path, label, arch))

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nDone. Saved to {OUTPUT_CSV}")
    print(f"Shape: {df.shape}")
    print(f"\nWindows per architecture:")
    print(df.groupby("arch")["window"].count().sort_values(ascending=False))
    print(f"\nLabel distribution:")
    print(df.groupby(["label","arch"]).size())

if __name__ == "__main__":
    main()
