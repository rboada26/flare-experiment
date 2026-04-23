# FLARE Extension — Fine-Grained FL Architecture Fingerprinting

An extension of the FLARE side-channel attack demonstrating that passive
wireless traffic analysis can identify not just CNN vs. RNN model families,
but specific architectures within each family.

> Based on: *"FLARE: A Wireless Side-Channel Fingerprinting Attack on
> Federated Learning"* — Shuvo et al., arXiv:2512.10296 (2025)

---

## Protocol Defense Experiment — MIST

This branch extends the lab with a proposed network-level defense called
**MIST** (Masking via Interleaved Scheduled Transmissions). MIST belongs
to the Traffic Flow Confidentiality (TFC) family of defenses: it interposes
a bidirectional proxy between each FL client and server that reshapes all
traffic into a constant-rate, fixed-size stream, injecting cryptographically
random decoy packets whenever the application has nothing to send.

The goal is to defeat FLARE by eliminating the two signals it exploits:
packet size variation (which leaks model parameter count) and inter-arrival
timing (which leaks round boundaries and training duration).

The interactive dashboard includes an **Eavesdropper View** that lets you
run a live test session with MIST enabled and see in real time what the
attacker observes on the wire — packet size distributions, cumulative PCAP
sizes per architecture, coefficient of variation, and whether any
architecture remains distinguishable.

For a full technical description of the protocol, its wire format, and the
proxy architecture, see [MIST.md](MIST.md).

---

## Key Results

| Task | Variant | F1 Score |
|------|---------|----------|
| 6-class (all architectures) | Packet-only | **0.977 ± 0.016** |
| 6-class (all architectures) | Fusion MetaLR | 0.968 ± 0.035 |
| CNN family only (3-class) | Flow-only RF | **0.974 ± 0.013** |
| RNN family only (3-class) | Flow-only RF | **0.951 ± 0.028** |

**The original FLARE paper achieved ~98% F1 on binary CNN vs. RNN.
This extension matches that performance on a 6-class fine-grained problem.**

---

## Architectures Fingerprinted

| Label | Architecture | Family | Parameters | Avg. capture size |
|-------|-------------|--------|------------|-------------------|
| 0 | SimpleCNN (custom) | CNN | ~500K | ~73 MB |
| 1 | ResNet18 | CNN | ~11.2M | ~1.1 GB |
| 2 | MobileNetV2 | CNN | ~3.4M | ~240 MB |
| 3 | GRU | RNN | ~64K | ~3.6 MB |
| 4 | LSTM | RNN | ~66K | ~2.4 MB |
| 5 | BiLSTM | RNN | ~132K | ~4.8 MB |

---

## Requirements

- Linux host (tested on Ubuntu 24, ThinkPad, 31GB RAM)
- Docker + Docker Compose
- `tcpdump` — `sudo apt install tcpdump`
- `tshark` — `sudo apt install tshark` (verification only)
- Python 3.11+ with: `scapy`, `pandas`, `numpy`, `scipy`, `scikit-learn`

---

## Setup

### 1. Allow tcpdump without password prompt

```bash
sudo visudo
# Add this line — replace YOUR_USERNAME with your actual username (run `whoami` to check):
# YOUR_USERNAME ALL=(ALL) NOPASSWD: /usr/sbin/tcpdump
```

### 2. Build all Docker images

```bash
docker compose build --no-cache
```

This builds 6 client images and 1 shared server image. Total build time
is approximately 10–15 minutes depending on your connection.

### 3. Collect dataset

```bash
chmod +x collect_data.sh capture.sh
./collect_data.sh
```

Runs 8 sessions × 6 architectures = 48 labeled PCAP files in `captures/`.
Each session takes approximately 3–5 minutes (MobileNet is the bottleneck).
Total collection time: ~30–40 minutes.

### 4. Extract features

```bash
pip install scapy pandas numpy scipy scikit-learn --break-system-packages
python3 extract_features.py
```

Produces `features.csv` — 302 labeled windows, 69 features each.

### 5. Run classifier

```bash
python3 classify.py
```

Outputs precision/recall/F1 table, confusion matrix, per-class report,
and intra-family sub-analysis. Saves results to:
- `cv_results_multiclass.csv`
- `confusion_matrix.csv`
- `feature_importance_multiclass.csv`

---

## Project Structure

```
flare-experiment/
├── docker-compose.yml          # 6 isolated FL networks, one per architecture
├── capture.sh                  # Single-session traffic capture
├── collect_data.sh             # Automated multi-session data collection
├── extract_features.py         # PCAP → labeled feature matrix
├── classify.py                 # Random Forest + late fusion classifier
├── server/
│   ├── server.py               # Flower FL server (5 rounds, shared)
│   └── Dockerfile
├── clients/
│   ├── requirements.txt
│   ├── cnn_client/             # SimpleCNN on CIFAR-10
│   ├── resnet_client/          # ResNet18 on CIFAR-10
│   ├── mobilenet_client/       # MobileNetV2 on CIFAR-10
│   ├── rnn_client/             # GRU on synthetic sequence data
│   ├── lstm_client/            # LSTM on synthetic sequence data
│   └── bilstm_client/          # BiLSTM on synthetic sequence data
└── captures/                   # Generated PCAPs (gitignored)
```

---

## Network Architecture

```
CNN family                          RNN family
──────────────────────────────      ──────────────────────────────
simplecnn_network  172.20.0.0/24    gru_network     172.21.0.0/24
resnet_network     172.20.1.0/24    lstm_network    172.21.1.0/24
mobilenet_network  172.20.2.0/24    bilstm_network  172.21.2.0/24

Each network:  server (.10) ← client (.11) + client (.12)
               ↑
        tcpdump on host bridge interface
        (simulates passive AP eavesdropper)
```

---

## How It Works

Traffic is captured at the Docker bridge interface on the host — the
same vantage point as a compromised wireless AP in the FLARE threat model.
Each architecture produces a distinct traffic signature due to differences
in model parameter counts, which determine update payload sizes:

- ResNet18 sends ~44MB per parameter update (11M params × 4 bytes)
- SimpleCNN sends ~2MB per update
- BiLSTM sends ~0.5MB — but with a different timing profile than GRU

The classifier learns these signatures from two feature views: flow-level
statistics (packet rates, directional sizes, inter-arrival times) and
packet-level features (size histograms, edge packet characteristics).
A late fusion meta-classifier combines both views.

---

## Key Findings

**1. Fine-grained fingerprinting is feasible.**
97.7% macro F1 across 6 architectures using only encrypted traffic metadata —
matching the original paper's binary attack performance on a much harder task.

**2. Packet features dominate at fine-grained resolution.**
Unlike binary CNN vs. RNN classification where flow features suffice, the
6-class problem requires the packet size histogram — which captures
architecture-specific fragmentation patterns invisible in aggregate statistics.

**3. Confusion is architecturally structured.**
Errors occur only between similar models (MobileNet ↔ ResNet, GRU ↔ LSTM).
No CNN window is ever misclassified as an RNN, confirming that inter-family
separation remains perfect even in the fine-grained setting.

**4. Downlink variance is the dominant fine-grained signal.**
In binary classification, uplink standard deviation dominates (large vs. small
updates). In 6-class classification, downlink variance takes over — capturing
how the server returns aggregated updates differently depending on the model's
parameter tensor structure.

**5. Intra-family discrimination is viable.**
CNN family: 97.4% ± 1.3% F1 (SimpleCNN vs. ResNet18 vs. MobileNetV2)
RNN family: 95.1% ± 2.8% F1 (GRU vs. LSTM vs. BiLSTM)

---

## Confusion Matrix (packet-only, 302 windows)

```
              SimpleCNN  ResNet18  MobileNet  GRU  LSTM  BiLSTM
SimpleCNN            47         0          1    0     0       0
ResNet18              0        55          1    0     0       0
MobileNet             0         1         53    0     0       0
GRU                   0         0          0   46     1       1
LSTM                  0         0          0    1    47       0
BiLSTM                0         0          0    0     1      47
```

---

## Open Problems / Future Work

- [ ] Add Transformer architectures (ViT, small BERT) as a 7th class
- [ ] Evaluate packet-padding defenses against intra-family discrimination
- [ ] Test under hardware heterogeneity (multiple physical devices)
- [ ] Extend to open-world setting with unknown architecture classes
- [ ] Assess performance degradation with shorter observation windows

---

## Reference

Shuvo, M.N.H., Hossain, M., Mallik, A., Twigg, J., & Dagefu, F. (2025).
FLARE: A Wireless Side-Channel Fingerprinting Attack on Federated Learning.
*arXiv:2512.10296*. https://arxiv.org/abs/2512.10296
