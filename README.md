# FLARE Experiment — Wireless Side-Channel Fingerprinting of Federated Learning

A Docker-based replication and extension of the FLARE attack from:
> "FLARE: A Wireless Side-Channel Fingerprinting Attack on Federated Learning"
> Shuvo et al., arXiv:2512.10296 (2025)

## What this is

This repo sets up a controlled FL environment where a CNN client and an RNN client
train simultaneously on separate Docker networks. Traffic is captured at the virtual
bridge interface (simulating the AP vantage point from the paper) and saved as
labeled PCAPs for downstream fingerprinting analysis.

## Requirements

- Docker + Docker Compose
- tcpdump (`sudo apt install tcpdump`)
- tshark for verification (`sudo apt install tshark`)
- Linux host (tested on Ubuntu 24, ThinkPad)

## Setup

### 1. Allow tcpdump without password prompt
```bash
sudo visudo
# Add this line (replace YOUR_USERNAME):
# YOUR_USERNAME ALL=(ALL) NOPASSWD: /usr/sbin/tcpdump
```

### 2. Build images
```bash
docker compose build --no-cache
```

### 3. Collect dataset (8 sessions, ~15 minutes)
```bash
chmod +x collect_data.sh capture.sh
./collect_data.sh
```

This generates 16 labeled PCAPs in `captures/`:
- `session{1-8}_cnn_*.pcap` — CNN client traffic (~72MB each)
- `session{1-8}_rnn_*.pcap` — RNN client traffic (~3.6MB each)

## Project structure
```
flare-experiment/
├── docker-compose.yml        # Two isolated FL networks (CNN + RNN)
├── capture.sh                # Single-session traffic capture
├── collect_data.sh           # Automated multi-session data collection
├── server/
│   ├── server.py             # Flower FL server (shared by both networks)
│   └── Dockerfile
├── clients/
│   ├── requirements.txt
│   ├── cnn_client/
│   │   ├── client.py         # SimpleCNN trained on CIFAR-10 subset
│   │   └── Dockerfile
│   └── rnn_client/
│       ├── client.py         # GRU model trained on synthetic sequence data
│       └── Dockerfile
└── captures/                 # Generated PCAPs go here (gitignored)
```

## Architecture
```
CNN Network (172.20.0.0/24)        RNN Network (172.21.0.0/24)
┌─────────────────────┐            ┌─────────────────────┐
│  cnn_server :8080   │            │  rnn_server :8080   │
│  172.20.0.10        │            │  172.21.0.10        │
├─────────────────────┤            ├─────────────────────┤
│  cnn_client         │            │  rnn_client         │
│  172.20.0.11        │            │  172.21.0.11        │
│  cnn_client2        │            │  rnn_client2        │
│  172.20.0.12        │            │  172.21.0.12        │
└─────────────────────┘            └─────────────────────┘
         ▲                                   ▲
    tcpdump on                          tcpdump on
    br-xxxxxxxx                         br-xxxxxxxx
    (host bridge)                       (host bridge)
```

Traffic is captured from the host on Docker bridge interfaces, 
replicating the passive AP eavesdropper in the FLARE threat model.

## Expected output

Each session produces clean FL traffic with no external IPs:
- CNN: ~1300 packets, ~37MB per client, ~60s duration
- RNN: smaller updates, faster convergence, ~3.6MB total

## Next steps

- [ ] Feature extraction (flow-level + packet-level)
- [ ] Random Forest base classifiers
- [ ] Late fusion meta-classifier (MetaLR + MetaXGB)
- [ ] Open-world evaluation
- [ ] Extension: fine-grained architecture discrimination (ResNet vs MobileNet)

## Reference

Shuvo, M.N.H., Hossain, M., Mallik, A., Twigg, J., & Dagefu, F. (2025).
FLARE: A Wireless Side-Channel Fingerprinting Attack on Federated Learning.
https://arxiv.org/abs/2512.10296
