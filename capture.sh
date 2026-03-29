#!/usr/bin/env bash
set -euo pipefail

CAPTURE_DIR="./captures"
LABEL="${1:-session}"

get_bridge() {
  local net=$1
  local id
  id=$(docker network inspect "$net" --format '{{.Id}}' 2>/dev/null | cut -c1-12)
  echo "br-${id}"
}

CNN_IFACE=$(get_bridge "flare-experiment_cnn_network")
RNN_IFACE=$(get_bridge "flare-experiment_rnn_network")
mkdir -p "$CAPTURE_DIR"

echo "Capturing CNN traffic on $CNN_IFACE"
echo "Capturing RNN traffic on $RNN_IFACE"
echo "Press Ctrl+C to stop both."

# Filter: only traffic between our server and clients, no multicast, no external
CNN_FILTER="(src net 192.168.100.0/24 and dst net 192.168.100.0/24) and not multicast"
RNN_FILTER="(src net 192.168.101.0/24 and dst net 192.168.101.0/24) and not multicast"

sudo tcpdump -i "$CNN_IFACE" -w "${CAPTURE_DIR}/${LABEL}_cnn_$(date +%Y%m%d_%H%M%S).pcap" \
  -s0 "$CNN_FILTER" &
PID_CNN=$!

sudo tcpdump -i "$RNN_IFACE" -w "${CAPTURE_DIR}/${LABEL}_rnn_$(date +%Y%m%d_%H%M%S).pcap" \
  -s0 "$RNN_FILTER" &
PID_RNN=$!

trap "kill $PID_CNN $PID_RNN 2>/dev/null; echo 'Capture stopped.'" INT
wait
