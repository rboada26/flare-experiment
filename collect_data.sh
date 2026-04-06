#!/usr/bin/env bash
set -euo pipefail

NUM_SESSIONS=8
ARCHITECTURES=(simplecnn resnet mobilenet gru lstm bilstm)
SUBNETS=(172.20.0 172.20.1 172.20.2 172.21.0 172.21.1 172.21.2)

wait_for_all_clients() {
    echo "Polling for training completion..."
    while true; do
        all_done=true
        status_line=""
        for arch in "${ARCHITECTURES[@]}"; do
            s1=$(docker inspect -f '{{.State.Status}}' "fl_${arch}_client"  2>/dev/null || echo "gone")
            s2=$(docker inspect -f '{{.State.Status}}' "fl_${arch}_client2" 2>/dev/null || echo "gone")
            status_line+="${arch}:${s1}/${s2} "
            if [[ "$s1" != "exited" || "$s2" != "exited" ]]; then
                all_done=false
            fi
        done
        echo "  $status_line"
        $all_done && { echo "All clients finished."; break; }
        sleep 5
    done
}

for i in $(seq 1 $NUM_SESSIONS); do
    echo ""
    echo "============================================"
    echo " Starting session $i / $NUM_SESSIONS"
    echo "============================================"

    docker compose down --remove-orphans 2>/dev/null || true
    sleep 2

    docker compose up -d \
      simplecnn_server resnet_server mobilenet_server \
      gru_server lstm_server bilstm_server
    sleep 10

    mkdir -p captures
    PIDS=()
    for idx in "${!ARCHITECTURES[@]}"; do
        arch="${ARCHITECTURES[$idx]}"
        subnet="${SUBNETS[$idx]}"
        net_name="flare-experiment_${arch}_network"
        net_id=$(docker network inspect "$net_name" --format '{{.Id}}' 2>/dev/null | cut -c1-12)
        iface="br-${net_id}"
        pcap="captures/session${i}_${arch}_$(date +%Y%m%d_%H%M%S).pcap"
        filter="(src net ${subnet}.0/24 and dst net ${subnet}.0/24) and not multicast"
        sudo tcpdump -i "$iface" -w "$pcap" -s0 "$filter" &
        PIDS+=($!)
        sleep 0.5
    done

    sleep 2

    docker compose up -d \
      simplecnn_client simplecnn_client2 \
      resnet_client resnet_client2 \
      mobilenet_client mobilenet_client2 \
      gru_client gru_client2 \
      lstm_client lstm_client2 \
      bilstm_client bilstm_client2

    wait_for_all_clients

    sleep 3

    for pid in "${PIDS[@]}"; do
        sudo kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true

    docker compose down --remove-orphans
    sleep 3

    echo "Session $i done:"
    ls -lh captures/session${i}_*.pcap 2>/dev/null | awk '{print $5, $9}'
done

echo ""
echo "All sessions complete."
ls -lh captures/
