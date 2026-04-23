#!/usr/bin/env bash
set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
NUM_SESSIONS=8
NUM_ROUNDS=5
ARCH_FILTER=""

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --sessions) NUM_SESSIONS="$2"; shift 2 ;;
        --rounds)   NUM_ROUNDS="$2";   shift 2 ;;
        --archs)    ARCH_FILTER="$2";  shift 2 ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: $0 [--sessions N] [--rounds N] [--archs arch1,arch2,...]" >&2
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo " ext-FLARE-lab Data Collection"
echo " Sessions : $NUM_SESSIONS"
echo " FL Rounds: $NUM_ROUNDS"
echo " Mode     : all 6 architectures concurrent"
echo "============================================"
echo ""

printf "NUM_ROUNDS=%d\n" "$NUM_ROUNDS" > .env
echo "[init] Written .env with NUM_ROUNDS=$NUM_ROUNDS"

if docker image inspect partner-lab-tcpdump &>/dev/null; then
    echo "[init] tcpdump image already exists, skipping build."
else
    echo "[init] Building tcpdump capture image..."
    docker build -q -t partner-lab-tcpdump ./tcpdump
fi
echo "[init] tcpdump image ready."

echo "[init] Building FL server images..."
docker compose build simplecnn_server resnet_server mobilenet_server \
    gru_server lstm_server bilstm_server --quiet 2>&1 || true
echo "[init] FL server images ready."

echo "[init] Building FL client images..."
docker compose build --no-deps simplecnn_client simplecnn_client2 \
    resnet_client resnet_client2 \
    mobilenet_client mobilenet_client2 \
    gru_client gru_client2 \
    lstm_client lstm_client2 \
    bilstm_client bilstm_client2 --quiet 2>&1 || true
echo "[init] FL client images ready."

ALL_ARCHITECTURES=(simplecnn resnet mobilenet gru lstm bilstm)
ALL_SUBNETS=(172.20.0 172.20.1 172.20.2 172.21.0 172.21.1 172.21.2)

ARCHITECTURES=()
SUBNETS=()
if [[ -n "$ARCH_FILTER" ]]; then
    IFS=',' read -ra FILTER_LIST <<< "$ARCH_FILTER"
    for i in "${!ALL_ARCHITECTURES[@]}"; do
        arch="${ALL_ARCHITECTURES[$i]}"
        for f in "${FILTER_LIST[@]}"; do
            if [[ "$arch" == "$f" ]]; then
                ARCHITECTURES+=("$arch")
                SUBNETS+=("${ALL_SUBNETS[$i]}")
                break
            fi
        done
    done
else
    ARCHITECTURES=("${ALL_ARCHITECTURES[@]}")
    SUBNETS=("${ALL_SUBNETS[@]}")
fi

mkdir -p captures

for i in $(seq 1 "$NUM_SESSIONS"); do
    echo ""
    echo "============================================"
    echo " Starting session $i / $NUM_SESSIONS"
    echo "============================================"

    docker compose down --remove-orphans 2>/dev/null || true
    sleep 2

    # Force-remove any stale containers before starting
    for arch in "${ARCHITECTURES[@]}"; do
        docker rm -f "fl_${arch}_server" "fl_${arch}_client" "fl_${arch}_client2" \
            "cap_${arch}_${i}" &>/dev/null || true
    done

    # Start all servers
    for arch in "${ARCHITECTURES[@]}"; do
        docker compose up -d "${arch}_server" 2>&1 | grep -v "^$" || true
    done
    sleep 8

    # Start tcpdump for each arch
    for idx in "${!ARCHITECTURES[@]}"; do
        arch="${ARCHITECTURES[$idx]}"
        subnet="${SUBNETS[$idx]}"
        pcap_name="session${i}_${arch}_$(date +%Y%m%d_%H%M%S).pcap"
        if docker run --rm -d \
            --network "container:fl_${arch}_server" \
            --name "cap_${arch}_${i}" \
            -v "$(pwd)/captures:/out" \
            partner-lab-tcpdump \
            -i eth0 -w "/out/${pcap_name}" -s0 "net ${subnet}.0/24" \
            &>/dev/null; then
            echo "  [tcpdump] ${arch} → captures/${pcap_name}"
        fi
        sleep 0.3
    done
    sleep 2

    # Start all clients
    for arch in "${ARCHITECTURES[@]}"; do
        docker compose up -d "${arch}_client" "${arch}_client2" 2>&1 | grep -v "^$" || true
    done

    # Wait for all clients to finish
    echo "  Waiting for all clients to finish..."
    while true; do
        all_done=true
        for arch in "${ARCHITECTURES[@]}"; do
            s1=$(docker inspect -f '{{.State.Status}}' "fl_${arch}_client"  2>/dev/null || echo "gone")
            s2=$(docker inspect -f '{{.State.Status}}' "fl_${arch}_client2" 2>/dev/null || echo "gone")
            if [[ ( "$s1" != "exited" && "$s1" != "gone" ) || \
                  ( "$s2" != "exited" && "$s2" != "gone" ) ]]; then
                all_done=false; break
            fi
        done
        $all_done && break
        sleep 5
    done
    sleep 3

    # Stop tcpdumps
    for arch in "${ARCHITECTURES[@]}"; do
        docker stop "cap_${arch}_${i}" &>/dev/null || true
    done
    sleep 1

    # Save logs before teardown
    mkdir -p debug
    for arch in "${ARCHITECTURES[@]}"; do
        for role in server client client2; do
            docker logs "fl_${arch}_${role}" > "debug/session${i}_${arch}_${role}.log" 2>&1 || true
        done
    done

    # Tear down using direct docker commands to avoid compose state issues
    for arch in "${ARCHITECTURES[@]}"; do
        docker stop "fl_${arch}_server" "fl_${arch}_client" "fl_${arch}_client2" &>/dev/null || true
        docker rm   "fl_${arch}_server" "fl_${arch}_client" "fl_${arch}_client2" &>/dev/null || true
    done
    sleep 1

    echo ""
    echo "  [session $i] Done. Captures:"
    ls -lh "captures/session${i}_"*.pcap 2>/dev/null | awk '{print "    " $5, $9}' \
        || echo "    (no pcaps found)"
done

echo ""
echo "============================================"
echo " All $NUM_SESSIONS sessions complete."
echo "============================================"
ls -lh captures/
