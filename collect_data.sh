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
echo " PARTNER-LAB Data Collection"
echo " Sessions : $NUM_SESSIONS"
echo " FL Rounds: $NUM_ROUNDS"
echo "============================================"
echo ""

# Write .env so docker compose picks up NUM_ROUNDS
printf "NUM_ROUNDS=%d\n" "$NUM_ROUNDS" > .env
echo "[init] Written .env with NUM_ROUNDS=$NUM_ROUNDS"

# Build the tcpdump capture image (fast — alpine base, cached after first build)
echo "[init] Building tcpdump capture image..."
docker build -q -t partner-lab-tcpdump ./tcpdump
echo "[init] tcpdump image ready."

ALL_ARCHITECTURES=(simplecnn resnet mobilenet gru lstm bilstm)
ALL_SUBNETS=(172.20.0 172.20.1 172.20.2 172.21.0 172.21.1 172.21.2)

# Filter to selected architectures (or use all if none specified)
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

echo " Architectures: ${ARCHITECTURES[*]}"

wait_for_all_clients() {
    echo "  [wait] Polling for training completion..."
    while true; do
        all_done=true
        status_line=""
        for arch in "${ARCHITECTURES[@]}"; do
            s1=$(docker inspect -f '{{.State.Status}}' "fl_${arch}_client"  2>/dev/null || echo "gone")
            s2=$(docker inspect -f '{{.State.Status}}' "fl_${arch}_client2" 2>/dev/null || echo "gone")
            status_line+="${arch}:${s1}/${s2} "
            if [[ "$s1" != "exited" && "$s1" != "gone" ]] || \
               [[ "$s2" != "exited" && "$s2" != "gone" ]]; then
                all_done=false
            fi
        done
        echo "  [status] $status_line"
        $all_done && { echo "  [wait] All clients finished."; break; }
        sleep 5
    done
}

mkdir -p captures

for i in $(seq 1 "$NUM_SESSIONS"); do
    echo ""
    echo "============================================"
    echo " Starting session $i / $NUM_SESSIONS"
    echo "============================================"

    # Tear down any leftovers
    docker compose down --remove-orphans 2>/dev/null || true
    # Remove any stale capture containers from a previous interrupted run
    for arch in "${ARCHITECTURES[@]}"; do
        docker rm -f "cap_${arch}_${i}" &>/dev/null || true
    done
    sleep 2

    # Start FL servers (only selected architectures)
    echo "  [session $i] Starting FL servers..."
    SERVERS=()
    for arch in "${ARCHITECTURES[@]}"; do SERVERS+=("${arch}_server"); done
    docker compose up -d "${SERVERS[@]}"

    echo "  [session $i] Waiting 10s for servers to be ready..."
    sleep 10

    # Start a tcpdump container sharing each server's network namespace.
    # --network container:<server> means we share the server's eth0 —
    # the server is party to every FL exchange, so we capture all gradient traffic.
    # BPF subnet filter excludes internet noise (dataset downloads, etc.).
    CAP_NAMES=()
    for idx in "${!ARCHITECTURES[@]}"; do
        arch="${ARCHITECTURES[$idx]}"
        subnet="${SUBNETS[$idx]}"
        pcap_name="session${i}_${arch}_$(date +%Y%m%d_%H%M%S).pcap"
        cap_name="cap_${arch}_${i}"

        if docker run --rm -d \
            --network "container:fl_${arch}_server" \
            --name "$cap_name" \
            -v "$(pwd)/captures:/out" \
            partner-lab-tcpdump \
            -i eth0 -w "/out/${pcap_name}" -s0 "net ${subnet}.0/24" \
            &>/dev/null; then
            echo "  [tcpdump] ${arch} → container:fl_${arch}_server → captures/${pcap_name}"
        else
            echo "  [warn] Could not start tcpdump container for ${arch}"
        fi
        CAP_NAMES+=("$cap_name")
        sleep 0.3
    done

    sleep 2

    # Start all clients (only selected architectures)
    echo "  [session $i] Starting FL clients..."
    CLIENTS=()
    for arch in "${ARCHITECTURES[@]}"; do CLIENTS+=("${arch}_client" "${arch}_client2"); done
    docker compose up -d "${CLIENTS[@]}"

    # Wait for all 12 client containers to exit
    wait_for_all_clients

    sleep 3

    # Stop tcpdump containers (docker stop flushes the pcap buffer cleanly)
    echo "  [session $i] Stopping capture containers..."
    for cap_name in "${CAP_NAMES[@]}"; do
        docker stop "$cap_name" &>/dev/null || true
    done
    sleep 1

    # Tear down compose
    docker compose down --remove-orphans
    sleep 3

    echo "  [session $i] Done. Captures:"
    ls -lh "captures/session${i}_"*.pcap 2>/dev/null | awk '{print "    " $5, $9}' \
        || echo "    (no pcaps found)"
done

echo ""
echo "============================================"
echo " All $NUM_SESSIONS sessions complete."
echo "============================================"
ls -lh captures/
