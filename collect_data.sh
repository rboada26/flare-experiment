#!/usr/bin/env bash
set -euo pipefail

NUM_SESSIONS=8

wait_for_training() {
    echo "Polling for training completion..."
    while true; do
        # Check if all 4 client containers have exited
        CNN1=$(docker inspect -f '{{.State.Status}}' fl_cnn_client  2>/dev/null || echo "gone")
        CNN2=$(docker inspect -f '{{.State.Status}}' fl_cnn_client2 2>/dev/null || echo "gone")
        RNN1=$(docker inspect -f '{{.State.Status}}' fl_rnn_client  2>/dev/null || echo "gone")
        RNN2=$(docker inspect -f '{{.State.Status}}' fl_rnn_client2 2>/dev/null || echo "gone")

        echo "  Status — cnn_client: $CNN1 | cnn_client2: $CNN2 | rnn_client: $RNN1 | rnn_client2: $RNN2"

        if [[ "$CNN1" == "exited" && "$CNN2" == "exited" && \
              "$RNN1" == "exited" && "$RNN2" == "exited" ]]; then
            echo "All clients finished training."
            break
        fi

        sleep 5
    done
}

for i in $(seq 1 $NUM_SESSIONS); do
    LABEL="session${i}"
    echo ""
    echo "============================================"
    echo " Starting session $i / $NUM_SESSIONS"
    echo "============================================"

    # Clean slate
    docker compose down --remove-orphans 2>/dev/null || true
    sleep 2

    # Step 1: bring up networks and servers only
    docker compose up -d cnn_server rnn_server
    sleep 8

    # Step 2: find bridge interfaces and start capture
    CNN_ID=$(docker network inspect flare-experiment_cnn_network \
        --format '{{.Id}}' 2>/dev/null | cut -c1-12)
    RNN_ID=$(docker network inspect flare-experiment_rnn_network \
        --format '{{.Id}}' 2>/dev/null | cut -c1-12)
    CNN_IFACE="br-${CNN_ID}"
    RNN_IFACE="br-${RNN_ID}"

    mkdir -p captures
    CNN_PCAP="captures/${LABEL}_cnn_$(date +%Y%m%d_%H%M%S).pcap"
    RNN_PCAP="captures/${LABEL}_rnn_$(date +%Y%m%d_%H%M%S).pcap"

    CNN_FILTER="(src net 172.20.0.0/24 and dst net 172.20.0.0/24) and not multicast"
    RNN_FILTER="(src net 172.21.0.0/24 and dst net 172.21.0.0/24) and not multicast"

    sudo tcpdump -i "$CNN_IFACE" -w "$CNN_PCAP" -s0 "$CNN_FILTER" &
    PID_CNN=$!
    sudo tcpdump -i "$RNN_IFACE" -w "$RNN_PCAP" -s0 "$RNN_FILTER" &
    PID_RNN=$!

    sleep 2

    # Step 3: start clients
    docker compose up -d cnn_client cnn_client2 rnn_client rnn_client2

    echo "Capture PIDs: $PID_CNN (CNN) $PID_RNN (RNN)"

    # Step 4: poll until all clients exit
    wait_for_training

    # Let last packets flush
    sleep 3

    # Stop capture
    sudo kill $PID_CNN 2>/dev/null || true
    sudo kill $PID_RNN 2>/dev/null || true
    wait $PID_CNN 2>/dev/null || true
    wait $PID_RNN 2>/dev/null || true

    # Tear down
    docker compose down --remove-orphans
    sleep 3

    echo "Session $i done. Captures so far:"
    ls -lh captures/*.pcap 2>/dev/null | awk '{print $5, $9}'
done

echo ""
echo "All sessions complete."
ls -lh captures/
```

The key change is `wait_for_training()` which polls `docker inspect` every 5 seconds and prints the status of each container so you can see exactly what's happening. Run it and you should see output like:
```
Status — cnn_client: running | cnn_client2: running | rnn_client: running | rnn_client2: running
Status — cnn_client: running | cnn_client2: running | rnn_client: exited | rnn_client2: exited
Status — cnn_client: exited  | cnn_client2: exited  | rnn_client: exited  | rnn_client2: exited
All clients finished training.
