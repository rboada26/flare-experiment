import os
import sys
import time
import flwr as fl

NUM_ROUNDS = int(os.environ.get("NUM_ROUNDS", "5"))
print(f"[server] Starting Flower server with NUM_ROUNDS={NUM_ROUNDS}", flush=True)


class InstrumentedFedAvg(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        result = super().aggregate_fit(server_round, results, failures)
        # Include millisecond Unix timestamp so the dashboard can assign accurate per-round timing
        print(f"[ROUND_COMPLETE] {server_round} {int(time.time() * 1000)}", flush=True)
        return result


strategy = InstrumentedFedAvg(
    min_fit_clients=3,
    min_evaluate_clients=3,
    min_available_clients=3,
)

if __name__ == "__main__":
    _mist = os.environ.get('MIST_ENABLED', '').lower() == 'true'
    if _mist:
        import threading, asyncio
        from mist_proxy import MISTConfig, run_server_proxy
        _cfg = MISTConfig(
            int(os.environ.get('MIST_P_FIXED', '262144')),
            int(os.environ.get('MIST_RATE', '10')),
            float(os.environ.get('MIST_SESSION_DURATION', '0')),
        )
        # MIST proxy listens on :9090 (clients connect here), forwards decoded to localhost:8080
        threading.Thread(
            target=lambda: asyncio.run(run_server_proxy(9090, 'localhost', 8080, _cfg)),
            daemon=True,
        ).start()
        time.sleep(0.5)
        _bind = 'localhost:8080'   # Flower is internal-only; MIST proxy handles external traffic
        print(f'[MIST-server] proxy up on :9090 → localhost:8080', flush=True)
    else:
        _bind = '0.0.0.0:8080'

    print(f"[SERVER_START] {int(time.time() * 1000)}", flush=True)
    fl.server.start_server(
        server_address=_bind,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )
