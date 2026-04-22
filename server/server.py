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
    min_fit_clients=2,
    min_evaluate_clients=2,
    min_available_clients=2,
)

if __name__ == "__main__":
    print(f"[SERVER_START] {int(time.time() * 1000)}", flush=True)
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )
