import os
import flwr as fl

NUM_ROUNDS = int(os.environ.get("NUM_ROUNDS", "5"))

print(f"[server] Starting Flower server with NUM_ROUNDS={NUM_ROUNDS}")

strategy = fl.server.strategy.FedAvg(
    min_fit_clients=2,
    min_evaluate_clients=2,
    min_available_clients=2,
)

if __name__ == "__main__":
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )
