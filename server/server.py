import flwr as fl

strategy = fl.server.strategy.FedAvg(
    min_fit_clients=2,
    min_evaluate_clients=2,
    min_available_clients=2,
)

if __name__ == "__main__":
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )
