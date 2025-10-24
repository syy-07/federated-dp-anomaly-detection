from src.data import make_synthetic_anomaly_data, split_by_clients

def test_splits():
    (Xtr, ytr), _ = make_synthetic_anomaly_data(n=1000, dim=16, anomaly_rate=0.05, seed=1)
    parts = split_by_clients(Xtr, ytr, clients=4, iid=False, seed=1)
    assert len(parts) == 4
    assert sum(len(p[0]) for p in parts) == len(Xtr)
