from __future__ import annotations
import argparse, os, json, numpy as np, torch, yaml
from sklearn.metrics import roc_auc_score, average_precision_score
from src.models import AE
from src.utils import timestamp

def eval_run(run_dir):
    with open(os.path.join(run_dir, "cfg.json")) as f:
        cfg = json.load(f)
    ckpt = os.path.join(run_dir, "global_final.pt")
    if not os.path.exists(ckpt):
        # fallback: last available
        cands = sorted([p for p in os.listdir(run_dir) if p.endswith(".pt")])
        ckpt = os.path.join(run_dir, cands[-1])
    model = AE(cfg["input_dim"], cfg["hidden_dims"], cfg["latent_dim"])
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()

    # Recreate test data deterministically
    from src.data import make_synthetic_anomaly_data
    (Xtr, ytr), (Xte, yte) = make_synthetic_anomaly_data(
        n=cfg["data"]["total_samples"],
        dim=cfg["input_dim"],
        anomaly_rate=cfg["data"]["anomaly_rate"],
        seed=cfg["seed"]
    )
    with torch.no_grad():
        xr = model(torch.tensor(Xte, dtype=torch.float32)).numpy()
    errs = ((Xte - xr)**2).mean(axis=1)
    return dict(
        auroc=float(roc_auc_score(yte, errs)),
        auprc=float(average_precision_score(yte, errs)),
        samples=int(len(yte))
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    args = ap.parse_args()
    res = eval_run(args.run_dir)
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
