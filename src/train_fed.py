from __future__ import annotations
import argparse, os, json
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from src.utils import set_seed, ensure_dir, timestamp, save_json
from src.data import make_synthetic_anomaly_data, split_by_clients
from src.models import AE
from src.federated import make_client_loaders, fedavg, select_clients
from src.dp import clip_gradients_per_sample, add_gaussian_noise, rdp_gaussian
import yaml
from sklearn.metrics import roc_auc_score, average_precision_score

def loss_recon(x, xrec):
    return ((x - xrec)**2).mean()

def evaluate(model, X, y, device):
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(X, dtype=torch.float32, device=device)
        xr = model(xb).cpu().numpy()
    errs = ((X - xr)**2).mean(axis=1)
    auroc = roc_auc_score(y, errs)
    auprc = average_precision_score(y, errs)
    return dict(auroc=float(auroc), auprc=float(auprc))

def train_local_dp(model, loader, epochs, lr, clip_norm, noise_multiplier, device):
    model.train()
    opt = optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        for xb, _ in loader:
            xb = xb.to(device)
            opt.zero_grad()
            # forward
            xr = model(xb)
            loss = ((xb - xr)**2).mean()
            # compute per-sample grads via microbatch = 1 (simplified for demo)
            # This is computationally heavy in practice; here, we loop for clarity.
            per_param_grads = [torch.zeros((xb.size(0), *p.shape), device=device) for p in model.parameters()]
            for i in range(xb.size(0)):
                opt.zero_grad(set_to_none=True)
                xi = xb[i:i+1]
                li = ((xi - model(xi))**2).mean()
                li.backward(retain_graph=True)
                for j, p in enumerate(model.parameters()):
                    per_param_grads[j][i] = p.grad.detach().clone()
            # clip per-sample grads
            clipped_grads, _ = clip_gradients_per_sample(per_param_grads, clip_norm)
            # aggregate
            agg_grads = [g.sum(dim=0, keepdim=True) for g in clipped_grads]
            # add noise
            add_gaussian_noise(agg_grads, noise_multiplier, clip_norm, xb.size(0), device)
            # set grads on params and step
            for p, g in zip(model.parameters(), agg_grads):
                p.grad = g.squeeze(0) / xb.size(0)
            opt.step()
    return model.state_dict()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/fed_dp_small.yml")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    (Xtr, ytr), (Xte, yte) = make_synthetic_anomaly_data(
        n=cfg["data"]["total_samples"],
        dim=cfg["input_dim"],
        anomaly_rate=cfg["data"]["anomaly_rate"],
        seed=cfg["seed"]
    )
    clients = split_by_clients(Xtr, ytr, clients=cfg["clients"], iid=cfg["data"].get("iid", False), seed=cfg["seed"])
    loaders = make_client_loaders(clients, batch_size=cfg["batch_size"])

    # Model
    model = AE(input_dim=cfg["input_dim"], hidden_dims=cfg["hidden_dims"], latent_dim=cfg["latent_dim"]).to(device)
    global_w = model.state_dict()

    run_dir = os.path.join("runs", f"fed_dp_{timestamp()}")
    ensure_dir(run_dir)
    save_json(os.path.join(run_dir, "cfg.json"), cfg)

    history = []
    steps = 0
    for r in range(cfg["rounds"]):
        sel = select_clients(len(loaders), cfg["participation_rate"])
        w_local = []
        for cid in sel:
            # load global weights
            model.load_state_dict(global_w)
            # local train
            if cfg["dp"]["enabled"]:
                sd = train_local_dp(
                    model, loaders[cid], cfg["local_epochs"], cfg["lr"],
                    cfg["dp"]["clip_norm"], cfg["dp"]["noise_multiplier"], device
                )
                # dp steps approximation
                for _ in loaders[cid]:
                    steps += 1
            else:
                # standard local update
                opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
                model.train()
                for _ in range(cfg["local_epochs"]):
                    for xb, _ in loaders[cid]:
                        xb = xb.to(device)
                        opt.zero_grad()
                        xr = model(xb); loss = ((xb - xr)**2).mean()
                        loss.backward(); opt.step()
                sd = model.state_dict()
            w_local.append(sd)
        # aggregate
        global_w = fedavg(w_local)
        model.load_state_dict(global_w)

        # evaluation
        if (r+1) % cfg["metrics"]["eval_every"] == 0:
            m = evaluate(model, Xte, yte, device)
            rec = {"round": r+1, **m}
            # rough privacy accounting
            if cfg["dp"]["enabled"]:
                q = cfg["participation_rate"]  # rough proxy; real q depends on sampling
                rdp = rdp_gaussian(q, cfg["dp"]["noise_multiplier"], steps)
                rec["rdp_eps"] = {str(int(a)): float(eps) for a, eps in rdp}
            history.append(rec)
            print(rec)

        # save checkpoints
        if (r+1) % cfg["save_every"] == 0:
            torch.save(global_w, os.path.join(run_dir, f"global_round_{r+1}.pt"))
            save_json(os.path.join(run_dir, "history.json"), history)

    # final save
    torch.save(global_w, os.path.join(run_dir, f"global_final.pt"))
    save_json(os.path.join(run_dir, "history.json"), history)
    print("Saved:", run_dir)

if __name__ == "__main__":
    main()
