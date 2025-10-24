from __future__ import annotations
import argparse, os, json
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    args = ap.parse_args()
    hist_path = os.path.join(args.run_dir, "history.json")
    if not os.path.exists(hist_path):
        print("No history.json found.")
        return
    with open(hist_path) as f:
        H = json.load(f)
    rounds = [h["round"] for h in H]
    auroc = [h["auroc"] for h in H]
    auprc = [h["auprc"] for h in H]
    plt.figure()
    plt.plot(rounds, auroc, label="AUROC")
    plt.plot(rounds, auprc, label="AUPRC")
    plt.xlabel("Round"); plt.ylabel("Score")
    plt.legend()
    out = os.path.join(args.run_dir, "metrics_curve.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print("Saved", out)

if __name__ == "__main__":
    main()
