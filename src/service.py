# Simple REST service to load a checkpoint and score inputs (for demo/Helm).
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import torch
from src.models import AE

CKPT = None
MODEL = None

def load_model(ckpt_path, input_dim, hidden_dims, latent_dim):
    global CKPT, MODEL
    MODEL = AE(input_dim, hidden_dims, latent_dim)
    MODEL.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    MODEL.eval()
    CKPT = ckpt_path

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        data = json.loads(body.decode())
        X = torch.tensor(data["X"], dtype=torch.float32)
        with torch.no_grad():
            xr = MODEL(X).numpy()
        errs = ((X.numpy() - xr)**2).mean(axis=1).tolist()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"recon_error": errs, "ckpt": CKPT}).encode())

if __name__ == "__main__":
    # Lazy defaults; in practice pass via env vars or args.
    load_model("runs/example/global_final.pt", 32, [64,32], 12)
    HTTPServer(("0.0.0.0", 8080), Handler).serve_forever()
