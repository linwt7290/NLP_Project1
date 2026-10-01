"""You should not need to change this file. Run: python train.py

Trains the Language Model on poems.txt with your config.json hyperparameters (seeded by your student ID),
then saves the weights, vocab and loss curve into output/.
"""

import argparse
import json
import os
import time
import zlib

import torch

import student
from evaluate import measure
from model import GPT


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


def train_loss_estimate(model, data, cfg, device, n=10):
    was_training = model.training
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(n):
            x, y = get_batch(data, cfg["block_size"], cfg["batch_size"], device)
            losses.append(student.loss(model(x), y).item())
    if was_training:
        model.train()
    return sum(losses) / len(losses)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=".", help="folder holding config.json, poems.txt, val.txt")
    ap.add_argument("--out", default="output", help="folder to save the model and loss.png into")
    args = ap.parse_args()

    with open(os.path.join(args.data, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    sid = str(cfg.get("student_id", "")).strip()
    if not sid:
        raise SystemExit('Put your student number in config.json first: "student_id".')

    torch.manual_seed(zlib.crc32(sid.encode("utf-8")))

    device = pick_device()
    print(f"student {sid}   device {device}", flush=True)

    poems_text = open(os.path.join(args.data, "poems.txt"), encoding="utf-8").read()
    val_text = open(os.path.join(args.data, "val.txt"), encoding="utf-8").read()
    chars = sorted(set(poems_text) | set(val_text))
    stoi = {c: i for i, c in enumerate(chars)}
    train_data = torch.tensor([stoi[c] for c in poems_text])
    val_data = torch.tensor([stoi[c] for c in val_text])
    print(f"poems {poems_text.count(chr(10)):,}   vocab {len(chars):,}", flush=True)

    model = GPT(len(chars), cfg).to(device)
    print(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"],
                            weight_decay=cfg["weight_decay"])

    history = []
    t0 = time.time()
    for step in range(1, cfg["max_steps"] + 1):
        x, y = get_batch(train_data, cfg["block_size"], cfg["batch_size"], device)
        loss = student.loss(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % cfg["eval_interval"] == 0 or step == cfg["max_steps"]:
            tr = train_loss_estimate(model, train_data, cfg, device)
            va = measure(model, val_data)
            history.append((step, tr, va))
            print(f"step {step:5d}  train {tr:.3f}  val {va:.3f}  {time.time() - t0:6.0f}s",
                  flush=True)

    os.makedirs(args.out, exist_ok=True)
    sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    from safetensors.torch import save_file

    save_file(sd, os.path.join(args.out, f"{sid}.safetensors"))
    with open(os.path.join(args.out, f"{sid}.json"), "w", encoding="utf-8") as f:
        json.dump({**cfg, "chars": chars, "val_loss": history[-1][2]}, f, ensure_ascii=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [h[0] for h in history]
    plt.plot(steps, [h[1] for h in history], label="train")
    plt.plot(steps, [h[2] for h in history], label="validation")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(os.path.join(args.out, "loss.png"), dpi=120)
    print(f"saved {sid}.safetensors, {sid}.json, loss.png  ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
