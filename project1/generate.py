"""You should not need to change this file. Run: python generate.py [first characters]

Prints one sample poem using the temperature and top-p from your config,
flagging lines copied verbatim from the Tang corpus with *.
"""

import argparse
import os
import re
import sys

import torch

import student
from evaluate import load_model

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("start", nargs="?", default="", help="first characters of the poem")
    args = ap.parse_args()

    model, saved = load_model(".", "output")
    chars = saved["chars"]
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}

    missing = [c for c in "\n" + args.start if c not in stoi]
    if missing:
        raise SystemExit(f"characters not in the model's vocabulary: {''.join(missing)!r}")

    known = set()
    for name in ("poems.txt", "val.txt", "withheld.txt"):
        p = name
        if os.path.exists(p):
            known |= set(re.split(r"[，。？！；\n]", open(p, encoding="utf-8").read()))
    known.discard("")

    print(f"temperature {saved['temperature']}   top-p {saved['top_p']}"
          + ("   (* = line copied verbatim from the Tang corpus)" if known else ""))
    torch.manual_seed(0)
    with torch.no_grad():
        idx = torch.tensor([[stoi[c] for c in "\n" + args.start]])
        out = student.generate(model, idx, 80, saved["temperature"], saved["top_p"])
        poem = "".join(itos[i] for i in out[0].tolist())[1:]
        poem = poem.split("\n")[0]
        copied = [l for l in re.split(r"[，。]", poem) if l and l in known]
        flag = f"   * copied: {' '.join(copied)}" if copied else ""
        print(poem + flag)


if __name__ == "__main__":
    main()
