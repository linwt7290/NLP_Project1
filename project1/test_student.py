"""You should not need to change this file. Run it to check your six functions.

Unit tests for softmax, loss, add_positions, causal_mask, attention and generate in student.py.
"""

import torch
import torch.nn.functional as F

import student

torch.manual_seed(0)
FAILED = []


def check(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
    except NotImplementedError:
        FAILED.append(name)
        print(f"  FAIL  {name} — not written yet")
    except AssertionError as e:
        FAILED.append(name)
        print(f"  FAIL  {name} — {e}")


def t_softmax():
    for shape in [(7,), (2, 5), (3, 4, 9)]:
        x = torch.randn(*shape) * 3
        got = student.softmax(x)
        assert torch.allclose(got, F.softmax(x, dim=-1), atol=1e-5), "does not match softmax"
        assert got.shape == x.shape, "output shape must equal input shape"
        assert torch.allclose(got.sum(dim=-1), torch.ones(x.shape[:-1]), atol=1e-4), \
            "each last-dimension slice must sum to 1"
    hard = torch.tensor([[1000.0, 1000.0, 0.0], [-1000.0, 0.0, 1000.0]])
    got = student.softmax(hard)
    assert torch.isfinite(got).all(), "must stay finite for scores like 1000"
    assert torch.allclose(got, F.softmax(hard, dim=-1), atol=1e-5), "does not match softmax"


def t_loss():
    logits = torch.randn(4, 6, 11)
    targets = torch.randint(0, 11, (4, 6))
    got = student.loss(logits, targets)
    want = F.cross_entropy(logits.view(-1, 11), targets.view(-1))
    assert got.dim() == 0, "must return a scalar"
    assert torch.isfinite(got), "must be finite"
    assert torch.allclose(got, want, atol=1e-5), "does not match cross-entropy"


def t_add_positions():
    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    pos = torch.tensor([[10.0, 0.0], [0.0, 10.0], [5.0, 5.0]])
    want = torch.tensor([[[11.0, 2.0], [3.0, 14.0]]])
    got = student.add_positions(x, pos)
    assert got.shape == x.shape, "output shape must equal x's shape"
    assert torch.allclose(got, want, atol=1e-6), "wrong result on the hand-checked example"

    y = torch.randn(2, 5, 8)
    got = student.add_positions(y, torch.randn(9, 8))
    assert got.shape == y.shape, "output shape must equal x's shape"


def t_causal_mask():
    m = student.causal_mask(4)
    want = torch.tensor([
        [True, False, False, False],
        [True, True, False, False],
        [True, True, True, False],
        [True, True, True, True],
    ])
    assert m.dtype == torch.bool, "must be boolean"
    assert m.shape == (4, 4), "must be (T, T)"
    assert m.device.type == "cpu", "must be created on the CPU"
    assert torch.equal(m, want), "True exactly where the column does not come after the row"


def t_attention():
    B, H, T, dh = 2, 3, 5, 4
    q, k, v = (torch.randn(B, H, T, dh, dtype=torch.float64) for _ in range(3))
    mask = torch.tensor([
        [True, False, False, False, False],
        [True, True, False, False, False],
        [True, True, True, False, False],
        [True, True, True, True, False],
        [True, True, True, True, True],
    ])
    got = student.attention(q, k, v, mask)
    want = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    assert got.shape == q.shape, "output shape must equal q's shape"
    assert torch.allclose(got, want, atol=1e-10), "does not match scaled dot-product attention"


class StubModel:
    def __init__(self, logits, block_size=4):
        self.logits = logits
        self.block_size = block_size
        self.seen = []

    def __call__(self, idx):
        self.seen.append(idx.shape[1])
        return self.logits.view(1, 1, -1).expand(1, idx.shape[1], -1)


def t_generate():
    stub = StubModel(torch.tensor([0.1, 2.0, -0.5]))
    start = torch.tensor([[5, 2]])

    torch.manual_seed(0)
    out = student.generate(stub, start, 6, temperature=0.01, top_p=1.0)
    assert out.shape == (1, 8), "must return the start ids with 6 new ids appended"
    assert torch.equal(out[0, :2], start[0]), "must keep the start ids"
    assert (out[0, 2:] == 1).all(), "at tiny temperature the top character must always win"

    torch.manual_seed(0)
    out_a = student.generate(stub, start, 6, temperature=1.0, top_p=0.4)
    torch.manual_seed(0)
    out_b = student.generate(stub, start, 6, temperature=1.0, top_p=0.4)
    assert torch.equal(out_a, out_b), "same seed must give the same poem"
    assert (out_a[0, 2:] == 1).all(), "top_p below the top probability must keep only the top character"

    stub = StubModel(torch.tensor([0.1, 2.0, -0.5]), block_size=4)
    student.generate(stub, start, 10, temperature=1.0, top_p=1.0)
    assert max(stub.seen) <= 4, f"never send the model more than block_size characters (sent {max(stub.seen)})"


check("softmax", t_softmax)
check("loss", t_loss)
check("add_positions", t_add_positions)
check("causal_mask", t_causal_mask)
check("attention", t_attention)
check("generate", t_generate)

print()
if FAILED:
    raise SystemExit(f"{len(FAILED)} of 6 functions fail: {', '.join(FAILED)}")
print("All six functions pass. You are ready to train: python train.py")
