"""The six functions YOU write.

Each docstring says what the function does and exactly what goes in and out.
Run `python test_student.py` after you write them. Do not go on until all six pass.

These functions are the only thing we take from your folder and run with our
own copy of `model.py`, so they must do exactly what the docstring says.
"""

import torch


def softmax(logits):
    """Turn scores into probabilities.

    In:  logits — any shape of float scores, e.g. (B, T, V) or (V,).
    Out: same shape; each slice along the LAST dimension is non-negative and
         sums to 1. Must stay finite for scores like 1000.
    """
    return torch.softmax(logits, dim=-1)


def loss(logits, targets):
    """Cross-entropy: how surprised the model was by the real next character.

    In:  logits  — (B, T, V) scores from the model.
         targets — (B, T) integer character ids (the real next characters).
    Out: a scalar — the MEAN over all B*T positions, in natural log (nats),
         like torch.nn.functional.cross_entropy.
    """
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
    )


def add_positions(x, pos):
    """Add position information to token embeddings.

    In:  x   — (B, T, C) token embeddings.
         pos — (max_T, C) position embedding matrix (model.pos.weight).
    Out: (B, T, C) — x with position t receiving row pos[t].
    """
    return x + pos[:x.size(1)]


def causal_mask(block_size):
    """The mask that stops each character from seeing the characters after it.

    In:  block_size — an integer T.
    Out: a (T, T) boolean tensor, ON THE CPU, where entry [i, j] is True when
         position i is ALLOWED to look at position j (j <= i) and False
         otherwise. model.py moves it to the right device for you.
    """
    return torch.ones((block_size, block_size), dtype=torch.bool, device="cpu").tril_()


def attention(q, k, v, mask):
    """Aggregate information from the earlier characters.

    In:  q, k, v — (B, H, T, d_head) each: queries, keys, values per head.
         mask    — (T, T) boolean, True = allowed to attend (from causal_mask).
    Out: (B, H, T, d_head) — for each head: the values v weighted by softmax
         over allowed positions of (q·k / sqrt(d_head)).
    """
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)


def generate(model, start_ids, max_new_tokens, temperature, top_p):
    """Write one character at a time, using temperature and top-p.

    In:  model          — a GPT from model.py; model(ids) gives logits (1, t, V),
                          and model.block_size is the longest context it takes.
         start_ids      — (1, t0) integer ids to start from (already on the
                          model's device).
         max_new_tokens — how many characters to add.
         temperature    — float > 0, divides the scores before softmax.
         top_p          — float in (0, 1]; keep the most likely characters until
                          their probabilities add up to top_p, drop the rest,
                          then sample from what is left. top_p = 1 keeps all.
    Out: (1, t0 + max_new_tokens) — the start ids with the new ids appended.
    """
    if max_new_tokens <= 0:
        return start_ids

    start_length = start_ids.size(1)
    ids = start_ids.new_empty((1, start_length + max_new_tokens))
    ids[:, :start_length] = start_ids
    with torch.no_grad():
        for end in range(start_length, ids.size(1)):
            logits = model(ids[:, max(0, end - model.block_size):end])[:, -1, :]
            probabilities = torch.softmax(logits / temperature, dim=-1)
            if top_p < 1.0:
                probabilities, indices = probabilities.sort(dim=-1, descending=True)
                cumulative = probabilities.cumsum(dim=-1)
                remove = cumulative > top_p
                remove[:, 1:] = remove[:, :-1].clone()
                remove[:, 0] = False
                probabilities.masked_fill_(remove, 0)
                sampled = torch.multinomial(probabilities, num_samples=1)
                next_id = indices.gather(-1, sampled)
            else:
                next_id = torch.multinomial(probabilities, num_samples=1)
            ids[:, end:end + 1] = next_id
    return ids
