import torch

B = 2
L = 128
V = 16
seed = 1234

torch.manual_seed(7)
logits = torch.randn(B, L, V)

def global_sample(x, order):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    y = x[order]
    out = torch.multinomial(
        y.softmax(dim=-1).reshape(-1, V),
        num_samples=1,
        replacement=True,
        generator=g,
    ).reshape(len(order), L)
    return {sid: out[i] for i, sid in enumerate(order)}

def per_sample_sample(x, order):
    result = {}
    for sid in order:
        g = torch.Generator(device="cpu")
        g.manual_seed(seed + sid)
        row = x[sid]
        result[sid] = torch.multinomial(
            row.softmax(dim=-1).reshape(-1, V),
            num_samples=1,
            replacement=True,
            generator=g,
        ).reshape(L)
    return result

a = global_sample(logits, [0, 1])
b = global_sample(logits, [1, 0])

global_same = all(torch.equal(a[i], b[i]) for i in [0, 1])
print("GLOBAL_RNG_ORDER_INVARIANT", global_same)

c = per_sample_sample(logits, [0, 1])
d = per_sample_sample(logits, [1, 0])

per_sample_same = all(torch.equal(c[i], d[i]) for i in [0, 1])
print("PER_SAMPLE_RNG_ORDER_INVARIANT", per_sample_same)

assert not global_same
assert per_sample_same
print("RNG_ORDER_TEST_OK")
