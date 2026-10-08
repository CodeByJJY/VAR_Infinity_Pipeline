import torch
import torch.nn as nn


def kv_modules(model):
    return [
        m for m in model.modules()
        if hasattr(m, "cached_k")
        and hasattr(m, "cached_v")
        and hasattr(m, "caching")
    ]


def split_cfg_kv_cache(model, logical_batch):
    modules = kv_modules(model)
    states = [[] for _ in range(logical_batch)]

    for m in modules:
        k = m.cached_k
        v = m.cached_v

        if k is None or v is None:
            for s in states:
                s.append((None, None))
            continue

        assert k.shape[0] == 2 * logical_batch
        assert v.shape[0] == 2 * logical_batch

        for i in range(logical_batch):
            idx = torch.tensor(
                [i, logical_batch + i],
                device=k.device,
                dtype=torch.long,
            )
            ki = k.index_select(0, idx).detach().clone()
            vi = v.index_select(0, idx).detach().clone()
            states[i].append((ki, vi))

    return states


def load_cfg_kv_group(model, sample_states):
    modules = kv_modules(model)

    assert len(sample_states) > 0
    assert all(len(s) == len(modules) for s in sample_states)

    for j, m in enumerate(modules):
        entries = [s[j] for s in sample_states]

        if entries[0][0] is None:
            m.cached_k = None
            m.cached_v = None
            m.caching = True
            continue

        ks = [x[0] for x in entries]
        vs = [x[1] for x in entries]

        assert all(x.shape[0] == 2 for x in ks)
        assert all(x.shape[0] == 2 for x in vs)

        cond_k = torch.cat([x[0:1] for x in ks], dim=0)
        uncond_k = torch.cat([x[1:2] for x in ks], dim=0)
        cond_v = torch.cat([x[0:1] for x in vs], dim=0)
        uncond_v = torch.cat([x[1:2] for x in vs], dim=0)

        m.cached_k = torch.cat([cond_k, uncond_k], dim=0).contiguous()
        m.cached_v = torch.cat([cond_v, uncond_v], dim=0).contiguous()
        m.caching = True


def load_cfg_kv_group_consume(model, sample_states):
    modules = kv_modules(model)

    assert len(sample_states) > 0
    assert all(len(state) == len(modules) for state in sample_states)

    for j, m in enumerate(modules):
        entries = [state[j] for state in sample_states]

        if entries[0][0] is None:
            m.cached_k = None
            m.cached_v = None
            m.caching = True
        elif len(entries) == 1:
            m.cached_k = entries[0][0]
            m.cached_v = entries[0][1]
            m.caching = True
        else:
            ks = [x[0] for x in entries]
            vs = [x[1] for x in entries]

            assert all(x.shape[0] == 2 for x in ks)
            assert all(x.shape[0] == 2 for x in vs)

            m.cached_k = torch.cat(
                [x[0:1] for x in ks] + [x[1:2] for x in ks],
                dim=0,
            ).contiguous()
            m.cached_v = torch.cat(
                [x[0:1] for x in vs] + [x[1:2] for x in vs],
                dim=0,
            ).contiguous()
            m.caching = True

        for state in sample_states:
            state[j] = (None, None)


def split_cfg_kv_cache_release(model, logical_batch):
    modules = kv_modules(model)
    states = [[] for _ in range(logical_batch)]

    for m in modules:
        k = m.cached_k
        v = m.cached_v

        if k is None or v is None:
            for state in states:
                state.append((None, None))
        else:
            assert k.shape[0] == 2 * logical_batch
            assert v.shape[0] == 2 * logical_batch

            if logical_batch == 1:
                states[0].append((k.detach(), v.detach()))
            else:
                for i in range(logical_batch):
                    ki = torch.cat(
                        (k[i:i+1], k[logical_batch+i:logical_batch+i+1]),
                        dim=0,
                    ).detach()
                    vi = torch.cat(
                        (v[i:i+1], v[logical_batch+i:logical_batch+i+1]),
                        dim=0,
                    ).detach()
                    states[i].append((ki, vi))

        m.cached_k = None
        m.cached_v = None
        m.caching = False

    return states


def clear_kv_cache(model):
    for m in kv_modules(model):
        m.cached_k = None
        m.cached_v = None
        m.caching = False


if __name__ == "__main__":
    class FakeAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.caching = True
            self.cached_k = None
            self.cached_v = None

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = FakeAttn()
            self.b = FakeAttn()

    model = FakeModel()
    B = 3

    for n, m in enumerate(kv_modules(model)):
        base = torch.arange(2 * B).view(2 * B, 1, 1, 1) + 100 * n
        m.cached_k = base.clone()
        m.cached_v = base.clone() + 1000

    states = split_cfg_kv_cache(model, B)

    load_cfg_kv_group(model, [states[2], states[0]])

    expected0 = torch.tensor([2, 0, 5, 3])
    got0 = kv_modules(model)[0].cached_k[:, 0, 0, 0].cpu()

    assert torch.equal(got0, expected0), (got0, expected0)
    assert kv_modules(model)[0].cached_k.shape[0] == 4

    print("KV_CACHE_STATE_TEST_OK")
    print("regrouped physical order =", got0.tolist())
    print("expected CFG order       = [cond2, cond0, uncond2, uncond0]")
