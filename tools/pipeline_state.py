import torch
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class PipelineSampleState:
    sample_id: int
    scale_idx: int
    last_stage: Optional[torch.Tensor] = None
    summed_codes: Optional[torch.Tensor] = None
    cond_BD: Optional[torch.Tensor] = None
    cond_BD_or_gss: Optional[torch.Tensor] = None
    ca_kv: Any = None
    kv_cache: Any = None
    rng_state: Optional[torch.Tensor] = None


def split_cfg_tensor(x, logical_batch):
    assert x.shape[0] == 2 * logical_batch
    return [
        torch.cat((x[i:i+1], x[logical_batch+i:logical_batch+i+1]), dim=0)
        for i in range(logical_batch)
    ]


def merge_cfg_tensor(xs):
    assert len(xs) > 0
    assert all(x.shape[0] == 2 for x in xs)
    cond = torch.cat([x[0:1] for x in xs], dim=0)
    uncond = torch.cat([x[1:2] for x in xs], dim=0)
    return torch.cat((cond, uncond), dim=0).contiguous()


def split_logical_tensor(x, logical_batch):
    assert x.shape[0] == logical_batch
    return [x[i:i+1] for i in range(logical_batch)]


def merge_logical_tensor(xs):
    assert len(xs) > 0
    return torch.cat(xs, dim=0).contiguous()


def split_ca_kv(ca_kv, logical_batch):
    kv, cu, max_len = ca_kv
    assert cu.numel() == 2 * logical_batch + 1

    pieces = []
    for i in range(logical_batch):
        p0, p1 = int(cu[i]), int(cu[i+1])
        u0, u1 = int(cu[logical_batch+i]), int(cu[logical_batch+i+1])

        pos = kv[p0:p1]
        neg = kv[u0:u1]

        sample_kv = torch.cat((pos, neg), dim=0)
        sample_cu = torch.tensor(
            [0, len(pos), len(pos) + len(neg)],
            dtype=cu.dtype,
            device=cu.device,
        )
        pieces.append((sample_kv, sample_cu, max(len(pos), len(neg))))

    return pieces


def merge_ca_kv(parts):
    assert len(parts) > 0

    positives = []
    negatives = []

    for kv, cu, _ in parts:
        cut = int(cu[1])
        positives.append(kv[:cut])
        negatives.append(kv[cut:])

    seqs = positives + negatives
    merged = torch.cat(seqs, dim=0)

    lengths = [len(x) for x in seqs]
    offsets = [0]
    for n in lengths:
        offsets.append(offsets[-1] + n)

    cu = torch.tensor(
        offsets,
        dtype=parts[0][1].dtype,
        device=merged.device,
    )

    return merged, cu, max(lengths)
