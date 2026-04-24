"""Contrastive alignment head and loss for the LaProteina VAE.

The encoder is run twice per training step — once on the full batch (Pass A),
once on a shallow-copied batch with a configurable set of keys zeroed (Pass B,
structure-only by default). A small projection head maps each view of the
per-residue encoder mean to a unit-norm embedding; a symmetric per-residue
InfoNCE with batch-level negatives pulls the two views into alignment.
"""

from typing import Dict, Iterable, List

import math

import torch
import torch.nn.functional as F
from torch import nn


class ContrastiveHead(nn.Module):
    """Shared projection head for both views plus a learnable logit scale."""

    def __init__(
        self,
        in_dim: int,
        hidden: int = 64,
        out_dim: int = 128,
        init_temperature: float = 0.07,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim, bias=False),
        )
        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(1.0 / init_temperature), dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


def build_masked_batch(
    batch: Dict[str, torch.Tensor], keys: Iterable[str]
) -> Dict[str, torch.Tensor]:
    """Return a shallow copy of batch with the listed keys replaced by zeros.

    The input batch dict is not mutated; only the entries named in `keys` are
    swapped for a zero tensor of the same shape/dtype/device. Missing keys are
    silently skipped (strict_feats=False in the encoder config means the
    FeatureFactory tolerates absent entries).
    """
    out = dict(batch)
    for k in keys:
        v = out.get(k)
        if isinstance(v, torch.Tensor):
            out[k] = torch.zeros_like(v)
    return out


def symmetric_infonce(
    h_a: torch.Tensor,
    h_b: torch.Tensor,
    mask: torch.Tensor,
    logit_scale: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Symmetric per-residue InfoNCE with batch-level negatives.

    Args:
        h_a: projected view A, shape [b, n, d].
        h_b: projected view B, shape [b, n, d].
        mask: boolean residue mask [b, n].
        logit_scale: learnable scalar, used as exp(logit_scale).clamp(max=100).

    Returns:
        Dict with:
            - "loss": scalar InfoNCE loss, averaged over the two directions.
            - "pos_sim": mean cosine similarity of positive pairs (scalar, detached).
            - "temperature": exp(-logit_scale), detached.
    """
    a = F.normalize(h_a, dim=-1)
    b = F.normalize(h_b, dim=-1)

    idx = mask.nonzero(as_tuple=False)  # [M, 2]
    za = a[idx[:, 0], idx[:, 1]]  # [M, d]
    zb = b[idx[:, 0], idx[:, 1]]  # [M, d]

    scale = logit_scale.exp().clamp(max=100.0)
    logits = (za @ zb.T) * scale  # [M, M]
    target = torch.arange(logits.shape[0], device=logits.device)
    loss_ab = F.cross_entropy(logits, target)
    loss_ba = F.cross_entropy(logits.T, target)
    loss = 0.5 * (loss_ab + loss_ba)

    pos_sim = (za * zb).sum(dim=-1).mean().detach()
    temperature = (1.0 / scale).detach()

    return {"loss": loss, "pos_sim": pos_sim, "temperature": temperature}


def warmup_weight(
    step: int, start: int, end: int, target_weight: float
) -> float:
    """Linear warmup from 0 to target_weight between [start, end] steps."""
    if end <= start:
        return target_weight if step >= start else 0.0
    frac = (step - start) / float(end - start)
    return target_weight * max(0.0, min(1.0, frac))
