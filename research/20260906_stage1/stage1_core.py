"""Small, tested data and attention components for the information screen."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def atomic_json(path, record):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(record, indent=2), encoding='utf-8')
    temporary.replace(path)


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def internal_holdout(users, seed):
    """Hold out one training interaction per eligible user, keeping history."""
    users = np.asarray(users)
    order = np.argsort(users, kind='stable')
    boundaries = np.r_[0, np.flatnonzero(np.diff(users[order])) + 1, len(users)]
    rng = np.random.RandomState(seed)
    selected = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end - start >= 2:
            selected.append(order[start + rng.randint(end - start)])
    return np.asarray(sorted(selected), dtype=np.int64)


def matched_permutation(mask, seed):
    """Shuffle content within identical missingness patterns."""
    rng = np.random.RandomState(seed)
    groups = {}
    for index, row in enumerate(np.asarray(mask, dtype=np.bool_)):
        groups.setdefault(row.tobytes(), []).append(index)
    permutation = np.arange(len(mask))
    for indices in groups.values():
        permutation[indices] = rng.permutation(indices)
    return permutation


def leave_one_out_profile(sums, counts, item_features, users, excluded_items=None):
    total = sums[users]
    count = counts[users]
    if excluded_items is not None:
        total = total - item_features[excluded_items]
        count = count - 1
    profile = total / count.clamp_min(1).unsqueeze(-1)
    return F.normalize(profile, dim=-1)


class EvidenceMatcher(nn.Module):
    def __init__(self, global_features, global_mask, local_bank, local_mask, variant, seed=731):
        super().__init__()
        if variant not in ('global', 'local', 'shuffled'):
            raise ValueError('Unknown evidence variant')
        if variant == 'global':
            bank, mask = global_features, global_mask
        else:
            if variant == 'shuffled':
                permutation = matched_permutation(local_mask, seed)
                local_bank, local_mask = local_bank[permutation], local_mask[permutation]
            bank = np.concatenate((global_features, local_bank), axis=1)
            mask = np.concatenate((global_mask, local_mask), axis=1)
        self.register_buffer('bank', torch.tensor(bank, dtype=torch.float16), persistent=False)
        self.register_buffer('mask', torch.tensor(mask, dtype=torch.bool), persistent=False)
        input_dim = global_features.shape[-1]
        self.query = nn.Sequential(nn.Linear(2 * input_dim, 64), nn.LayerNorm(64))
        self.evidence = nn.Linear(input_dim, 64, bias=False)
        self.attention = nn.MultiheadAttention(64, 4, dropout=0.0, batch_first=True)
        self.scorer = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))
        nn.init.zeros_(self.scorer[-1].weight)
        nn.init.zeros_(self.scorer[-1].bias)

    def forward(self, profiles, item_ids):
        shape = item_ids.shape
        ids = item_ids.reshape(-1)
        if item_ids.ndim == 2:
            profiles = profiles[:, None, :].expand(-1, shape[1], -1).reshape(len(ids), -1)
        q = self.query(profiles).unsqueeze(1)
        evidence = self.evidence(self.bank[ids].float())
        mask = self.mask[ids].clone()
        # MultiheadAttention requires at least one unmasked position.
        mask[~mask.any(dim=1), 0] = True
        context, _ = self.attention(q, evidence, evidence, key_padding_mask=~mask, need_weights=False)
        q, context = q.squeeze(1), context.squeeze(1)
        combined = torch.cat((q * context, (q - context).abs()), dim=-1)
        return self.scorer(combined).squeeze(-1).tanh().reshape(shape)
