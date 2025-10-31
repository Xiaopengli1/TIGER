import math
from typing import List

import torch
import torch.nn as nn

from .vq import VectorQuantizer


class ResidualVectorQuantizer(nn.Module):
    """Residual vector quantizer with usage statistics and adaptive pruning."""

    def __init__(
        self,
        n_e_list,
        e_dim,
        sk_epsilons,
        beta: float = 0.25,
        kmeans_init: bool = False,
        kmeans_iters: int = 100,
        sk_iters: int = 100,
        usage_ema_decay: float = 0.99,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.n_e_list = list(n_e_list)
        self.e_dim = e_dim
        self.num_quantizers = len(self.n_e_list)
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = list(sk_epsilons)
        self.sk_iters = sk_iters

        self.vq_layers = nn.ModuleList(
            [
                VectorQuantizer(
                    n_e,
                    e_dim,
                    beta=self.beta,
                    kmeans_init=self.kmeans_init,
                    kmeans_iters=self.kmeans_iters,
                    sk_epsilon=sk_epsilon,
                    sk_iters=sk_iters,
                )
                for n_e, sk_epsilon in zip(self.n_e_list, self.sk_epsilons)
            ]
        )

        self.usage_ema_decay = usage_ema_decay
        self.eps = eps

        # Usage statistics (kept on CPU to simplify resizing during pruning).
        self._hist_ema: List[torch.Tensor] = [torch.zeros(n_e) for n_e in self.n_e_list]
        self.register_buffer("usage_ema", torch.zeros(self.num_quantizers))
        self.register_buffer("entropy_ema", torch.zeros(self.num_quantizers))

    def get_codebook(self):
        all_codebook = []
        for quantizer in self.vq_layers:
            codebook = quantizer.get_codebook()
            all_codebook.append(codebook)
        return torch.stack(all_codebook)

    def forward(self, x, use_sk: bool = True):
        all_losses = []
        all_indices = []

        x_q = 0
        residual = x
        for layer_idx, quantizer in enumerate(self.vq_layers):
            x_res, loss, indices = quantizer(residual, use_sk=use_sk)
            residual = residual - x_res
            x_q = x_q + x_res

            all_losses.append(loss)
            all_indices.append(indices)

            if self.training:
                flat_indices = indices.view(-1).detach().cpu()
                hist = torch.bincount(flat_indices, minlength=self.n_e_list[layer_idx]).float()
                total = hist.sum().clamp_min(1.0)
                probs = (hist / total).clamp_min(self.eps)
                entropy = -torch.sum(probs * probs.log())
                norm_entropy = entropy / math.log(self.n_e_list[layer_idx])
                usage_ratio = (hist > 0).float().mean()

                self._hist_ema[layer_idx] = (
                    self._hist_ema[layer_idx] * self.usage_ema_decay
                    + hist * (1 - self.usage_ema_decay)
                )
                self.usage_ema[layer_idx] = (
                    self.usage_ema[layer_idx] * self.usage_ema_decay
                    + usage_ratio * (1 - self.usage_ema_decay)
                )
                self.entropy_ema[layer_idx] = (
                    self.entropy_ema[layer_idx] * self.usage_ema_decay
                    + norm_entropy * (1 - self.usage_ema_decay)
                )

        mean_losses = torch.stack(all_losses).mean()
        all_indices = torch.stack(all_indices, dim=-1)

        return x_q, mean_losses, all_indices

    def get_usage_statistics(self) -> List[dict]:
        """Return a list of usage statistics for each quantizer layer."""

        stats: List[dict] = []
        for idx, n_e in enumerate(self.n_e_list):
            hist = self._hist_ema[idx]
            total = float(hist.sum().item())
            if total > 0:
                probs = (hist / total).clamp_min(self.eps)
                entropy = -float(torch.sum(probs * probs.log()).item())
                norm_entropy = entropy / math.log(n_e)
                usage_ratio = float((hist > 0).float().mean().item())
            else:
                norm_entropy = float(self.entropy_ema[idx].item())
                usage_ratio = float(self.usage_ema[idx].item())

            stats.append(
                {
                    "layer": idx,
                    "num_codes": n_e,
                    "usage_ratio": usage_ratio,
                    "normalized_entropy": norm_entropy,
                }
            )

        return stats

    def prune_quantizers(
        self,
        usage_threshold: float,
        entropy_threshold: float,
        min_quantizers: int = 1,
    ) -> List[int]:
        """Prune quantizer layers based on usage ratio and normalized entropy."""

        if self.num_quantizers <= min_quantizers:
            return []

        stats = self.get_usage_statistics()
        keep_indices: List[int] = []
        pruned_indices: List[int] = []

        for idx, stat in enumerate(stats):
            should_prune = (
                stat["usage_ratio"] < usage_threshold
                and stat["normalized_entropy"] < entropy_threshold
                and (self.num_quantizers - len(pruned_indices)) > min_quantizers
            )

            if should_prune:
                pruned_indices.append(idx)
            else:
                keep_indices.append(idx)

        if not pruned_indices:
            return []

        self.vq_layers = nn.ModuleList([self.vq_layers[i] for i in keep_indices])
        self.n_e_list = [self.n_e_list[i] for i in keep_indices]
        self.sk_epsilons = [self.sk_epsilons[i] for i in keep_indices]
        self.num_quantizers = len(self.n_e_list)
        self._hist_ema = [self._hist_ema[i].clone() for i in keep_indices]
        self.usage_ema = self.usage_ema[keep_indices].clone()
        self.entropy_ema = self.entropy_ema[keep_indices].clone()

        return pruned_indices
