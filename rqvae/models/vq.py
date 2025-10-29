import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import kmeans, sinkhorn_algorithm


class VectorQuantizer(nn.Module):

    def __init__(self, n_e, e_dim,
                 beta = 0.25, kmeans_init = False, kmeans_iters = 10,
                 sk_epsilon=0.003, sk_iters=100,
                 use_post_linear=False, post_linear_bias=True):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilon = sk_epsilon
        self.sk_iters = sk_iters
        self.use_post_linear = use_post_linear

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        if not kmeans_init:
            self.initted = True
            self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        else:
            self.initted = False
            self.embedding.weight.data.zero_()

        if self.use_post_linear:
            self.post_quant_linear = nn.Linear(self.e_dim, self.e_dim, bias=post_linear_bias)
            self._init_post_linear()
        else:
            self.post_quant_linear = None

    def get_codebook(self):
        return self.apply_post_linear(self.embedding.weight)

    def _init_post_linear(self):
        if not self.use_post_linear:
            return

        with torch.no_grad():
            weight = self.post_quant_linear.weight
            if weight.shape[0] == weight.shape[1]:
                nn.init.eye_(weight)
            else:
                nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
            if self.post_quant_linear.bias is not None:
                self.post_quant_linear.bias.zero_()

    def apply_post_linear(self, tensor):
        if self.post_quant_linear is None:
            return tensor
        return self.post_quant_linear(tensor)

    def get_codebook_entry(self, indices, shape=None):
        # get quantized latent vectors
        z_q = self.embedding(indices)
        z_q = self.apply_post_linear(z_q)
        if shape is not None:
            z_q = z_q.view(shape)

        return z_q

    def init_emb(self, data):

        centers = kmeans(
            data,
            self.n_e,
            self.kmeans_iters,
        )

        self.embedding.weight.data.copy_(centers)
        self.initted = True

    @staticmethod
    def center_distance_for_constraint(distances):
        # distances: B, K
        max_distance = distances.max()
        min_distance = distances.min()

        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle + 1e-5
        assert amplitude > 0
        centered_distances = (distances - middle) / amplitude
        return centered_distances

    def forward(self, x, use_sk=True):
        # Flatten input
        latent = x.view(-1, self.e_dim)

        if not self.initted and self.training:
            self.init_emb(latent)

        # Calculate the L2 Norm between latent and Embedded weights
        embedding_for_distance = self.embedding.weight
        if self.post_quant_linear is not None:
            embedding_for_distance = self.apply_post_linear(embedding_for_distance)

        d = torch.sum(latent**2, dim=1, keepdim=True) + \
            torch.sum(embedding_for_distance**2, dim=1, keepdim=True).t()- \
            2 * torch.matmul(latent, embedding_for_distance.t())
        if not use_sk or self.sk_epsilon <= 0:
            indices = torch.argmin(d, dim=-1)
        else:
            d = self.center_distance_for_constraint(d)
            d = d.double()
            Q = sinkhorn_algorithm(d, self.sk_epsilon, self.sk_iters)

            if torch.isnan(Q).any() or torch.isinf(Q).any():
                print(f"Sinkhorn Algorithm returns nan/inf values.")
            indices = torch.argmax(Q, dim=-1)

        # indices = torch.argmin(d, dim=-1)

        x_q = self.embedding(indices).view(x.shape)
        x_hat = self.apply_post_linear(x_q)

        # compute loss for embedding
        commitment_loss = F.mse_loss(x_hat.detach(), x)
        codebook_loss = F.mse_loss(x_hat, x.detach())
        loss = codebook_loss + self.beta * commitment_loss

        # preserve gradients
        x_q = x + (x_hat - x).detach()

        indices = indices.view(x.shape[:-1])

        return x_q, loss, indices


