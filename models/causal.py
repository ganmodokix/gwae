import torch
from torch import nn
from torch.nn import functional as F

def softsort(input: torch.Tensor, tau: float = 1.0, hard: bool = False) -> torch.Tensor:
    """ SoftSort [Prillo & Eisenschlos, 2020] for 1d-tensor """

    num_items = input.shape[0]

    a = torch.abs(input[:,None] - input[None,:])
    b = a.sum(dim=1, keepdim=True).tile(1, num_items)
    s = (num_items - 1 - 2 * (torch.arange(num_items, device=input.device))).float()
    c = input[:,None] * s[None,:]
    pmax = (c - b).T
    phat = F.softmax(pmax / tau, dim=-1)

    if hard:

        phard = torch.zeros_like(input)[:,None].tile(1, num_items)
        phard[torch.argmax(phat, dim=1), torch.arange(num_items)] = 1

        phat = (phard - phat).detach() + phat

    return phat


def gumbel_top_k_sampling(logit: torch.Tensor, tau: float = 1.0, hard: bool = False) -> torch.Tensor:
    """
    input: 1-dimensional
    """

    u = torch.rand_like(logit)
    g = -(-u.log()).log()

    s = logit + g
    return softsort(s, tau, hard=hard)

class DifferentiableDAG(nn.Module):

    def __init__(self, num_features: int) -> None:
        super().__init__()

        self.num_features = int(num_features)
        assert self.num_features > 0

        self.permutation_score = nn.Parameter(torch.randn(size=[self.num_features, ]), requires_grad=True)
        self.edge_score = nn.Parameter(torch.randn(size=[self.num_features, self.num_features]), requires_grad=True)
        self.tau = 1

    def sample_permutation(self):

        return gumbel_top_k_sampling(self.permutation_score, tau=self.tau, hard=not self.training)

    def sample_edges(self):

        p = torch.stack([self.edge_score, -self.edge_score], dim=2)
        g = p + -(-torch.rand_like(p).log()).log()
        v = F.softmax(g / self.tau, dim=2)
        r = v[:,:,1]
        if not self.training:
            h = torch.where(v[:,:,1] > v[:,:,0], 1.0, 0.0)
            r = (h - r).detach() + r
        return r

    def forward(self):

        mask = torch.triu(torch.ones([self.num_features, self.num_features], device=self.permutation_score.device), 1)
        pi = self.sample_permutation()
        permuted_mask = pi @ mask @ pi.T

        dag_adj = self.sample_edges() * permuted_mask
        return dag_adj

class CausalLinear(nn.Module):

    def __init__(self, num_features: int) -> None:
        super().__init__()

        self.num_features = int(num_features)
        assert self.num_features > 0

        r = self.num_features ** -0.5
        self.weight = nn.Parameter(torch.rand([self.num_features, self.num_features]) * r * 2 - r, requires_grad=True)
        self.bias = nn.Parameter(torch.rand([self.num_features, ]) * r * 2 - r, requires_grad=True)

    def forward(self, input: torch.Tensor, adjacent_matrix: torch.Tensor) -> None:
        
        return input + input @ (self.weight * adjacent_matrix.T) + self.bias