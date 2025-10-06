# %%
from sklearn.model_selection import ParameterGrid
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import numpy as np
from torch.optim import lr_scheduler
from transformers import PreTrainedModel, PreTrainedTokenizer
import torch
import cvxpy as cp
import numpy as np


def simplex_projection(x: torch.Tensor) -> torch.Tensor:
    """
    Project x onto the simplex.

    Args:
        x: (L, V) tensor of $L$ vectors in $\mathbb{R}^V$ to project

    Returns:
        projected: (L, V) tensor
    """
    # To project $x\in \mathbb{R}^{V}$ onto the simplex $\Delta^{V}$ we solve the following QP:
    # $\min_{x'}{||x-x'||_2^2}$ st $\mathbb{1}^\top x'=1, y\geq 0$
    # Solving the KKT yields the following algorithm

    # 1. sort in descending order
    L, V = x.shape
    x_sorted, _ = torch.sort(x, dim=1, descending=True)

    # 2. calc prefix sums of sorted elements
    cssv = torch.cumsum(x_sorted, dim=1) - 1

    # 3. calc threshold for each row
    rho_idx = torch.arange(1, V + 1, device=x.device).view(1, -1)
    t = x_sorted - cssv / rho_idx
    mask = t > 0
    rho = mask.sum(dim=1)

    # 4. grab theta for each row and reshape to (L,1) for broadcasting
    theta = cssv[torch.arange(L), rho - 1].view(L, 1) / rho.view(L, 1)

    return torch.clamp(x - theta, min=0)


def gini_index(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the Gini index for each row of x.

    Args:
        x: (L, V) tensor where each row is a probability vector.

    Returns:
        gini: (L,) tensor of Gini indices
    """
    # Gini index: G = 1 - sum_i x_i^2 (for normalized probabilities)
    return 1 - (x ** 2).sum(dim=1)


def entropy_projection(x: torch.Tensor, max_gini: float = 1) -> torch.Tensor:
    """
    Project x onto the simplex with an upper bound on the gini index.

    Args:
        x: (L, V) tensor of $L$ vectors in $\mathbb{R}^V$ to project
        max_gini: float, maximum gini index (set to 1 for no projection)

    Returns:
        projected: (L, V) tensor
    """
    # If we just project $x$ onto the simplex, we may end up with homogenous solutions
    # We would like to encourage sparsity by dissallowing high entropy $x$.
    # To do so, we constrain the gini index $S_{q=2}(x) < 1-r^2$, or equivilantly $||x||_2^2 > r^2$.
    # So $x$ must lie on the simplex but outside the $V$-sphere of radius $r$ about the origin.
    # Geometrically, you can imagine this sphere cutting a "hole" out of each facet of our simplex.
    # Or you can imagine adding a vector in the entropy-decreasing direction to the gradient.

    # To perform this projection we know $c=\frac{1}{V}\mathbb{1}$ must be the center of this "hole",
    # so the radius is just how far along the simplex we have to walk in any
    # direction $d$ to reach $||x||_2^2 = r^2$. Let $d = [1-V, \mathbb{1}_{V-1}^\top]^\top$, then solve $||c + \lambda d||_2^2 = r^2$ to find $\lambda = r^2 - \frac{1}{V}$
    # So we just need to project $x$ onto the $V$ sphere with center $c$ and radius $\lambda$ which is quite straightforward
    x = simplex_projection(x)
    max_gini = max(0, min(max_gini, 1))
    r = (1 - max_gini) ** 0.5

    # 1. compute center
    n_nonzero = (x > 0).sum(axis=1).unsqueeze(1)
    c = (x > 0) / n_nonzero
    lmbda = (r ** 2 - 1 / n_nonzero) ** 0.5

    # 2. if already outside the forbidden ball, we're done so mask out those tokens
    l2 = (x ** 2).sum(dim=1, keepdim=True)  # $||x||_2^2$
    mask = (l2 < r ** 2).squeeze(1)

    # 3. otherwise, project onto $||x||_2^2 = r^2$
    d = ((x-c) ** 2).sum(dim=1, keepdim=True) ** 0.5  # $||x - c||_2$
    d = torch.clamp(d, min=1e-8)
    x_proj = (lmbda / d) * (x - c) + c
    x_proj = simplex_projection(x_proj)

    x[mask] = x_proj[mask]
    return x


# Simplex projection tests
x = torch.Tensor([[0.2, 0.2]])
proj = simplex_projection(x)
target = torch.Tensor([[0.5, 0.5]])
assert torch.allclose(proj, target, atol=1e-4)

x = torch.Tensor([[0.7, 0.3]])  # already normalized
proj = simplex_projection(x)
assert torch.allclose(proj, x, atol=1e-4)

x = torch.Tensor([[-0.5, 1.5]])  # negative values
proj = simplex_projection(x)
target = torch.Tensor([[0, 1]])
assert torch.allclose(proj, target, atol=1e-4)

x = torch.Tensor([[0.2, 0.2, 0.7]])  # 3D vector
proj = simplex_projection(x)
target = torch.Tensor([[0.1667, 0.1667, 0.6667]])
assert torch.allclose(proj, target, atol=1e-4)

x = torch.Tensor([[0, 0, 0]])  # all zeros
proj = simplex_projection(x)
target = torch.Tensor([[1/3, 1/3, 1/3]])
assert torch.allclose(proj, target, atol=1e-5)

# Entropy projection tests
x = torch.Tensor([[0.6, 0.4]])
proj = entropy_projection(x.clone(), max_gini=0)  # force one-hot
target = torch.Tensor([[1, 0]])
assert torch.allclose(proj, target, atol=1e-4)

x = torch.Tensor([[0.6, 0.4]])
proj = entropy_projection(x.clone(), max_gini=0.5)  # gini already lt max
assert torch.allclose(proj, x, atol=1e-4)

x = torch.Tensor([[0.6, 0.4]])
proj = entropy_projection(x.clone(), max_gini=1)  # unchanged
assert torch.allclose(proj, x, atol=1e-4)

x = torch.Tensor([[0.3, 0.3, 0.4]])  # force one-hot
proj = entropy_projection(x.clone(), max_gini=0)
target = torch.Tensor([[0, 0, 1]])
assert torch.allclose(proj, target, atol=1e-4)

x = torch.Tensor([[0.1, 0.2, 0.7]])  # max_gini=1 leaves unchanged
proj = entropy_projection(x.clone(), max_gini=1)
assert torch.allclose(proj, x, atol=1e-4)

x = torch.Tensor([[0.05, 0.05, 0.1, 0.1, 0.7, 0, 0]])  # 7D vector
proj = entropy_projection(x.clone(), max_gini=0.5)
assert torch.isclose(proj.sum(), torch.tensor(1.0), atol=1e-5)

x = torch.Tensor([[0.25, 0.25, 0.25, 0.25]])  # equal values, low max_gini
proj = entropy_projection(x.clone(), max_gini=0.2)
assert torch.isclose(proj.sum(), torch.tensor(1.0), atol=1e-5)
