import torch
import torch.nn as nn

def periodic_distance(x1, x2, period):
    """
    Implements h_P(x) = min(|x1 - x2|, P - |x1 - x2|) from Eq. (8) and (9)
    This function ensures periodicity over a given interval.
    """
    diff = torch.abs(x1[:, None] - x2[None, :])
    return torch.minimum(diff, period - diff)

def composite_kernel(X1, X2, sigma_t=4.0, sigma_d=120.0):
    """
    Composite kernel function K(x1, x2) = K_t * K_d * K_c — Eq. (16)

    K_t: Time-of-day kernel using Eq. (8)
    K_d: Day-of-year kernel using Eq. (9)
    K_c: Day-type kernel using Eq. (10)
    """
    t1, d1, c1 = X1[:, 0], X1[:, 1], X1[:, 2]
    t2, d2, c2 = X2[:, 0], X2[:, 1], X2[:, 2]

    # Eq. (8): Time-of-day periodic kernel
    h_t = periodic_distance(t1, t2, period=24)
    K_t = torch.exp(-h_t / sigma_t)

    # Eq. (9): Day-of-year periodic kernel
    h_d = periodic_distance(d1, d2, period=365)
    K_d = torch.exp(-h_d / sigma_d)

    # Eq. (10): Day-type categorical kernel
    K_c = (c1[:, None] == c2[None, :]).float()

    # Eq. (16): Multiplicative composite kernel
    return K_t * K_d * K_c


class SharedBasisFunctions(nn.Module):
    """
    Implements shared latent functions g_k(x) = sum_i a_ik K(x_i, x)
    Referenced in Eq. (7)
    """
    def __init__(self, X_train, p=10, sigma_t=4.0, sigma_d=120.0):
        super().__init__()
        self.X_train = X_train.detach()
        self.sigma_t = sigma_t
        self.sigma_d = sigma_d

        # a_ik coefficients from Eq. (7)
        self.a = nn.Parameter(torch.randn(X_train.shape[0], p))

    def forward(self, X):
        # K(x, x_i) between new inputs and training points — part of Eq. (7)
        K = composite_kernel(X, self.X_train, sigma_t=self.sigma_t, sigma_d=self.sigma_d)
        G = K @ self.a  # Each g_k(x) = sum_i a_ik K(x_i, x) — Eq. (7)
        return G


class MultiTaskOKL(nn.Module):
    """
    Implements f_j(x) = sum_k b_jk g_k(x), where g_k are shared basis — Eq. (7)
    The output kernel L is factorized as L = B B^T, where b_jk are learnable
    """
    def __init__(self, X_train, num_tasks, p=10, sigma_t=4.0, sigma_d=120.0):
        super().__init__()
        self.shared_basis = SharedBasisFunctions(X_train, p=p, sigma_t=sigma_t, sigma_d=sigma_d)

        # b_jk coefficients from Eq. (7): low-rank factor of L
        self.b = nn.Parameter(torch.randn(num_tasks, p))

    def forward(self, X):
        # g_k(x) for all k — Eq. (7)
        G = self.shared_basis(X)  # [batch, p]

        # f_j(x) = sum_k b_jk g_k(x) — Eq. (7)
        return G @ self.b.T  # [batch, num_tasks]
