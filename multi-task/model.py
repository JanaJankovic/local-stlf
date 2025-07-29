import torch
import torch.nn as nn


def periodic_distance(x1, x2, period):
    diff = torch.abs(x1[:, None] - x2[None, :])
    return torch.minimum(diff, period - diff)


def composite_kernel(X1, X2, sigma_t=4.0, sigma_d=120.0):
    t1, d1, c1 = X1[:, 0], X1[:, 1], X1[:, 2]
    t2, d2, c2 = X2[:, 0], X2[:, 1], X2[:, 2]

    h_t = periodic_distance(t1, t2, 24)
    K_t = torch.exp(-h_t / sigma_t)

    h_d = periodic_distance(d1, d2, 365)
    K_d = torch.exp(-h_d / sigma_d)

    K_c = (c1[:, None] == c2[None, :]).float()
    return K_t * K_d * K_c


class SharedBasisFunctions(nn.Module):
    def __init__(self, X_train, p=10, sigma_t=4.0, sigma_d=120.0):
        super().__init__()
        self.register_buffer("X_train", X_train)
        self.sigma_t = sigma_t
        self.sigma_d = sigma_d
        self.register_buffer("A", torch.empty(0))  # Will be updated later

    def build_kernel(self, X1, X2, block_size=5000):
        """
        Computes the composite kernel matrix K(X1, X2) in blocks to avoid memory overflow.
        Returns a full matrix of shape (len(X1), len(X2)).
        """
        N1, N2 = X1.size(0), X2.size(0)
        K = torch.zeros(N1, N2, dtype=torch.float32)

        # Always compute on CPU to avoid GPU OOM
        X1_cpu = X1.cpu()
        X2_cpu = X2.cpu()

        for i in range(0, N1, block_size):
            i_end = min(i + block_size, N1)
            for j in range(0, N2, block_size):
                j_end = min(j + block_size, N2)

                # Compute partial kernel block and write into full matrix
                K_block = composite_kernel(
                    X1_cpu[i:i_end], X2_cpu[j:j_end], self.sigma_t, self.sigma_d
                )
                K[i:i_end, j:j_end] = K_block

                print(f"Computed K[{i}:{i_end}, {j}:{j_end}]")

        return K.to(X1.device)

    def forward(self, X):
        K = self.build_kernel(X, self.X_train)
        return K @ self.A  # [B, p]


class MultiTaskOKL(nn.Module):
    def __init__(
        self, X_train, num_tasks, p=10, horizon=24, sigma_t=4.0, sigma_d=120.0
    ):
        super().__init__()
        self.horizon = horizon
        self.num_tasks = num_tasks
        self.p = p

        self.shared_basis = SharedBasisFunctions(X_train, p, sigma_t, sigma_d)
        self.B = torch.randn(p, num_tasks, horizon) * 0.01  # good default
        self.L = torch.eye(num_tasks)  # [T, T]

    def compute_shared_basis(self, X):
        return self.shared_basis(X)  # [B, p]

    def predict_with_basis(self, G, task_ids):
        preds = torch.zeros(G.size(0), self.horizon, device=G.device)
        for i, task_id in enumerate(task_ids):
            b = self.B[:, task_id]  # [p, H]
            preds[i] = G[i] @ b  # [H]
        return preds
