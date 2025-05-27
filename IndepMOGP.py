import copy
import torch
from torch import Tensor
from torch import nn

from gpytorch.kernels import Kernel
from gpytorch.utils.transforms import inv_softplus
from linear_operator.utils.cholesky import psd_safe_cholesky

class GP(nn.Module):
    """
    Single Output GP
    """
    def __init__(self, kernel: Kernel, noise_var: float, train_X: Tensor, train_Y: Tensor,
                 fix_noise_var: bool = False, jitter=1e-6):
        # train_X: [n_train, D_x]; train_Y: [n_train, 1]
        # we assume zero mean!
        super(GP, self).__init__()
        assert train_Y.ndim == 2
        assert train_Y.size(-1) == 1
        assert train_Y.size(0) == train_X.size(0)
        self.kernel = kernel
        self.train_X = torch.as_tensor(train_X, dtype=torch.get_default_dtype())
        self.train_Y = torch.as_tensor(train_Y, dtype=torch.get_default_dtype())
        self.fix_noise_var = fix_noise_var
        self.jitter = jitter

        if self.fix_noise_var:
            self.register_buffer("_noise_var", torch.tensor([noise_var]))
        else:
            self.register_parameter("raw_noise_var", torch.nn.Parameter(inv_softplus(torch.tensor(noise_var)),
                                                                        requires_grad=True))

    @property
    def noise_var(self):
        if self.fix_noise_var:
            return self._noise_var
        else:
            return nn.functional.softplus(self.raw_noise_var)

    @torch.no_grad()
    def posterior(self, test_X: Tensor, diag=True):
        # noiseless f
        # test_X: [n_test, D_x]
        Kxx = self.kernel(self.train_X).to_dense()  
        noisy_Kxx = Kxx + self.noise_var * torch.eye(Kxx.size(-1))
        Kxxs = self.kernel(self.train_X, test_X).to_dense()  
        noisy_Lxx = psd_safe_cholesky(noisy_Kxx + self.jitter * torch.eye(Kxx.shape[-1]))
        noisy_Kxx_inv = torch.cholesky_solve(torch.eye(Kxx.shape[-1], dtype=Kxx.dtype), noisy_Lxx)

        pred_mean = (Kxxs.mT @ noisy_Kxx_inv @ self.train_Y).squeeze(-1)  

        if diag:
            Kxsxs = self.kernel(test_X, diag=True).to_dense()  
            pred_cov = torch.einsum('ij,jk,ki->i', Kxxs.mT, noisy_Kxx_inv, Kxxs)  
            assert torch.allclose(pred_cov, torch.diagonal(Kxxs.mT @ noisy_Kxx_inv @ Kxxs, dim1=-2, dim2=-1))
            pred_cov = Kxsxs - pred_cov
        else:
            raise NotImplementedError

        return pred_mean, pred_cov  

    def log_evidence(self):
        Kxx = self.kernel(self.train_X).to_dense()  
        noisy_Kxx = Kxx + self.noise_var * torch.eye(Kxx.size(-1))
        noisy_Lxx = psd_safe_cholesky(noisy_Kxx + self.jitter * torch.eye(noisy_Kxx.size(-1)))
        dist = torch.distributions.MultivariateNormal(loc=torch.zeros(noisy_Kxx.size(-1)), scale_tril=noisy_Lxx)
        return dist.log_prob(self.train_Y.squeeze(-1))

    def train_gp(self, optimizer: torch.optim.Optimizer, epochs: int):
        for epoch in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            loss = - self.log_evidence()
            loss.backward()
            optimizer.step()

class IndepMOGP(nn.Module):
    """
    Independent GP for every output
    """
    def __init__(self, kernel: Kernel, train_X: Tensor, train_Y: Tensor, train_mask: Tensor,
                 noise_var=1.0, fix_noise_var=False, jitter=1e-6):
        super(IndepMOGP, self).__init__()
        self.train_X = train_X
        self.train_Y = train_Y
        self.train_mask = train_mask
        self.GP_list = nn.ModuleList()
        self.setup_indepGPs(kernel=kernel,
                            noise_var=noise_var,
                            fix_noise_var=fix_noise_var,
                            jitter=jitter)

    def setup_indepGPs(self, kernel: Kernel, noise_var: float, fix_noise_var: bool, jitter=1e-6):
        P = self.train_Y.size(-1)
        for p in range(P):
            curr_mask = self.train_mask[:, p]  
            curr_train_X = self.train_X[curr_mask]  
            if curr_train_X.size(0) < 3:
                print(f"for output {p}, only {curr_train_X.size(0)} training examples!")
                raise NotImplementedError
            curr_train_Y = self.train_Y[:, p][curr_mask].unsqueeze(-1)  
            curr_gp = GP(kernel=copy.deepcopy(kernel), noise_var=noise_var, train_X=curr_train_X, train_Y=curr_train_Y,
                         fix_noise_var=fix_noise_var, jitter=jitter)
            self.GP_list.append(curr_gp)

    def train_indepmogp(self, optimizer: torch.optim.Optimizer, epochs: int):
        # approach 1
        for gp in self.GP_list:
            gp.train_gp(optimizer, epochs)

        # approach 2
        """
        for epoch in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            loss = - self.GP_list[0].log_evidence()
            for i, gp in enumerate(self.GP_list):
                if i == 0:
                    pass
                loss = loss - gp.log_evidence()
            loss.backward()
            optimizer.step()
        """

    @torch.no_grad()
    def predict(self, test_X: Tensor, output_ids: list=None, diag=True):
        assert diag == True
        pred_mean, pred_cov = [], []

        if output_ids is None:
            output_ids = [i for i in range(len(self.GP_list))]

        for id in output_ids:
            curr_gp = self.GP_list[id]
            curr_pred_mean, curr_pred_cov = curr_gp.posterior(test_X, diag=diag)
            pred_mean.append(curr_pred_mean)
            pred_cov.append(curr_pred_cov)

        pred_mean = torch.stack(pred_mean, dim=-1)  
        pred_cov = torch.stack(pred_cov, dim=-1)

        return pred_mean, pred_cov


    def predict_indepmogp(self):
        raise NotImplementedError


if __name__ == "__main__":
    from gpytorch.kernels import ScaleKernel, RBFKernel
    import matplotlib.pyplot as plt

    torch.set_default_dtype(torch.float64)

    # Synthetic dataset
    N_train = 64
    X_start, X_end = -3, 3
    train_X = torch.linspace(X_start, X_end, N_train).reshape(-1, 1)  
    train_Y_noiseless = torch.cat([torch.sin(train_X), torch.cos(train_X)], dim=-1)  
    train_Y = train_Y_noiseless + 0.05 * torch.randn_like(train_Y_noiseless)

    with torch.random.fork_rng():
        torch.manual_seed(123)
        train_m = torch.randint(0, 2, (N_train, 2), dtype=torch.bool)  

    kernel = ScaleKernel(RBFKernel())
    model = IndepMOGP(kernel, train_X, train_Y, train_m, noise_var=0.1, fix_noise_var=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train_indepmogp(optimizer, epochs=1000)

    n_test = 500
    test_X = torch.linspace(X_start - 1, X_end + 1, n_test).reshape(-1, 1)

    pred_mean, pred_cov = model.predict(test_X)  

    plt.figure(figsize=(16, 9))

    plt.subplot(1, 2, 1)  # (rows, columns, index)
    plt.plot(test_X.squeeze(-1), torch.sin(test_X.squeeze(-1)), color='black', label='sin(x)')
    plt.plot(test_X.squeeze(-1), pred_mean[:, 0], color='blue', label='pred mean')
    plt.scatter(train_X.squeeze(-1)[train_m[:, 0]], train_Y[:, 0][train_m[:, 0]], marker='x', color='red',
                label='train data', s=10)
    plt.fill_between(
        test_X.squeeze(-1),
        pred_mean[:, 0] - 1.96 * pred_cov[:, 0].sqrt(),
        pred_mean[:, 0] + 1.96 * pred_cov[:, 0].sqrt(),
        alpha=0.2,
        color='blue',
    )
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(test_X.squeeze(-1), torch.cos(test_X.squeeze(-1)), color='black', label='cos(x)')
    plt.plot(test_X.squeeze(-1), pred_mean[:, 1], color='blue', label='pred mean')
    plt.scatter(train_X.squeeze(-1)[train_m[:, 1]], train_Y[:, 1][train_m[:, 1]], marker='x', color='red',
                label='train data', s=10)
    plt.fill_between(
        test_X.squeeze(-1),
        pred_mean[:, 1] - 1.96 * pred_cov[:, 1].sqrt(),
        pred_mean[:, 1] + 1.96 * pred_cov[:, 1].sqrt(),
        alpha=0.2,
        color='blue',
    )
    plt.legend()

    plt.tight_layout()  # Adjust layout to avoid overlap
    plt.savefig('./synthetic_IndepMOGP.pdf')

