import copy
from typing import Union, Tuple, Any, Dict
import torch
from torch import Tensor
from torch import nn
from torch.optim import Optimizer
from torch.distributions import MultivariateNormal
from torch.utils.data import Dataset, DataLoader

from gpytorch.likelihoods import Likelihood
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel
from linear_operator.utils.cholesky import psd_safe_cholesky

class Prior_H(nn.Module):
    """p(H), fully factorized prior"""
    def __init__(self, mean_pH: Tensor, cov_pH: Union[Tensor]):
        super(Prior_H, self).__init__()
        self.register_buffer("mean_pH", mean_pH)  
        assert torch.all(cov_pH > 0.), "Prior H cov must be positive."
        self.register_buffer("cov_pH", cov_pH)    


class Variational_H(nn.Module):
    """q(H)"""
    def __init__(self, Q: int, P: int, D_h: int, mean_field=True):
        super(Variational_H, self).__init__()
        self.mean_field = mean_field
        if mean_field:
            self.register_parameter("mean_qH", nn.Parameter(torch.zeros((Q, P, D_h)), requires_grad=True))
            self.register_parameter("raw_cov_qH", nn.Parameter(torch.ones(Q, P, D_h), requires_grad=True))
        else:
            raise NotImplementedError

    @property
    def cov_qH(self):
        if self.mean_field:
            return nn.functional.softplus(self.raw_cov_qH)
        else:
            raise NotImplementedError

    def sample(self, ids):
        if self.mean_field:
            _mean = self.mean_qH[:, ids, :]  
            _std = torch.sqrt(self.cov_qH[:, ids, :])
            _eps = torch.randn_like(_mean)
            return _mean + _eps * _std  
        else:
            raise NotImplementedError

class Variational_inducing_dist(nn.Module):
    """q(U), over H and X"""
    def __init__(self, M_H: int, M_X: int):
        super(Variational_inducing_dist, self).__init__()
        self.register_parameter("mean_qU", nn.Parameter(torch.zeros((int(M_H * M_X))), requires_grad=True))
        self.register_parameter("chol_cov_qU_H",
                                nn.Parameter(torch.eye(int(M_H)),
                                             requires_grad=True))  
        self.register_parameter("chol_cov_qU_X",
                                nn.Parameter(torch.eye(int(M_X)),
                                             requires_grad=True))  

    @property
    def cov_qU_H(self):
        return self.chol_cov_qU_H @ self.chol_cov_qU_H.mT  

    @property
    def cov_qU_X(self):
        return self.chol_cov_qU_X @ self.chol_cov_qU_X.mT  

    @property
    def cov_qU(self):
        return torch.kron(self.cov_qU_H, self.cov_qU_X)


class Inducing_points(nn.Module):
    """Z_H or Z_X, inducing points/locations"""  
    def __init__(self, IP_name: str, Q: int, num_inducing_points: int, num_dim: int, IP_init: Tensor=None, IP_joint=True, init: str="random_normal"):
        super(Inducing_points, self).__init__()
        self.IP_name = IP_name

        if IP_init is not None:
            assert IP_init.shape == torch.Size([Q, num_inducing_points, num_dim])
            IPs = IP_init
        else:
            if init == "random_normal":
                IPs = torch.randn((Q, num_inducing_points, num_dim))  
            else:
                raise NotImplementedError

        if IP_joint:
            self.register_parameter(IP_name, nn.Parameter(IPs, requires_grad=True))
        else:
            self.register_buffer(IP_name, IPs)


class LVMOGP(nn.Module):
    """
    Latent Variable MOGP.
    Notations:
    Q: number of coregionalization matrices
    D_x: input dims
    D_h: latent variable dims
    M_X: num of inducing variables in input space
    M_H: num of inducing variables in latent variable space
    P: number of outputs
    """
    def __init__(self, input_kernels: list, latent_kernels: list, N_train: int, pH: Prior_H, qH: Variational_H,
                 qU: Variational_inducing_dist, zH: Inducing_points, zX: Inducing_points, likelihood: Likelihood,
                 whitening=True, jitter=1e-6,
    ):
        super(LVMOGP, self).__init__()

        self.N_train = N_train  # num of observations
        self.pH = pH
        self.qH = qH
        self.qU = qU
        self.zH = zH  # inducing points, latent space
        self.zX = zX  # inducing points, input space
        self.num_outputs = self.pH.mean_pH.shape[-2]

        self.input_kernels = self._check_list_of_kernels(input_kernels)
        self.latent_kernels = self._check_list_of_kernels(latent_kernels)
        self.Q = len(self.input_kernels)

        self.likelihood = likelihood  # this lik is used for all outputs
        self.whitening = whitening
        self.jitter = jitter


    @property
    def pU(self):
        if self.whitening:
            # Zero mean, identity cov matrix.
            raise NotImplementedError
        else:
            raise NotImplementedError

    def _eval_K_input(self, x: Tensor, y: Tensor, diag: bool = False):
        """x: [(Q),b1,D_x]; y: [(Q),b2,D_x]"""
        if diag:
            assert x.size(-2) == y.size(-2)
        x, y = self._expand_tensor(x), self._expand_tensor(y)
        list_K_input = [self.input_kernels[q](x[q], y[q], diag=diag).to_dense() for q in range(self.Q)]
        Q_K_input = torch.stack(list_K_input, dim=0)
        return Q_K_input  # [Q, b1=b2] or [Q, b1, b2]

    def _eval_K_latent(self, x: Tensor, y: Tensor, diag: bool = False):
        """x: [(Q),b1,D_h]; y:[(Q),b2,D_h]"""
        if diag:
            assert x.size(-2) == y.size(-2)
        x, y = self._expand_tensor(x), self._expand_tensor(y)
        list_K_latent = [self.latent_kernels[q](x[q], y[q], diag=diag).to_dense() for q in range(self.Q)]
        Q_K_latent = torch.stack(list_K_latent, dim=0)
        return Q_K_latent  # [Q, b1=b2] or [Q, b1, b2]

    def variational_f(self, x: Tensor, H: Tensor, diag: bool = False):
        r"""
        q(f) = \int p(f|U) q(U) dU
        x: [b1,D_x]; H: [(Q),b2,D_h]
        # TODO: handle missing data
        """
        assert self.whitening

        cache = {}

        # prepare
        Q_K_ff_input = self._eval_K_input(x, x, diag)  
        Q_K_ff_latent = self._eval_K_latent(H, H, diag)  
        if diag:
            Q_K_ff = torch.einsum('qi,qk->qik', Q_K_ff_latent, Q_K_ff_input)
            Q_K_ff = Q_K_ff.reshape(-1, Q_K_ff_latent.shape[-1] * Q_K_ff_input.shape[-1])  
            K_ff = Q_K_ff.sum(dim=0)  
        else:
            Q_K_ff = torch.einsum('qij,qkl->qikjl', Q_K_ff_latent, Q_K_ff_input)
            Q_K_ff = Q_K_ff.reshape(-1, Q_K_ff_latent.shape[-2] * Q_K_ff_input.shape[-2],
                                    Q_K_ff_latent.shape[-1] * Q_K_ff_input.shape[-1])  
            K_ff = Q_K_ff.sum(dim=0)  

        # test
        test_Q_K_ff_input = self._eval_K_input(x, x, diag=False)  
        test_Q_K_ff_latent = self._eval_K_latent(H, H, diag=False)  
        test_sum_kron_products = 0.
        for q in range(self.Q):
            test_sum_kron_products += torch.kron(test_Q_K_ff_latent[q], test_Q_K_ff_input[q])
        if diag:
            assert torch.allclose(K_ff, test_sum_kron_products.diagonal(dim1=-2, dim2=-1))
        else:
            assert torch.allclose(K_ff, test_sum_kron_products)

        Q_K_fu_input = self._eval_K_input(x, self.zX.zX)  # [Q,b1,M_X]
        Q_K_fu_latent = self._eval_K_latent(H, self.zH.zH)  # [Q,b2,M_H]
        Q_K_fu = torch.einsum('qij,qkl->qikjl', Q_K_fu_latent, Q_K_fu_input)
        Q_K_fu = Q_K_fu.reshape(-1, Q_K_fu_latent.shape[-2] * Q_K_fu_input.shape[-2],
                                    Q_K_fu_latent.shape[-1] * Q_K_fu_input.shape[-1])  
        K_fu = Q_K_fu.sum(dim=0)  

        # test
        test_sum_kron_products = 0.
        for q in range(self.Q):
            test_sum_kron_products += torch.kron(Q_K_fu_latent[q], Q_K_fu_input[q])
        assert torch.allclose(K_fu, test_sum_kron_products)

        Q_K_uu_input = self._eval_K_input(self.zX.zX, self.zX.zX)  # [Q, M_X, M_X]
        Q_K_uu_latent = self._eval_K_latent(self.zH.zH, self.zH.zH)  # [Q, M_H, M_H]

        ## cholesky factor of K_uu

        # case1: Q=1
        if self.Q == 1:
            L_uu_input = psd_safe_cholesky(Q_K_uu_input.squeeze(0) + self.jitter * torch.eye(Q_K_uu_input.size(-1), device=Q_K_uu_input.device))
            L_uu_latent = psd_safe_cholesky(Q_K_uu_latent.squeeze(0) + self.jitter * torch.eye(Q_K_uu_latent.size(-1), device=Q_K_uu_latent.device))
            L_uu = torch.kron(L_uu_latent, L_uu_input)  

        # case2: Q>1 
        else:
            Q_K_uu = torch.einsum('qij,qkl->qikjl', Q_K_uu_latent, Q_K_uu_input)
            Q_K_uu = Q_K_uu.reshape(-1, Q_K_uu_latent.shape[-2] * Q_K_uu_input.shape[-2],
                                        Q_K_uu_latent.shape[-1] * Q_K_uu_input.shape[-1])
            K_uu = Q_K_uu.sum(dim=0)  

            # test
            test_sum_kron_products = 0.
            for q in range(self.Q):
                test_sum_kron_products += torch.kron(Q_K_uu_latent[q], Q_K_uu_input[q])
            assert torch.allclose(K_uu, test_sum_kron_products)

            L_uu = psd_safe_cholesky(K_uu + self.jitter * torch.eye(K_uu.size(-1), device=K_uu.device))  

        L_uu_inv_K_uf = torch.linalg.solve_triangular(L_uu, K_fu.mT, upper=False)  

        qf_mean = L_uu_inv_K_uf.mT @ self.qU.mean_qU  
        if diag:
            tmp = torch.einsum('ji,jk,ki->i', L_uu_inv_K_uf, (self.qU.cov_qU - torch.eye(self.qU.cov_qU.size(-1))), L_uu_inv_K_uf)  

            # test
            tmp2 = L_uu_inv_K_uf.mT @ (self.qU.cov_qU - torch.eye(self.qU.cov_qU.size(-1))) @ L_uu_inv_K_uf
            assert torch.allclose(tmp, tmp2.diagonal(dim1=-2, dim2=-1), atol=1e-5)

            qf_cov = (K_ff + tmp + self.jitter)  
        else:
            qf_cov = K_ff + L_uu_inv_K_uf.mT @ (self.qU.cov_qU - torch.eye(self.qU.cov_qU.size(-1))) @ L_uu_inv_K_uf + self.jitter * torch.eye(K_ff.shape[-1])  

        cache['L_uu'] = L_uu  # save for next time usage

        return qf_mean, qf_cov, cache


    def KL_qU_pU(self):
        """
        KL divergence term between q(U) and p(U)
        """
        M_H, M_X = self.qU.cov_qU_H.size(-1), self.qU.cov_qU_X.size(-1)

        # cholesky of variational cov matrices
        chol_var_cov_H = psd_safe_cholesky(self.qU.cov_qU_H + self.jitter * torch.eye(M_H))  
        chol_var_cov_X = psd_safe_cholesky(self.qU.cov_qU_X + self.jitter * torch.eye(M_X))  

        half_trace_H = torch.diagonal(chol_var_cov_H, dim1=-1, dim2=-2).sum()
        half_trace_X = torch.diagonal(chol_var_cov_X, dim1=-1, dim2=-2).sum()
        half_log_det_H = torch.diagonal(chol_var_cov_H, dim1=-1, dim2=-2).log().sum()
        half_log_det_X = torch.diagonal(chol_var_cov_X, dim1=-1, dim2=-2).log().sum()

        m_T_m = (self.qU.mean_qU ** 2).sum()
        KL = 2 * half_trace_H * half_trace_X - 0.5 * (M_H * M_X - m_T_m) - M_H * half_log_det_X - M_X * half_log_det_H

        return KL

    def KL_qH_pH(self, output_idx: Tensor = None):
        """
        mini-batch approximation for KL between q(H) and p(H), both are fully factorized.
        """
        assert self.qH.mean_field
        mean_pH = self._expand_tensor(self.pH.mean_pH)  
        cov_pH = self._expand_tensor(self.pH.cov_pH)  
        mean_qH = self._expand_tensor(self.qH.mean_qH)  
        cov_qH = self._expand_tensor(self.qH.cov_qH)  

        if output_idx is not None:
            mean_pH, cov_pH = mean_pH[:, output_idx, :], cov_pH[:, output_idx, :]
            mean_qH, cov_qH = mean_qH[:, output_idx, :], cov_qH[:, output_idx, :]

        term1 = torch.log(cov_pH) - torch.log(cov_qH)  
        term2 = (cov_qH + (mean_qH - mean_pH).pow(2)) / cov_pH  
        KLs = 0.5 * (term1 + term2 - 1.)  

        return KLs.sum(dim=(-3, -1)).mean()

    def elbo(self, x: Tensor, y: Tensor, m: Tensor, output_idx: Tensor, beta_u=1., beta_h=1.):
        """
        mini-batch elbo
        x: [b,D_x] i.e. xs are shared across output TODO: how about inputs are NOT shared?
        y: [b,P], P is the size of the subset of all outputs
        m: [b,P], where 0 indicate missing
        output_idx: [P]
        """
        H = self.qH.sample(output_idx)  
        qf_mean, qf_cov, cache = self.variational_f(x, H, diag=True)  
        qf_dist = MultivariateNormal(qf_mean, torch.diag_embed(qf_cov))

        # term 1/3 - exp_log_lik
        # TODO: mask before compute
        y = y.mT.flatten()  
        m = m.mT.flatten()
        _exp_log_lik = self.likelihood.expected_log_prob(y, qf_dist)  
        _exp_log_lik = _exp_log_lik[m.bool()]
        if not _exp_log_lik.tolist():  
            print('Encounter empty log lik!')
            exp_log_lik = 0.
        else:
            exp_log_lik = _exp_log_lik.mean()

        # term 2/3 - KL(q(U)||p(U))
        KL_qU_pU = self.KL_qU_pU()  

        # term 3/3 - KL(q(H)||p(H))
        KL_qH_pH = self.KL_qH_pH(output_idx)  

        elbo = self.N_train * self.num_outputs * exp_log_lik - beta_u * KL_qU_pU - beta_h * self.num_outputs * KL_qH_pH

        return elbo

    @torch.no_grad()
    def predict(self, x_star: Tensor, output_idx: Tensor=None, num_samples: int=1, diag=True):
        """
        x_star: [b1,D_x]
        make predictions for output_idx on x_star. If output_idx is None, then make predictions for all outputs.
        """
        if output_idx is None:
            output_idx = torch.arange(self.num_outputs)

        pred_means, pred_covs = [], []
        b1, b2 = x_star.size(-2), len(output_idx)

        for i in range(num_samples):
            H_samples = self.qH.sample(output_idx)  
            qf_mean, qf_cov, _ = self.variational_f(x_star, H_samples, diag=diag)
            qf_mean = qf_mean.reshape(b2, b1)
            if diag:
                qf_cov = qf_cov.reshape(b2, b1)
            else:
                qf_cov = qf_cov.reshape(b2, b1, b2, b1)
            pred_means.append(qf_mean)
            pred_covs.append(qf_cov)

        pred_means = torch.stack(pred_means, dim=0)  
        pred_covs = torch.stack(pred_covs, dim=0)  

        return pred_means, pred_covs

    def train_lvmogp(self, train_dataloader: DataLoader, output_batch_size: int, optimizer: Optimizer, epochs: int,
                     beta_u=1., beta_h=1., print_epochs=10):
        
        output_index_dataloader = None  # cache

        for epoch in range(epochs):
            for batch_X, batch_all_Y, batch_all_m in train_dataloader:
                if output_index_dataloader is None:
                    output_index_dataset = IndexDataset(num_data=batch_all_Y.size(-1))
                    output_index_dataloader = DataLoader(output_index_dataset, batch_size=output_batch_size, shuffle=True)

                for output_idx in output_index_dataloader:
                    batch_Y = batch_all_Y[..., output_idx]
                    batch_m = batch_all_m[..., output_idx]
                    optimizer.zero_grad(set_to_none=True)
                    loss = - self.elbo(batch_X, batch_Y, batch_m, output_idx, beta_u, beta_h)
                    loss.backward()
                    optimizer.step()

            if (epoch + 1) % print_epochs == 0:
                print(f'Epoch {epoch + 1} / {epochs}； Loss: {loss.item():.6f}')

    @torch.no_grad()
    def predict_lvmogp(self):
        """"""
        raise NotImplementedError

    def _check_list_of_kernels(self, list_of_kernels):
        Q = self.qH.mean_qH.shape[0]
        if len(list_of_kernels) == Q:
            return nn.ModuleList(list_of_kernels)
        else:
            assert len(list_of_kernels) == 1
            kernel = list_of_kernels[0]
            list_of_kernels = nn.ModuleList([copy.deepcopy(kernel) for _ in range(Q)])
            # list_of_kernels = nn.ModuleList(kernel.__class__() for _ in range(Q))
            return list_of_kernels

    def _expand_tensor(self, x: Tensor):
        if x.ndim == 2:
            x = x.unsqueeze(0).expand(self.Q, x.size(-2), x.size(-1))
        else:
            assert x.ndim == 3

        return x


class IndexDataset(Dataset):
    def __init__(self, num_data: int):
        self.num_data = num_data

    def __len__(self):
        return self.num_data

    # override
    def __getitem__(self, idx):
        return idx


class MyDataset(Dataset):
    def __init__(self, X, Y, m, batch_shape=torch.Size([]), data_device='cpu'):
        self.batch_shape = batch_shape
        self.X = torch.as_tensor(X, dtype=torch.get_default_dtype(), device=data_device)  
        self.Y = torch.as_tensor(Y, dtype=torch.get_default_dtype(), device=data_device)  
        self.m = torch.as_tensor(m, dtype=torch.get_default_dtype(), device=data_device)  

    def __len__(self):
        return self.X.shape[-2]

    def __getitem__(self, idx):
        """get data along inputs, not outputs"""
        if self.batch_shape == torch.Size([]):
            return self.X[idx], self.Y[idx], self.m[idx]
        else:
            slices_y = [slice(None)] * len(self.batch_shape) + [idx]
            sample_X = self.X[slices_y] if self.X.ndim > 2 else self.X[idx]
            sample_Y = self.Y[slices_y]
            masks = self.m[slices_y]
            return sample_X, sample_Y, masks

class synthetic_LVMOGP(LVMOGP):
    def __init__(self, input_kernels: list, latent_kernels: list, N_train: int, mean_pH: Tensor, likelihood: Likelihood,
                 pH_cov=1., D_x=1, num_outputs=2, Q=3, M_H=5, M_X=5,
                 latent_IP_joint=True, input_IP_joint=True):
        # mean_pH: (long, lat) spatial locs
        D_h = mean_pH.shape[-1]
        cov_pH = pH_cov * torch.ones_like(mean_pH)  
        pH = Prior_H(mean_pH, cov_pH)
        qH = Variational_H(Q, num_outputs, D_h)
        qU = Variational_inducing_dist(M_H, M_X)
        zH = Inducing_points("zH", Q=Q, num_inducing_points=M_H, num_dim=D_h, IP_joint=latent_IP_joint)  
        zX = Inducing_points("zX", Q=Q, num_inducing_points=M_X, num_dim=D_x, IP_joint=input_IP_joint)  

        super(synthetic_LVMOGP, self).__init__(
            input_kernels, latent_kernels, N_train, pH=pH, qH=qH,
            qU=qU, zH=zH, zX=zX, likelihood=likelihood, whitening=True, jitter=1e-6,
        )

class demo_LVMOGP(LVMOGP):
    def __init__(self, input_kernels: list, latent_kernels: list, N_train: int,
                 mean_pH: Tensor, zH_init: Tensor, zX_init: Tensor, likelihood: Likelihood,
                 pH_cov=1., latent_IP_joint=True, input_IP_joint=True, jitter=1e-6):
        # mean_pH: (long, lat) spatial locs
        num_outputs, D_h = mean_pH.shape[-2:]
        Q, M_H, _ = zH_init.shape[-3:]
        _, M_X, D_x = zX_init.shape[-3:]
        cov_pH = pH_cov * torch.ones_like(mean_pH)  
        pH = Prior_H(mean_pH, cov_pH)
        qH = Variational_H(Q, num_outputs, D_h)
        qU = Variational_inducing_dist(M_H, M_X)
        zH = Inducing_points("zH", Q=Q, num_inducing_points=M_H, num_dim=D_h, IP_init=zH_init, IP_joint=latent_IP_joint)
        zX = Inducing_points("zX", Q=Q, num_inducing_points=M_X, num_dim=D_x, IP_init=zX_init, IP_joint=input_IP_joint)

        super(demo_LVMOGP, self).__init__(
            input_kernels, latent_kernels, N_train, pH=pH, qH=qH,
            qU=qU, zH=zH, zX=zX, likelihood=likelihood, whitening=True, jitter=jitter,
        )

if __name__ == '__main__':
    from gpytorch.likelihoods import GaussianLikelihood
    import matplotlib.pyplot as plt

    def run_LVMOGP(args, train_X, train_Y, train_m):

        mean_pH = torch.zeros((args['Q'], args['num_outputs'], args['D_h']))

        model = synthetic_LVMOGP(
            input_kernels=args['input_kernels'], latent_kernels=args['latent_kernels'], N_train=args['N_train'],
            mean_pH=mean_pH, likelihood=args['likelihood'], D_x=args['D_x'], num_outputs=args['num_outputs'],
            Q=args['Q'], M_H=args['M_H'], M_X=args['M_X'])

        train_dataset = MyDataset(X=train_X, Y=train_Y, m=train_m)
        train_dataloader = DataLoader(train_dataset, batch_size=args['input_batch_size'], shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])

        model.train_lvmogp(train_dataloader, args['output_batch_size'], optimizer, args['epochs'], args['beta_u'],
                           args['beta_h'], args['print_epochs'])

        return model

    N_train = 64
    X_start, X_end = -3, 3
    train_X = torch.linspace(X_start, X_end, N_train).reshape(-1, 1)  
    train_Y_noiseless = torch.cat([torch.sin(train_X), torch.cos(train_X)], dim=-1)  
    train_Y = train_Y_noiseless + 0.05 * torch.randn_like(train_Y_noiseless)

    with torch.random.fork_rng():
        torch.manual_seed(123)
        train_m = torch.randint(0, 2, (N_train, 2), dtype=torch.bool)  

    args = {
        "input_kernels": [ScaleKernel(RBFKernel())],
        "latent_kernels": [ScaleKernel(RBFKernel())],
        "N_train": N_train,
        "likelihood": GaussianLikelihood(),
        "D_x": 1,
        "D_h": 2,
        "num_outputs": 2,
        "Q": 3,
        "M_H": 10,
        "M_X": 10,
        "input_batch_size": 16,
        "output_batch_size": 2,
        "lr": 0.02,
        "epochs": 1000,
        "beta_u": 1.,
        "beta_h": 1.,
        "print_epochs": 10
    }

    torch.set_default_dtype(torch.float64)
    model = run_LVMOGP(args, train_X, train_Y, train_m)

    n_test = 500
    test_X = torch.linspace(X_start - 1, X_end + 1, n_test).reshape(-1, 1)

    test_pred_means, test_pred_vars = model.predict(x_star=test_X, output_idx=None, num_samples=1, diag=True)  

    plt.figure(figsize=(16, 9))

    plt.subplot(1, 2, 1)  
    plt.plot(test_X.squeeze(-1), torch.sin(test_X.squeeze(-1)), color='black', label='sin(x)')
    plt.plot(test_X.squeeze(-1), test_pred_means[0, 0, :], color='blue', label='pred mean')
    plt.scatter(train_X.squeeze(-1)[train_m[:, 0]], train_Y[:, 0][train_m[:, 0]], marker='x', color='red', label='train data', s=10)
    plt.fill_between(
        test_X.squeeze(-1),
        test_pred_means[0, 0, :] - 1.96 * test_pred_vars[0, 0, :].sqrt(),
        test_pred_means[0, 0, :] + 1.96 * test_pred_vars[0, 0, :].sqrt(),
        alpha=0.2,
        color='blue',
    )
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(test_X.squeeze(-1), torch.cos(test_X.squeeze(-1)), color='black', label='cos(x)')
    plt.plot(test_X.squeeze(-1), test_pred_means[0, 1, :], color='blue', label='pred mean')
    plt.scatter(train_X.squeeze(-1)[train_m[:, 1]], train_Y[:, 1][train_m[:, 1]], marker='x', color='red', label='train data', s=10)
    plt.fill_between(
        test_X.squeeze(-1),
        test_pred_means[0, 1, :] - 1.96 * test_pred_vars[0, 1, :].sqrt(),
        test_pred_means[0, 1, :] + 1.96 * test_pred_vars[0, 1, :].sqrt(),
        alpha=0.2,
        color='blue',
    )
    plt.legend()

    plt.tight_layout()  # Adjust layout to avoid overlap
    plt.savefig('./synthetic_LVMOGP.pdf')

