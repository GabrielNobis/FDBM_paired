import numpy as np
from functools import partial

import torch
import numpy as np
from scipy.integrate import solve_ivp
from torch.distributions import MultivariateNormal
import time

import torch.nn as nn
import einops

from .optimal_weights import omega_optimized, gamma_by_gamma_max, gamma_by_r, gamma_by_range

from fdbm.utils.definitions import DEVICE

def get_diffusivity_schedule(g_max, H=0.5, K=5, norm=False, device='cpu'):
    return FractionalBrownianBridge(H=H, K=K, norm=norm, g_max=g_max, device=device)

class FractionalBrownianBridge(nn.Module):

    """Approximate fractional brownian bridge process"""

    def __init__(self, H=0.5, K=5, norm=False, g_max=1.0, gamma_max=20.0, gamma_min=0.1, approx_cov=False, T=1.0, pd_eps=1e-4, threshold=1e-3, device="cpu"):
        super(FractionalBrownianBridge, self).__init__()

        """parameters of fBM approximation"""
        self.register_buffer("H", torch.as_tensor(H, device=device))
        self.register_buffer("gamma_max", torch.as_tensor(gamma_max, device=device))
        if gamma_min is not None:
            self.register_buffer("gamma_min", torch.as_tensor(gamma_min, device=device))
        self.register_buffer("T", torch.as_tensor([[T]], device=device))
        self.K = K

        """parameters of augmented process"""
        self.aug_dim = K + 1
        self.pd_eps = pd_eps

        self.approx_cov = approx_cov
        self.device = device

        if self.K > 0:
            if self.K == 1:
                gamma = gamma_by_r(K, torch.sqrt(torch.tensor(gamma_max)), device=device)
            else:
                if gamma_min is None:
                    gamma = gamma_by_gamma_max(K, self.gamma_max, device=device)
                else:
                    gamma = gamma_by_range(K, self.gamma_min, self.gamma_max)

            output , cost = omega_optimized(
                gamma, self.H, self.T, return_Ab=True, device=device,return_cost=True
            )
            omega, A, b = output

        else:
            gamma = torch.tensor([0.0])
            omega = torch.tensor([1.0])
            A = torch.tensor([1.0])
            b = torch.tensor([1.0])

        self.register_buffer("gamma", torch.as_tensor(gamma, device=device)[None, :])
        self.register_buffer("gamma_i", self.gamma[:, :, None].clone())
        self.register_buffer("gamma_j", self.gamma[:, None, :].clone())
        self.dt = 1/100
        self.update_omega(omega,A=A,b=b)
        self.check_dt(self.dt)
        self.g_max =  torch.tensor(g_max)
        #self.g_max = torch.sqrt(torch.tensor(0.1370))
        #self.g_max = 0.1
        self.norm = norm

        if self.K>0:
            self.solve_cov_ode()

        #only valid for T=1
        if self.norm and K>0:
            var_T = self.cond_var(torch.zeros_like(self.T),self.T,self.omega,self.gamma,1.0)
            omega = omega/torch.sqrt(var_T)[:,0]
            self.update_omega(omega,A=A,b=b)
        
        
        var_T = self.cond_var(torch.zeros_like(self.T),self.T,self.omega,self.gamma,self.g_max)

        if self.K>0:
            F = torch.zeros(K+1,K+1)
            F[:,1:] = -torch.vstack([self.g_max*(self.omega * self.gamma)[0],torch.diag(self.gamma[0])])
            self.register_buffer("F_t", F)

            G = torch.ones(K+1)
            G[0] = torch.sum(self.omega) * self.g_max
            self.register_buffer("G_t", G)
        
        self.t_max = self.largest_t(self.omega,self.gamma,self.g_max,threshold=threshold,T=1.0)

    def update_omega(self,omega,A=None,b=None):

        if A is not None:
            self.register_buffer("A", torch.as_tensor(A, device=self.device))
        if b is not None:
            self.register_buffer("b", torch.as_tensor(b, device=self.device))

        self.register_buffer("omega", torch.as_tensor(omega, device=self.device)[None, :].clone())
        self.register_buffer('sum_omega', torch.sum(self.omega))
        self.register_buffer("omega_i", self.omega[:, :, None].clone())
        self.register_buffer("omega_j", self.omega[:, None, :].clone())
        self.double_sum_omega = torch.sum(self.omega_i * self.omega_j, dim=(1, 2))

    def check_dt(self, dt):
        assert self.gamma_max * dt < .5, 'dt too large for stable integration, please reduce dt or decrease largest gamma'

    def g(self,t):
        if self.K>0:
            return self.g_max
        else:
            return torch.ones_like(torch.tensor(t)) * self.g_max
    
    # def g(self,t):
    #     return self.g_max
    
    def zeta(self,s,t,gamma,g):

        # expects s,t of shape (batch_size,1,1) and s<=t
        # expects omega and gamma of shape (1,1,K)

        return g*(torch.exp(-gamma*(t-s))-1)

    def meanX(self,s,t,x,Y,omega,gamma,g):

        # compute E[X(t)|Z_s=z) with s<t - Z_s = (x,Y) 
        # expects t,s of shape (batch_size,1)
        # mean of X_T conditioned on Z_t = (x,Y)

        s = s[:,:,None]
        t = t[:,:,None]
        gamma = gamma[:,None,:]
        omega = omega[:,None,:]

        weight = omega * self.zeta(s,t,gamma,g) 
        y_part = (torch.sum(weight*Y, dim=-1)) 

        return x + y_part

    def meanY(self,s,t,Y,gamma):

        # compute E[Y(t)|Z_s=z) with s<t 
        # expects t,s of shape (batch_size,1)
        # expects gamma of shape (1,K)
        # mean of X_T conditioned on Z_t = (x,Y)

        s = s[:,:,None]
        t = t[:,:,None]
        gamma = gamma[:,None,:]

        return torch.exp(-gamma*(t-s)) * Y

    def meanZ(self,s,t,x,Y,omega,gamma,g):

        mean_x = self.meanX(s,t,x,Y,omega,gamma,g)
        mean_y = self.meanY(s,t,gamma)
    
        return torch.cat([mean_x.unsqueeze(-1),mean_y],dim=-1)
    
    def cond_var(self,s,t,omega,gamma,g):
                    
        # compute cov(X(t),X(t)|Z_s=z) with s<t 
        # expects s,t of shape (batch_size,1)
        # expects omega,gamma of shape (1,K)

        return self.covX(s,t,t, omega, gamma, g)[:,None]
    
    def largest_t(self,omega,gamma,g_max,threshold=1e-3,T=1.0,steps=10000,start_from=0.7, verbose=False):

        """
        Find largest t_max with \sigma_{1|t_{max}} > threshold such that 1/\sigma_{1|t} < 1/threshold for all t\in[0,t_max]
        """

        t_max = torch.tensor(1e-3)
        T = torch.tensor(T,device=omega.device)
        for t in torch.linspace(start_from,T,steps=steps):
            if self.K==0:
                sigma_Tt = (g_max**2)*(1-t)
                if sigma_Tt < threshold:
                    t_max = t
                    break
            else:
                sigma_Tt = self.cond_var(t[None,None].to(omega.device),T[None,None],omega,gamma,g_max)
                if sigma_Tt < threshold:
                    t_max = t
                    break
        if verbose:
            print(f'Setting t_max={t_max}')

        return t_max

    def covX(self,s,t,T, omega, gamma, g):

        # compute cov(X(t),X(T)|Z_s=z) with s<t<=T 
        # expects t,s of shape (batch_size,1)
        # expects T of shape (1,)
        # expects omega,gamma of shape (1,K)

        t = t[:,:,None]
        s = s[:,:,None] 
        T = T[:,:,None]

        omega_ij = omega[:,:,None]*omega[:,None,:]
        gamma_i = gamma[:,:,None]
        gamma_j = gamma[:,None,:]
        gamma_ij =  gamma_i + gamma_j

        weight = omega_ij/ gamma_ij
        #S = weight * (torch.exp(t*gamma_ij) -torch.exp(s*(gamma_ij))) * torch.exp(-T*gamma_j - t*gamma_i) 
        S = weight * (torch.exp(-(T-t)*gamma_j) - torch.exp(-(T-s)*gamma_j)* torch.exp(- (t-s)*gamma_i))
        return g**2 * (torch.sum(S,axis=(1,2)))
    
    def covYX(self,t,T, omega, gamma, g):

        # compute cov(Y(t),X(T)) with s<t<=T 
        # expects t of shape (batch_size,1)
        # expects T of shape (1,)
        # expects omega,gamma of shape (1,K)

        t = t[:,:,None] 
        T = T[:,:,None]

        gamma_l = gamma[:,:,None] #dim of Y_l
        omega_k = omega[:,None,:]
        gamma_k = gamma[:,None,:]

        weight = omega_k/(gamma_l+gamma_k) #dim of omega_k in X
        #S = weight *(torch.exp(t*(gamma_l+gamma_k)) -1)*torch.exp(-t*gamma_l-T*gamma_k)
        S = weight * (torch.exp(-(T-t)*gamma_k) -torch.exp(-t*gamma_l-T*gamma_k))

        return g * torch.sum(S,axis=2)

    def covY(self,s,t,T,gamma):

        # compute cov(Y(t),Y(T)|Z_s=z) with s<t<=T 
        # expects s,t of shape (batch_size,1)
        # expects T of shape (1,1)
        # expects omega,gamma of shape (1,K)
        
        t = t[:,:,None] 
        s = s[:,:,None]
        T = T[:,:,None]

        gamma_i = gamma[:,:,None]
        gamma_j = gamma[:,None,:]
        gamma_ij =  gamma_i + gamma_j

        #return (torch.exp(-T*gamma_j-t*gamma_i)*(torch.exp(t*gamma_ij)-torch.exp(s*gamma_ij)))/gamma_ij
        return (torch.exp(-(T-t)*gamma_j) - torch.exp(-(T-s)*gamma_j)*torch.exp(-(t-s)*gamma_i))/gamma_ij
        #return (torch.exp(-T*gamma_j-t*gamma_i)*(torch.exp(t*(gamma_i + gamma_j))-torch.exp(s*(gamma_i + gamma_j))))/gamma_ij

    def covZ(self,t,T, omega, gamma, g, s=None, eps=1e-3):

        # compute cov(X(t),X(T)|Z_s=z) with s<t<=T 
        # expects s,t of shape (batch_size,1)
        # expects T of shape (1,1)
        # expects omega,gamma of shape (1,K)

        s = torch.zeros_like(t) if s is None else s

        K = omega.shape[1]
        bs = t.shape[0]
        Sig = torch.zeros(bs, K+1,K+1)
        
        Sig_xy = self.covYX(t,T, omega, gamma, g)
        Sig[:,0,0] = self.covX(s,t,T, omega, gamma, g)

        Sig[:,1:,0] = Sig_xy
        Sig[:,0,1:] = Sig_xy
        Sig[:,1:,1:] = self.covY(s,t,T, gamma)

        # I_eps = torch.eye(K, K)[None, :, :] * torch.ones((bs, K, K)) * eps * torch.exp(-2 * gamma * t)[:, :, None]
        # Sig[:,1:,1:] = Sig[:,1:,1:] + I_eps
        # Sig[:,0,0] += eps

        if self.K>5:
            eps = 1e-2
        Sig = Sig + torch.eye(K+1, K+1)[None, :, :] * torch.ones((bs, K+1, K+1)) * eps

        assert ((torch.diag(Sig[0])>0).all()), f'Found negativ variance: \n {torch.diag(Sig[0])<0}'

        return Sig

    def sample_pinned(self,t,T,x0,xT,omega,gamma,g):

        K = omega.shape[1]

        bs = t.shape[0]
        d = x0.shape[1]

        s = torch.zeros_like(t)

        mu = torch.zeros(bs,d,K+1)
        mu[:,:,0] = x0

        Sig_zx = torch.zeros(bs,K+1)
        
        Sig_zx[:,0] = self.covX(s,t,T, omega, gamma, g)
        Sig_zx[:,1:] = self.covYX(t,T, omega, gamma, g)

        var = self.covX(s,T,T, omega, gamma, g)[:,None,None]

        mu_bar = mu + (1/var) * Sig_zx[:,None,:] * ((xT-x0)[:,:,None])

        Sig_bar = self.covZ(t,t,omega,gamma,g) - (1/var) * (Sig_zx[:,:,None] * Sig_zx[:,None,:])

        assert (Sig_bar.transpose(1, 2) == Sig_bar).all(), f'Covariance is not symmetric'

        noise = sample_from_batch_multivariate_normal(Sig_bar,c=d,h=1,w=1,batch_size=bs, aug_dim=K+1)[:,:,0,0,:]

        return mu_bar + noise

    def input_transform(self,x,Y,t,T,omega,gamma,g):

        t = t[:,:,None]
        T = T[:,:,None]
        gamma = gamma[:,None,:]
        omega = omega[:,None,:]

        weight = omega * self.zeta(t,T,gamma,g) 
        y_part = (torch.sum(weight*Y, dim=-1)) 

        return x + y_part

    def score(self, score_x, t,T, omega, gamma, g_max):

        # expects the output of a score model of dimension (batch_size1,batch_size2)

        #std = torch.sqrt(self.cond_var(t,T,omega,gamma,g_max))

        t = t[:,:,None]
        T = T[:,:,None]
    
        omega = omega[:,None,:]
        gamma = gamma[:,None,:]

        if torch.any((t-T)==0):
            std = torch.tensor(1.0)

        std = 1.0
        scale = torch.ones(1,1,self.K+1).to(DEVICE)
        scale[:,:,1:] = omega * self.zeta(t,T, gamma, g_max)
        return scale * score_x[:,:,None] / std
    
    def mu(self,t):
        return torch.zeros_like(t)
    
    def print_approximation_accuracy(self):
        print('[MA-fBM approximation error]')
        error_fn = lambda hurst: ma.omega_optimized_2(self.gamma, hurst, self.time_horizon, return_cost='normalized')

        hursts = jnp.linspace(0., 1., 21)
        _, errors = jax.vmap(error_fn)(hursts)
        hursts_str = [f'{hurst:.2f}' for hurst in hursts]
        errors_str = [f'{error:.2f}' for error in errors]
        print(f'H: {" ".join(hursts_str)}')
        print(f'E: {" ".join(errors_str)}')
        return

    def compute_YiYj(self,t):
        t = t[:,:,None]
        sum_gamma = self.gamma_i + self.gamma_j
        return ((1-torch.exp(-t*sum_gamma))/sum_gamma)

    def numpy_compute_YiYj(self,t):
        gamma_i, gamma_j = self.gamma[0,:, None].cpu().numpy(), self.gamma[0,None, :].cpu().numpy()
        return (1 - np.exp(- (gamma_i + gamma_j) * t.cpu().numpy())) / (gamma_i + gamma_j)

    def func(self,t, S):
        num_k = self.K
        t = torch.as_tensor(t)
        A = np.zeros((num_k + 1, num_k + 1))
        A[0, 0] = 2 * self.mu(t).cpu().numpy()
        A[0, 1:] = - 2 * (self.g(t) * self.omega[0] * self.gamma[0]).cpu().numpy()
        A[1:, 1:] = np.diag((self.mu(t) - self.gamma[0]).cpu().numpy())
        b = np.zeros(num_k + 1)
        b[0] = (self.omega[0].cpu().numpy().sum() * self.g(t).cpu().numpy()) ** 2
        b[1:] = self.g(t).cpu().numpy() * (
                    self.omega[0].cpu().numpy().sum() - self.numpy_compute_YiYj(t) @ (self.omega[0] * self.gamma[0]).cpu().numpy())

        return A @ S + b

    def solve_cov_ode(self,t=0.0):
        S_0 = np.zeros(self.K + 1)
        self.approx_sigma = solve_ivp(self.func, (t, 1.), S_0, dense_output=True)

    def compute_cov(self,t):

        S = self.approx_sigma.sol(t[:,0].cpu().numpy())
        cov = np.zeros((self.K + 1, self.K + 1,t.shape[0]))
        cov[0, :, :] = S
        cov[:, 0, :] = S
        sigma_t = torch.from_numpy(cov.astype(np.float32)).to(t.device).permute(2,1,0)
        sigma_t[:,1:,1:] = self.compute_YiYj(t)
        return sigma_t

def sample_from_batch_multivariate_normal(cov_matrix, c=2,h=1,w=1,batch_size=128, aug_dim=6, device='cpu'):

    # Ensure covariance matrix has shape [batch_size, dim, dim]
    assert cov_matrix.shape == (batch_size, aug_dim, aug_dim), "Covariance matrix must have shape [batch_size, dim, dim]"
    
    # Zero mean for each distribution in the batch
    mean = torch.zeros(batch_size, aug_dim,device=device)

    # Create the batch of Multivariate Normal distributions
    mvn = MultivariateNormal(mean, covariance_matrix=cov_matrix)

    # Sample from the distribution
    n_samples = int(c*h*w)
    samples = mvn.sample(sample_shape=(n_samples,))  # Samples will have shape [n_samples, batch_size, dim]        
    samples = einops.rearrange(samples, '(C H W) B K -> B C H W K', C=c, H=h, W=w)

    return samples


def matrix_vector_mp(A,v):

    '''
    # Example tensors A and v
    A = torch.randn(128, 2, 1, 1, 6, 6)
    v = torch.randn(128, 2, 1, 1, 6)
    '''

    # Reshape v to have an additional dimension at the end (for matrix-vector multiplication)
    v_expanded = v.unsqueeze(-1)  # Now v has shape (128, 2, 1, 1, 6, 1)

    # Perform matrix-vector multiplication
    result = A @ v_expanded

    # The result will have shape (128, 2, 1, 1, 6, 1), you might want to remove the last dimension
    result_squeezed = result.squeeze(-1)  # Now result has shape (128, 2, 1, 1, 6)

    return result_squeezed