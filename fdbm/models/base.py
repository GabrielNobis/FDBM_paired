import torch
import torch.nn as nn

from fdbm.utils.sb_utils import sample_from_brownian_bridge
from fdbm.models.sde_drift import SDEDrift
from fdbm.models.doobs_score import DoobHScore

from fdbm.diffusion import get_diffusivity_schedule

class AlignedSB(nn.Module):

    def __init__(self,
                 timestep_emb_dim: int = 64, 
                 n_layers: int = 3,
                 in_dim: int = 2,
                 out_dim: int = 2,
                 h_dim: int = 64,
                 activation: int = 'relu',
                 dropout_p: float = 0.0,
                 use_drift_in_doobs: bool = False,
                 args=None,
                 device='cpu',
                 **kwargs):
        super().__init__(**kwargs)

        self.sde_drift = SDEDrift(timestep_emb_dim=timestep_emb_dim,n_layers=n_layers, in_dim=in_dim,
                                  out_dim=out_dim, h_dim=h_dim, 
                                  activation=activation, dropout_p=dropout_p)

        self.doobs_h_score = DoobHScore(n_layers=n_layers, in_dim=in_dim,
                                     out_dim=out_dim, h_dim=h_dim,
                                     activation=activation, dropout_p=dropout_p,
                                     use_drift_in_doobs=use_drift_in_doobs)

        sde_drift_total_params = sum(p.numel() for p in self.sde_drift.parameters())
        print(f'Initialized sde_drift with {sde_drift_total_params} parameters',flush=True)

        doobs_h_score_total_params = sum(p.numel() for p in self.doobs_h_score.parameters())
        print(f'Initialized doobs_h_score with {doobs_h_score_total_params} parameters',flush=True)

        if args is not None:
            self.dif = get_diffusivity_schedule(args.max_diffusivity, H=args.H, K=args.K, norm=args.norm,device=device)

    def forward(self, data):
        if data.pos_t is None:
            assert data.pos_T is not None
            print("Sampling from brownian bridge...")
            data.pos_t = sample_from_brownian_bridge(data.t, x_0=data.pos_0, x_T=data.pos_T)

        drift_x = self.sde_drift(data.pos_t, data.t)
        doobs_score_x = self.doobs_h_score(data.pos_t, data.pos_T, drift_x, data.t)
        
        if data.pos_T is not None:
            drift_x_T = self.sde_drift(data.pos_T, torch.ones_like(data.t))
            doobs_score_x_T = self.doobs_h_score(data.pos_T, data.pos_T, drift_x_T, torch.ones_like(data.t))
            return drift_x, doobs_score_x, doobs_score_x_T

        return drift_x, doobs_score_x

    def run_drift(self, x, t, x0=None):
        if not len(t.shape):
            # unsqueeze to size of x
            for idx in range(len(x.shape)):
                t = t.unsqueeze(idx)

            # Expand to x-shape except for last dim
            t = t.expand(*x.shape[:-1], 1)

        drift = self.sde_drift(x, t)
        return drift

    def run_doobs_score(self, x, x_T, t, x0=None):
        drift = self.sde_drift(x, t)
        return self.doobs_h_score(x, x_T, drift, t)

class FDBMToy(nn.Module):

    def __init__(self,
                 timestep_emb_dim: int = 64, 
                 n_layers: int = 3,
                 in_dim: int = 2,
                 out_dim: int = 2,
                 h_dim: int = 64,
                 activation: int = 'relu',
                 dropout_p: float = 0.0,
                 args=None,
                 device='cpu',
                 **kwargs):
        super().__init__(**kwargs)

        self.sde_drift = SDEDrift(timestep_emb_dim=timestep_emb_dim,n_layers=n_layers, in_dim=in_dim,
                                  out_dim=out_dim, h_dim=h_dim, 
                                  activation=activation, dropout_p=dropout_p)

        if args is not None:
            self.dif = get_diffusivity_schedule(args.max_diffusivity, H=args.H, K=args.K, norm=args.norm,device=device)

        sde_drift_total_params = sum(p.numel() for p in self.sde_drift.parameters())
        print(f'Initialized sde_drift with {sde_drift_total_params} parameters',flush=True)

    def forward(self, data):
        stacked_in = torch.cat((data.pos_0,data.pos_t),dim=1)
        drift_x = self.sde_drift(stacked_in, data.t)
        
        if data.pos_T is not None:
            stacked_in = torch.cat((data.pos_0,data.pos_T),dim=1)
            return drift_x
        
        return drift_x
    
    def run_drift(self, x, t, x0=None):
        if not len(t.shape):
            # unsqueeze to size of x
            for idx in range(len(x.shape)):
                t = t.unsqueeze(idx)

            # Expand to x-shape except for last dim
            t = t.expand(*x.shape[:-1], 1)
        if x0 is not None:
            x = torch.cat((x0,x),dim=1)
        drift = self.sde_drift(x, t)
        return drift