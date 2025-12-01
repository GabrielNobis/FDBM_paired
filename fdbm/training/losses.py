import torch
import torch.nn as nn
import numpy as np
from functools import partial
import time

from fdbm.utils.sb_utils import beta
from fdbm.diffusion import get_diffusivity_schedule
from fdbm.utils.definitions import DEVICE


# ----------- Schroedinger Bridges with Paired Samples -------------------

def loss_function_sbalign(
        drift_x_pred, 
        doobs_score_x_pred, 
        doobs_score_xT_pred,
        data, 
        g, 
        drift_weight: float = 1.0,
        reg_weight_T: float = 1.0,
        reg_weight_t: float = 1.0,
        apply_mean: bool = True,
        criterion = None,
        loss_weight = 1.0, 
        K: int = 0,
    ):
    
    mean_dims = (0, 1) if apply_mean else 1

    if K > 0:
        t_diff = data.cond_var_t 
    else:
        t_diff = ((g(data.t)**2)*(1-data.t)) # this is only valid for a constant choice of the diffusion function

    x_diff = (data.pos_T - data.pos_t)

    bb_drift_true = (x_diff) / t_diff
    bb_drift_pred = drift_x_pred + doobs_score_x_pred

    if criterion is None:
        bb_loss = ((bb_drift_pred - bb_drift_true) ** 2).mean(mean_dims)
    else:
        bb_loss = loss_weight * criterion(bb_drift_pred, bb_drift_true)

    if doobs_score_xT_pred is not None:
        reg_loss_T = (doobs_score_xT_pred ** 2).sum(dim=-1).mean()
    else:
        reg_loss_T = torch.tensor(0.0, requires_grad=True)
    reg_loss_t = (doobs_score_x_pred ** 2).sum(dim=-1).mean()

    loss = drift_weight * bb_loss + reg_weight_T * reg_loss_T + reg_weight_t * reg_loss_t
    
    loss_dict = {
        "loss": loss.item(), 
        "bb_loss": bb_loss.item(),
        "reg_loss_T": reg_loss_T.item(), 
        "reg_loss_t": reg_loss_t.item()
    }

    for key, value in loss_dict.items():
        loss_dict[key] = np.round(value, 4)

    return loss, loss_dict


def loss_function_fdbm(
        drift_x_pred, 
        data, g, 
        drift_weight: float = 1.0, 
        apply_mean: bool = True,
        K: int = 0,
        loss_weight: float = 1.0,
        criterion = None,
    ):
    
    mean_dims = (0, 1) if apply_mean else 1

    if K > 0:
        beta_t_diff = data.cond_var_t
    else:
        beta_t_diff = ((g(data.t)**2)*(1-data.t))

    x_diff = (data.pos_T - data.pos_t)
    drift_true = (x_diff) / beta_t_diff

    if criterion is None:
        loss = drift_weight * ((drift_x_pred - drift_true) ** 2).mean(mean_dims)
    else:
        loss =  loss_weight * criterion(drift_x_pred, drift_true)
    
    loss_dict = {
        "loss": np.round(loss.item(), 4), 
    }

    return loss, loss_dict


def loss_fn_from_args(args):

    g = get_diffusivity_schedule(args.max_diffusivity, H=args.H, K=args.K, norm=args.norm).g
    
    if args.task == "synthetic":
        if args.method in ['fdbm','abm']:
            loss_fn_base = loss_function_fdbm
            loss_weight = 1/args.inference_steps
            criterion = nn.MSELoss()
            loss_fn = partial(
                loss_fn_base,
                g=g,
                drift_weight=args.drift_weight, 
                criterion=criterion,
                loss_weight=loss_weight,
                K=args.K,
            )
        elif args.method == 'sbalign':
            loss_fn_base = loss_function_sbalign
            loss_weight = 1/args.inference_steps
            criterion = nn.MSELoss()
            loss_fn = partial(
                loss_fn_base,
                g=g,
                drift_weight=args.drift_weight, 
                reg_weight_T=args.reg_weight_T,
                reg_weight_t=args.reg_weight_t,  
                criterion=criterion,
                loss_weight=loss_weight,
                K=args.K,
            )
    elif args.task == "conf":
        if args.method in ['fdbm','abm']:
            loss_fn_base = loss_function_fdbm
            loss_fn = partial(
                loss_fn_base,
                drift_weight=args.drift_weight, 
                g=g, 
                K=args.K
            )
        elif args.method == 'sbalign':
            loss_fn_base = loss_function_sbalign
            loss_fn = partial(
                loss_fn_base,
                drift_weight=args.drift_weight, 
                reg_weight_T=args.reg_weight_T,
                reg_weight_t=args.reg_weight_t,
                steps_num=args.inference_steps,
                g=g, 
                K=args.K
            )
    
    return loss_fn
