from argparse import ArgumentParser
from scipy.stats import wasserstein_distance
import torch
import pandas as pd
from pathlib import Path

import os
os.chdir(os.getcwd())
import sys
sys.path.append(os.getcwd())

#os.chdir("./scripts")
#print(f"Our CWD is {os.getcwd()}")

from toy.experiment import Experiment


def str2bool(s):
    # s is already bool
    if isinstance(s, bool):
        return s
    # s is string repr. of bool
    if s.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif s.lower() in ("no", "false", "f", "n", "0"):
        return False
    # s is something else
    else:
        return s

def list_of_ints(arg):
    return list(map(int, arg.split(",")))

def list_of_floats(arg):
    return list(map(float, arg.split(",")))

def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    print(f"Ensured directory exists: {path}")

def args():
    ap = ArgumentParser()
    ap.add_argument("--method", default="fdbm", choices=["fdbm","sbalign"]) # Using fdbm here with K=0 and H=0.5 will redcue to Augmented Bridge Matching (ABM)
    ap.add_argument("--dataset", type=str, default='moon',choices=['moon','tshaped']) 
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--num_aug", type=list_of_ints, default=[0,5]) 
    ap.add_argument("--hurst", type=list_of_floats, default=[0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]) 
    ap.add_argument("--n_epoch", type=int, default=20) 
    ap.add_argument("--samples_num", type=int, default=10000) 
    ap.add_argument("--norm", type=str2bool, default=True, help='whether to normalize the terminal variance of the diffusion process across all values of H')
    ap.add_argument("--g_max", type=float, default=0.2)
    ap.add_argument("--path_to_csv", type=str2bool, default='toy/reproducibility/results', help='path where the results are stored in a csv')
    return ap

parser = args()
args = parser.parse_args()

WSD_mean = torch.zeros(len(args.num_aug),len(args.hurst))
WSD_std = torch.zeros(len(args.num_aug),len(args.hurst))

for i,K in enumerate(args.num_aug):
    for j,H in enumerate(args.hurst):
        if K==0 and H!=0.5:
            continue
        wsd1 = torch.zeros(args.runs)
        wsd2 = torch.zeros(args.runs)
        for n in range(args.runs):
            in_dim = 4 if args.method =='fdbm' else 2

            if args.dataset == 'moon':
                try:
                    Experiment.run(f"--method={args.method} --dataset=moon --h_dim=64  --n_layers=2  --n_epochs={args.n_epoch} --timestep_emb_dim=32  --in_dim={in_dim} --out_dim=2 --max_diffusivity={args.g_max}  --H={H}  --K={K} --norm={args.norm} --use_drift_in_doobs=True  --activation=silu").save("moon_quant")
                except ValueError:
                    continue

                sampler = Experiment.load("moon_quant")

            elif args.dataset == 'tshaped':
                try:
                    Experiment.run(f"--dataset=diagonal_matching_inverse  --h_dim=32  --n_layers=3  --n_epochs={args.n_epoch}  --reg_weight=1.  --timestep_emb_dim=32  --in_dim={in_dim} --out_dim=2  --max_diffusivity={args.g_max} --H={H} --K={K} --norm={args.norm} --use_drift_in_doobs=True  --activation=selu").save("tshaped_quant")
                except ValueError:
                    continue

                sampler = Experiment.load("tshaped_quant")

            samples = sampler.sample(samples_num=args.samples_num, trials_num=1)
            if len(samples.shape) == 4:
                samples = samples[:,:,:,0]

            marginals = sampler.get_marginals(samples_num=args.samples_num)
            wsd1[n] = wasserstein_distance(samples[-1,:,0],marginals['final'][:,0])
            wsd2[n] = wasserstein_distance(samples[-1,:,1],marginals['final'][:,1])

        mean_wsd1 = torch.mean(wsd1)
        mean_wsd2 = torch.mean(wsd2)
        WSD_mean[i,j] = (mean_wsd1 + mean_wsd2)/2

        std_wsd1 = torch.std(wsd1)
        std_wsd2 = torch.std(wsd2)
        WSD_std[i,j] = (std_wsd1 + std_wsd2)/2

        df_mean = pd.DataFrame(WSD_mean.numpy(), columns=args.hurst)
        df_std = pd.DataFrame(WSD_std.numpy(), columns=args.hurst)

        # Save the DataFrame to a CSV file
        ensure_dir(args.path_to_csv)
        csv_mean = f"{args.method}_mean_{args.dataset}_norm{args.norm}_K{len(args.num_aug)}_H{len(args.hurst)}_runs{args.runs}_samples{args.samples_num}_epochs{args.n_epoch}.csv"
        csv_std = f"{args.method}_std_{args.dataset}_norm{args.norm}_K{len(args.num_aug)}_H{len(args.hurst)}_runs{args.runs}_samples{args.samples_num}_epochs{args.n_epoch}.csv"
        df_mean.to_csv(os.path.join(args.path_to_csv, csv_mean), index=False)
        df_std.to_csv(os.path.join(args.path_to_csv, csv_std), index=False)





