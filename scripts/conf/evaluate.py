import os
import argparse
import json
import yaml
from argparse import FileType

import numpy as np
import torch
from torch_geometric.loader import DataLoader

import os
os.chdir(os.getcwd())
import sys
sys.path.append(os.getcwd())

from fdbm.data import ListDataset
from fdbm.utils.ops import to_numpy
from fdbm.diffusion import get_diffusivity_schedule
from fdbm.utils.definitions import DEVICE

from proteins.conf_engine import ConfEngine
from proteins.models import build_model_from_args as build_conf_model


def prepare_inference_setup(args):

    with open(f'{args.log_dir}/{args.run_name}/config_train.yml') as f:
        model_args = argparse.Namespace(**yaml.full_load(f))
        model_args.method = args.method
        print('model_args.method',model_args.method)


    model = build_conf_model(model_args)
    model_dict = torch.load(f"{args.log_dir}/{args.run_name}/{args.model_name}",
                             map_location='cpu')
    if args.model_name == "last_model.pt":
        model.load_state_dict(model_dict["model"])
    else:                                 
        model.load_state_dict(model_dict)
    model.to(DEVICE)

    print('In prepare inference setup',args.data_dir)
    # Data Loader
    resolution = model_args.resolution
    processed_dir = f"{args.data_dir}/processed/{args.dataset}/resolution={resolution}"
    if model_args.center_conformations:
        processed_dir += f"_centered_conf"

    with open(f"{args.data_dir}/raw/{args.dataset}/splits.json", "r") as f:
        splits = json.load(f)

    pdb_ids = splits['test']
    pdb_ids = [pdb_id for pdb_id in pdb_ids
            if os.path.exists(f"{processed_dir}/{pdb_id}.pt")]

    print(f"Inference on {len(pdb_ids)} proteins", flush=True)
    print(flush=True)

    loader = DataLoader(
        ListDataset(
            processed_dir=processed_dir,
            id_list=pdb_ids
        )
    )

    # Inference Engine
    if args.method in ['sbalign','abm','fdbm']:
        # K = model_args.K
        # H = model_args.H
        # H=0.5
        # K=0
        dif = get_diffusivity_schedule(
            g_max=model_args.max_diffusivity,
            K=model_args.K,
            H=model_args.H,
            norm=model_args.norm
        )

        engine = ConfEngine(
            samples_per_protein=args.n_samples,
            inference_steps=args.inference_steps,
            model=model, dif=dif
        )
    else:
        engine = None

    return loader, model, engine


def run_inference(data, engine):
    _, metrics = engine.generate_conformations(data, apply_mean=False)
    return metrics

def print_statistics(args, metrics):
    # RMSD associated statistics
    print(f'\nMetrics ({args.n_samples} sampled trajectory per conformation - {args.inference_steps} sampling steps):')
    rmsds = []
    delta_rmsd  = []
    for pdb_dict in metrics:
        rmsds.extend(pdb_dict['rmsd'])
        delta_rmsd.extend((np.array(pdb_dict['init_rmsd']) - np.array(pdb_dict['rmsd'])).tolist())
    
    mean_rmsd = np.mean(rmsds)
    median_rmsd = np.median(rmsds)
    std_rmsd = np.std(rmsds)
    print(f"RMSD | Mean={mean_rmsd} | Std={std_rmsd} | Median={median_rmsd}")
    
    rmsd_stats = "RMSD (% <) | "
    rmsd_stats_list = []
    for threshold in [2.0, 5.0, 10.0]:
        rmsd_less_than_threshold = (np.asarray(rmsds) < threshold).mean()
        rmsd_stats_list.append(rmsd_less_than_threshold)
        rmsd_stats += f"{threshold}= {rmsd_less_than_threshold} | "
    print(rmsd_stats)

    mean_delta_rmsd = np.mean(delta_rmsd)
    median_delta_rmsd = np.median(delta_rmsd)
    std_delta_rmsd = np.std(delta_rmsd)
    print(f"DELTA RMSD | Mean={mean_delta_rmsd} | Std={std_delta_rmsd} | Median={median_delta_rmsd}")
    print(flush=True)
    return mean_rmsd, median_rmsd, std_rmsd, rmsd_stats_list, mean_delta_rmsd, std_delta_rmsd, median_delta_rmsd

def print_averaged_statistics(args, metrics):
    print(f'Metrics ({args.n_samples} sampled trajectory per conformation - {args.inference_steps} sampling steps - averaged over {args.sampling_runs} sampling runs):')
    print(f"RMSD | Mean={metrics['mean_rmsd']} | Std={metrics['std_rmsd']} | Median={metrics['median_rmsd']}")
    rmsd_stats = "RMSD (% <) | "
    for i, threshold in enumerate([2.0, 5.0, 10.0]):
        rmsd_stats += f"{threshold}= {metrics['rmsd_stats'][i]} | "
    print(rmsd_stats)
    print(f"DELTA RMSD | Mean={metrics['mean_delta_rmsd']} | Std={metrics['std_delta_rmsd']} | Median={metrics['median_delta_rmsd']}")
    return

def add_statistics(args, metrics, av_metrics):
    mean_rmsd, median_rmsd, std_rmsd, rmsd_stats_list, mean_delta_rmsd, std_delta_rmsd, median_delta_rmsd = print_statistics(args=args, metrics=metrics)
    av_metrics['mean_rmsd'] += mean_rmsd
    av_metrics['median_rmsd'] += median_rmsd
    av_metrics['std_rmsd'] += std_rmsd
    for i,rmsd_stat in enumerate(rmsd_stats_list):
        av_metrics['rmsd_stats'][i] += rmsd_stats_list[i]
    av_metrics['mean_delta_rmsd'] += mean_delta_rmsd
    av_metrics['std_delta_rmsd'] += std_delta_rmsd
    av_metrics['median_delta_rmsd'] += median_delta_rmsd

    return av_metrics

def average_statistics(metrics, runs):
    for metric in metrics.keys():
        if metric == 'rmsd_stats':
            for i,rmsd_stat in enumerate(metrics['rmsd_stats']):
                metrics['rmsd_stats'][i] = float(metrics['rmsd_stats'][i]/runs)
        else:
            metrics[metric] = float(metrics[metric]/runs)
    return metrics

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
    
def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default='fdbm/data', type=str)
    parser.add_argument("--log_dir", default='logs', type=str)
    parser.add_argument("--run_name", type=str)
    parser.add_argument("--model_name", default='best_inference_epoch_model.pt', type=str)
    parser.add_argument("--method", default='fdbm', type=str, choices=['fdbm','sbalign','abm'])

    parser.add_argument("--dataset", default="d3pm", type=str)
    parser.add_argument("--n_samples", default=1, type=int)
    parser.add_argument("--inference_steps", default=100, type=int)
    parser.add_argument("--sampling_runs", default=2, type=int)
    parser.add_argument("--max_diffusivity", default=0.2, type=float)

    args = parser.parse_args()
    return args
    
def main():
    args = parse_args()
    runs = args.sampling_runs
    summed_metrics = {'mean_rmsd':0.0, 'median_rmsd': 0.0, 'std_rmsd': 0.0, 'rmsd_stats': [0.0,0.0,0.0], 'mean_delta_rmsd':0.0, 'std_delta_rmsd':0.0, 'median_delta_rmsd':0.0}

    for _ in range(runs):
        loader, model, engine = prepare_inference_setup(args=args)
        metrics = []
        for data in loader:
            if "conf_id" in data:
                if len(data['conf_id']) > 0:
                    conf_id = data['conf_id'][0]
                    print(f"Running inference for {conf_id}", flush=True) 

            pdb_metrics = run_inference(data, engine)
            metrics.append(pdb_metrics)
    
        if args.sampling_runs > 1:
            summed_metrics = add_statistics(args, metrics, summed_metrics)
    
    if args.sampling_runs >1:
        averaged_metrics = average_statistics(summed_metrics, runs)
        print_averaged_statistics(args, averaged_metrics)
    else:
        print_statistics(args=args, metrics=metrics)

if __name__ == "__main__":
    main()
