import argparse
import time

import numpy as np
import os
import sys
import yaml

from lib.utils import load_graph_data
from model.ugcrnn_supervisor import UGCRNNSupervisor
import random
import torch
from measure_uncertainty import plot_measure_uncertainty


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_gcrnn(args):
    set_random_seed(100007 if 100007 != -1 else (int(round(time.time() * 1000)) % (2 ** 32 - 1)))
    with open(args.config_filename,'r', encoding='utf-8') as f:
        supervisor_config = yaml.load(f)

        graph_pkl_filename = supervisor_config['data'].get('graph_pkl_filename')
        station_ids, station_id_to_ind, adj_mx = load_graph_data(graph_pkl_filename)
        # print(station_ids)
        # print(station_id_to_ind)
        # print(adj_mx)
        # exit()

        supervisor = UGCRNNSupervisor(adj_mx=adj_mx, **supervisor_config)
        mean_score, outputs = supervisor.evaluate('val')
        np.savez_compressed(args.output_filename, **outputs)

        plot_measure_uncertainty(supervisor.ugcrnn_model)

        print("MAE : {}".format(mean_score))
        print('Predictions saved as {}.'.format(args.output_filename))

if __name__ == '__main__':
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cpu_only', default=False, type=str, help='Whether to run Pytorch on cpu.')
    parser.add_argument('--config_filename', default='data/model/gcrnn_jjj.yaml', type=str,
                        help='Config file for pretrained model.')
    parser.add_argument('--output_filename', default='data/test_jjj_prediction.npz')
    args = parser.parse_args()
    run_gcrnn(args)
