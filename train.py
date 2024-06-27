from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time

import yaml
import numpy as np

from lib.utils import load_graph_data
from model.ugcrnn_supervisor import UGCRNNSupervisor
import random
import torch
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(args):
    set_random_seed(100007 if 100007 != -1 else (int(round(time.time() * 1000)) % (2 ** 32 - 1)))
    with open(args.config_filename,'rb') as f:
        supervisor_config =yaml.load(f)

        graph_pkl_filename = supervisor_config['data'].get('graph_pkl_filename')

        station_ids, station_id_to_ind, adj_mx = load_graph_data(graph_pkl_filename)
        # print(adj_mx.shape)
        # exit()

        supervisor = UGCRNNSupervisor(adj_mx=adj_mx, **supervisor_config)
        supervisor.train()
        min_score,outputs = supervisor.evaluate('test')
        np.savez_compressed(args.output_filename, **outputs)
        # print("MAE : {}".format(mean_score))
        print("MAE_min:{}".format(min_score))
        print('Predictions saved as {}.'.format(args.output_filename))




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_filename",
                        default="./data/model/gcrnn_bj.yaml",
                        type=str,
                        help="Configuration filename for restoring the model.")
    parser.add_argument("--model_name",
                        default="ugcn",
                        type=str,
                        help="Configuration filename for restoring the model.")
    parser.add_argument("--use_cikipu_only",
                        default=False,
                        type=bool,
                        help="Set to true to only use cpu.")
    parser.add_argument('--output_filename', default='./data/prediction.npz')
    args = parser.parse_args()

    main(args)