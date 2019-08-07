#!/usr/bin/env python3
# coding: utf-8
import argparse
import os
import time
import shutil

import numpy as np
from sklearn.cluster import KMeans


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yes_dir", required=True)
    parser.add_argument("--no_dir", required=True)
    parser.add_argument("--rate", type=float, required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    print(f"# yes_dir: {args.yes_dir}")
    print(f"# no_dir: {args.no_dir}")
    print(f"# rate: {args.rate}")

    yes_npy_fnames = os.listdir(args.yes_dir)
    num_select_no_samples = int(len(yes_npy_fnames) * args.rate)
    print(f"# select no samples: {num_select_no_samples}")

    no_npy_fnames = os.listdir(args.no_dir)

    features = [(fname, np.load(os.path.join(args.no_dir, fname)))
                for fname in no_npy_fnames]

    start_time = time.time()
    print("start clustering..")
    pred = KMeans(n_clusters=num_select_no_samples).\
                              fit_predict([item[1] for item in features])

    elapsed_time = time.time() - start_time
    print("clustering finished. elapsed time: {}s".format(elapsed_time))

    selected_cluster_nums = []
    selected_fnames = []
    for num_cluster, (fname, _) in zip(pred, features):
        if num_cluster not in selected_cluster_nums:
            selected_cluster_nums.append(num_cluster)
            selected_fnames.append(fname)

    # yes
    for npy_fname in yes_npy_fnames:
        shutil.copy(os.path.join(args.yes_dir, npy_fname),
                    os.path.join(args.out_dir, "yes", npy_fname))

    # no
    for npy_fname in selected_fnames:
        shutil.copy(os.path.join(args.no_dir, npy_fname),
                    os.path.join(args.out_dir, "no", npy_fname))


if __name__ == "__main__":
    main()
