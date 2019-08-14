#!/usr/bin/env python3
# coding: utf-8
import argparse
import configparser
import os
from tqdm import tqdm

import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--time_stamp", required=True)
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)

    print(args.time_stamp)

    neg_npy_fnames = os.listdir(os.path.join(config["train"]["neg_rootpath"],
                                             args.time_stamp,
                                             "no"))
    neg_npy_list = [np.load(os.path.join(config["train"]["neg_rootpath"],
                                         args.time_stamp,
                                         "no",
                                         fname))
                    for fname in neg_npy_fnames]

    pos_npy_fnames = os.listdir(os.path.join(config["train"]["neg_rootpath"],
                                             args.time_stamp,
                                             "yes"))
    pos_npy_list = [np.load(os.path.join(config["train"]["pos_rootpath"],
                                         args.time_stamp,
                                         "yes",
                                         fname))
                    for fname in pos_npy_fnames]

    X = []
    y = []

    X.extend(neg_npy_list)
    y.extend([0 for _ in neg_npy_list])

    X.extend(pos_npy_list)
    y.extend([1 for _ in pos_npy_list])

    #  正規化
    scaler = StandardScaler()
    scaler.fit(X)
    X_normed = scaler.transform(X)

    random_state = int(config["DEFAULT"]["random_state"])
    X_train, X_test, \
        y_train, y_test = train_test_split(X_normed, y,
                                           test_size=0.1,
                                           random_state=random_state)
    tuned_parameters = [
        {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        {'C': [1, 10, 100, 1000], 'kernel': ['rbf'],
         'gamma': [0.001, 0.0001]},
        {'C': [1, 10, 100, 1000], 'kernel': ['poly'],
         'degree': [2, 3, 4],
         'gamma': [0.001, 0.0001]},
        {'C': [1, 10, 100, 1000], 'kernel': ['sigmoid'],
         'gamma': [0.001, 0.0001]}
    ]
    score = "f1"
    gscv = GridSearchCV(
        SVC(),
        tuned_parameters,
        cv=5,
        scoring=f"{score}_weighted",
        verbose=1
    )

    gscv.fit(X_train, y_train)
    clf = gscv.best_estimator_

    # AUC計算

    neg_npy_fnames = os.listdir(os.path.join(config["train"]["neg_rootpath"],
                                             args.time_stamp,
                                             "no"))
    neg_npy_list = [np.load(os.path.join(config["train"]["neg_rootpath"],
                                         args.time_stamp,
                                         "no",
                                         fname))
                    for fname in neg_npy_fnames]

    pos_npy_fnames = os.listdir(os.path.join(config["train"]["neg_rootpath"],
                                             args.time_stamp,
                                             "yes"))
    pos_npy_list = [np.load(os.path.join(config["train"]["neg_rootpath"],
                                         args.time_stamp,
                                         "yes",
                                         fname))
                    for fname in pos_npy_fnames]

    X = []
    y = []

    X.extend(neg_npy_list)
    y.extend([0 for _ in neg_npy_list])

    X.extend(pos_npy_list)
    y.extend([1 for _ in pos_npy_list])

    #  正規化
    scaler.fit(X)
    X_normed = scaler.transform(X)

    # AUC計算
    y_pred = clf.predict(X_normed)

    auc = roc_auc_score(y, y_pred)

    with open(f"{args.config}_{args.time_stamp}_score.txt",
              mode="w") as f:
        f.write(f"[{args.time_stamp}] auc: {auc}")


if __name__ == "__main__":
    main()
