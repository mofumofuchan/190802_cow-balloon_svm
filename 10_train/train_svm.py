#!/usr/bin/env python3
# coding: utf-8
import argparse
import configparser
import os
from tqdm import tqdm

import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)

    neg_npy_fnames = os.listdir(config["train"]["neg_path"])
    neg_npy_list = [np.load(os.path.join(config["train"]["neg_path"], fname))
                    for fname in neg_npy_fnames]

    pos_npy_fnames = os.listdir(config["train"]["pos_path"])
    pos_npy_list = [np.load(os.path.join(config["train"]["pos_path"], fname))
                    for fname in pos_npy_fnames]
    
    X = []
    y = []

    X.extend(neg_npy_list)
    y.extend([0 for _ in neg_npy_list])

    X.extend(pos_npy_list)
    y.extend([1 for _ in pos_npy_list])

    # TODO 正規化

    random_state = int(config["DEFAULT"]["random_state"])
    X_train, X_test, \
        y_train, y_test = train_test_split(X, y,
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
    clf = GridSearchCV(
        SVC(),
        tuned_parameters,
        cv=5,
        scoring=f"{score}_weighted"
    )

    clf.fit(X_train, y_train)

    # AUC計算


    neg_npy_fnames = os.listdir(config["eval"]["neg_path"])
    neg_npy_list = [np.load(os.path.join(config["eval"]["neg_path"], fname))
                    for fname in neg_npy_fnames]

    pos_npy_fnames = os.listdir(config["eval"]["pos_path"])
    pos_npy_list = [np.load(os.path.join(config["eval"]["pos_path"], fname))
                    for fname in pos_npy_fnames]


    # 正規化

    # AUC計算


if __name__ == "__main__":
    main()
