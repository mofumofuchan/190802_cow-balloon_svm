#!/usr/bin/env python3
# coding: utf-8
import argparse
import os
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_rootd", required=True)
    parser.add_argument("--new_feature_rootd", required=True)
    parser.add_argument("--timestamp_list", required=True)
    args = parser.parse_args()

    with open(args.timestamp_list) as f:
        timestamp_list = [line.strip() for line in f.readlines()]

    for to_timestamp in timestamp_list:
        print("to_timestamp: {}".format(to_timestamp))

        os.makedirs(os.path.join(args.new_feature_rootd, to_timestamp, "yes"))
        os.makedirs(os.path.join(args.new_feature_rootd, to_timestamp, "no"))

        for from_timestamp in timestamp_list:
            if from_timestamp == to_timestamp:
                continue

            # yes
            fnames = os.listdir(os.path.join(args.feature_rootd,
                                             from_timestamp, "yes"))
            for fname in fnames:
                shutil.copy(os.path.join(args.feature_rootd,
                                         from_timestamp,
                                         "yes",
                                         fname),
                            os.path.join(args.new_feature_rootd,
                                         to_timestamp,
                                         "yes",
                                         fname))

            # no
            fnames = os.listdir(os.path.join(args.feature_rootd,
                                             from_timestamp, "no"))
            for fname in fnames:
                shutil.copy(os.path.join(args.feature_rootd,
                                         from_timestamp,
                                         "no",
                                         fname),
                            os.path.join(args.new_feature_rootd,
                                         to_timestamp,
                                         "no",
                                         fname))


if __name__ == "__main__":
    main()
