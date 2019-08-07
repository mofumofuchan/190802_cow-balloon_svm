#!/usr/bin/env bash 
# coding: utf-8
set -eu

FEATURE_ROOTD="../../180305_cow_single-task_chainer_balloon/21_evaluation2/03_new-posteriors/190731_global-model-v0_hip_only-balloon_with-aug_14-7-3/model_epoch_22/190730_selected-kubota_annotated"

NEW_FEATURE_ROOTD="../../180305_cow_single-task_chainer_balloon/21_evaluation2/03_new-posteriors/190731_global-model-v0_hip_only-balloon_with-aug_14-7-3/model_epoch_22/190807_selected-kubota_annotated_aggregated"

TIMESTAMP_LIST_PATH="./calving_time.list"

###
mkdir -p $NEW_FEATURE_ROOTD

python aggregate_features.py \
    --feature_rootd $FEATURE_ROOTD \
    --new_feature_rootd $NEW_FEATURE_ROOTD \
    --timestamp_list $TIMESTAMP_LIST_PATH

