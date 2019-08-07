#!/usr/bin/env bash
set -eu

FEATURE_ROOTD="../../180305_cow_single-task_chainer_balloon/21_evaluation2/03_new-posteriors/190731_global-model-v0_hip_only-balloon_with-aug_14-7-3/model_epoch_22"

ORIGINAL_DATASET_NAME="190807_selected-kubota_annotated_aggregated"
NEW_DATASET_NAME="190807_selected-kubota_annotated_aggregated_balanced"


TIMESTAMP_STRS="
172137139645586-2019-03-15-18-00
172137139645586-2019-03-30-13-00
172137139645586-2019-05-11-17-00
66999552851844-2019-05-13-11-00
"

RATE="1.5"


###
for TIMESTAMP_STR in $TIMESTAMP_STRS; do
    OUT_DIR=$FEATURE_ROOTD/$NEW_DATASET_NAME/rate-${RATE}/$TIMESTAMP_STR
    mkdir -p $OUT_DIR/yes
    mkdir -p $OUT_DIR/no

    python select_negative_feature_with_clustering.py \
	   --yes_dir $FEATURE_ROOTD/$ORIGINAL_DATASET_NAME/$TIMESTAMP_STR/yes \
	   --no_dir $FEATURE_ROOTD/$ORIGINAL_DATASET_NAME/$TIMESTAMP_STR/no \
	   --rate $RATE \
	   --out_dir $OUT_DIR
done
