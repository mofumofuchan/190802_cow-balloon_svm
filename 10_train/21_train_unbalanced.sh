#!/usr/bin/env bash
# coding: utf-8

TIMESTAMPS="
172137139645586-2019-03-15-18-00
172137139645586-2019-05-11-17-00
172137139645586-2019-03-30-13-00
66999552851844-2019-05-13-11-00"

for time_stamp in $TIMESTAMPS; do
    python train_svm.py \
	   --config  00_configs/calving_unbalanced.ini \
           --time_stamp $time_stamp
done
