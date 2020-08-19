#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 17:52:45 2019

@author: akshita
"""
import os

os.system('''CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=8 python train_tfvaegan.py --encoded_noise --gzsl_od --nclass_all 101 \
--dataroot data_action --manualSeed 806 \
--syn_num 600 --preprocessing --cuda --gammaD 10 --gammaG 10 --image_embedding i3d \
--class_embedding att \
--nepoch 200 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 --dataset ucf101_i3d/split_{split} \
--image_embedding_path ucf101_i3d \
--batch_size 64 --nz 115 --attSize 115 --resSize 8192 --lr 0.0001 \
--recons_weight 0.1 --feedback_loop 2 --a2 1 --a1 1 --bce_att --feed_lr 0.00001 \
--manual_att --dec_lr 0.0001 --workers 8''')