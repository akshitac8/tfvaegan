#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 17:52:45 2019
@author: akshita
"""
import os

os.system('''CUDA_VISIBLE_DEVICES=0 python train_tfvaegan.py --encoded_noise --gzsl_od --nclass_all 51 \
--dataroot data_action --manualSeed 806 --syn_num 1200 --preprocessing --cuda --gammaD 10 --gammaG 10 \
--image_embedding i3d --class_embedding att --nepoch 200 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 \
--dataset hmdb_i3d/split_1 --image_embedding_path hmdb_i3d --batch_size 64 --nz 300 \
--attSize 300 --resSize 8192 --lr 0.0001 \
--recons_weight 0.1 --feed_lr 0.0001 \
--feedback_loop 2 --a2 1 --a1 1 --bce_att --dec_lr 0.0001 --workers 8 ''')
