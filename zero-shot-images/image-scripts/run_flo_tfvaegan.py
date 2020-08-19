#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 17:52:45 2019
@author: akshita
"""
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--finetuning", action='store_true', default=False)
parser.add_argument("--inductive", action='store_true', default=False)
parser.add_argument("--transductive", action='store_true', default=False)
opt = parser.parse_args()
#no encoded noise needed for flo inductive

#final model
if opt.inductive:
    os.system('''CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=8 python train_tfvaegan_inductive.py \
    --gammaD 10 --gammaG 10 --gzsl --nclass_all 102 --latent_size 1024 --manualSeed 806 \
    --syn_num 1200 --preprocessing --class_embedding att --nepoch 500 --ngh 4096 \
    --ndh 4096 --lambda1 10 --critic_iter 5 --dataset FLO --batch_size 64 --nz 1024 --attSize 1024 --resSize 2048 --lr 0.0001 \
    --classifier_lr 0.001 --cuda --image_embedding res101 --dataroot data \
    --recons_weight 0.01 --feedback_loop 2 --feed_lr 0.00001 --a1 0.5 --a2 0.5 --dec_lr 0.0001''')