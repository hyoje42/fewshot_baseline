#!/usr/bin/env bash

screen -d -m -S gpu11 bash -c 'source activate torch101 && python main.py --n_epochs 5000 --gpu 1 --exp_name proto1_res18 --alg proto_1' &
screen -d -m -S gpu12 bash -c 'source activate torch101 && python main.py --n_epochs 5000 --gpu 1 --exp_name proto2_res18 --alg proto_1' &
screen -d -m -S gpu01 bash -c 'source activate torch101 && python main.py --n_epochs 5000 --gpu 0 --exp_name convex1_res18 --alg convex' &
screen -d -m -S gpu02 bash -c 'source activate torch101 && python main.py --n_epochs 5000 --gpu 0 --exp_name convex2_res18 --alg convex' &
