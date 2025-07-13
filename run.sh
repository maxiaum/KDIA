#!/bin/bash


# fedavg
python main.py --dataset=cifar10 \
--resnet_model=0 \
--alg=fedavg \
--lr=0.01 \
--epochs=10 \
--comm_round=200 \
--n_parties=100 \
--partition=noniid \
--beta=0.5 \
--logdir=logs/ \
--datadir=data/ \
--sample_fraction=0.1 \
--gpu=0 \
--init_seed=0 \
--remark=base

# fedprox
python main.py --dataset=cifar10 \
--resnet_model=0 \
--alg=fedprox \
--mu=0.01 \
--lr=0.01 \
--epochs=10 \
--comm_round=200 \
--n_parties=100 \
--partition=noniid \
--beta=0.5 \
--logdir=logs/ \
--datadir=data/ \
--sample_fraction=0.1 \
--gpu=0 \
--init_seed=0 \
--remark=base

# fedavgm
python main.py --dataset=cifar10 \
--resnet_model=0 \
--alg=fedavg \
--server_momentum=1 \
--lr=0.01 \
--epochs=10 \
--comm_round=200 \
--n_parties=100 \
--partition=noniid \
--beta=0.5 \
--logdir=logs/ \
--datadir=data/ \
--sample_fraction=0.1 \
--gpu=0 \
--init_seed=0 \
--remark=base

# moon
python main.py --dataset=cifar10 \
--resnet_model=0 \
--alg=moon \
--mu=5.0 \
--lr=0.01 \
--epochs=10 \
--comm_round=200 \
--n_parties=100 \
--partition=noniid \
--beta=0.5 \
--logdir=logs/ \
--datadir=data/ \
--sample_fraction=0.1 \
--gpu=0 \
--init_seed=0 \
--remark=base

# fedgkd
python main.py --dataset=cifar10 \
--resnet_model=0 \
--alg=fedavg \
--gkd=1 \
--mu=0.2 \
--lr=0.01 \
--epochs=10 \
--comm_round=200 \
--n_parties=100 \
--partition=noniid \
--beta=0.5 \
--logdir=logs/ \
--datadir=data/ \
--sample_fraction=0.1 \
--gpu=0 \
--init_seed=0 \
--remark=base

# feddisco
python main.py --dataset=cifar10 \
--resnet_model=0 \
--alg=fedavg \
--disco=1 \
--disco_a=0.5 \
--disco_b=0.1 \
--lr=0.01 \
--epochs=10 \
--comm_round=200 \
--n_parties=100 \
--partition=noniid \
--beta=0.5 \
--logdir=logs/ \
--datadir=data/ \
--sample_fraction=0.1 \
--gpu=0 \
--init_seed=0 \
--remark=base

# kdia
python main.py --dataset=cifar10 \
--resnet_model=0 \
--alg=kdia \
--lambda_kd=0.5 \
--lambda_gen=1.0 \
--lr=0.01 \
--epochs=10 \
--comm_round=200 \
--n_parties=100 \
--partition=noniid \
--beta=0.5 \
--logdir=logs/ \
--datadir=data/ \
--sample_fraction=0.1 \
--gpu=0 \
--init_seed=0 \
--remark=base