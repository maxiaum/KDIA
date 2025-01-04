import numpy as np
import json
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import logging
import os
import copy
import datetime
import random
import math

from model import *
from utils import *
from disco import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--measure_difference', type=str, default='kl', help='how to measure difference. e.g. only_iid, cosine')
    parser.add_argument('--gkd', type=int, default=0, help='')
    parser.add_argument('--disco', type=int, default=0, help='w/o or w feddisco')
    parser.add_argument('--disco_a', type=float, default=0.5, help='')
    parser.add_argument('--disco_b', type=float, default=0.1, help='')
    parser.add_argument('--remark', type=str, default='base', help='')
    parser.add_argument('--noise_dim', type=int, default=100, help=' ')
    parser.add_argument('--feature_num', type=int, default=16, help=' ')
    parser.add_argument('--num_classes', type=int, default=10, help=' ')
    parser.add_argument('--gen_bs', type=int, default=64, help='the training batch size of generator')
    parser.add_argument('--global_gen_epoch', type=int, default=20, help=' ')
    parser.add_argument('--global_iter_per_epoch', type=int, default=100, help='')
    parser.add_argument('--gen_lr', type=float, default=1e-3, help='')
    parser.add_argument('--noise_std', type=float, default=1, help='')
    parser.add_argument('--alpha', type=float, default=0.5, help='')
    parser.add_argument('--algorithm', type=str, default='fedavg', choices={'fedavg','fedprox','fedcurv','moon','mine'}, help=' ')
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use')
    parser.add_argument('--model', type=str, default='resnet50', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='cifar100', help='dataset used for training')
    parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))
    parser.add_argument('--partition', type=str, default='homo', help='the data partitioning strategy')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate (default: 0.1)')
    parser.add_argument('--epochs', type=int, default=5, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=2, help='number of workers in a distributed cluster')
    parser.add_argument('--alg', type=str, default='fedavg',
                        help='communication strategy: fedavg/fedprox')
    parser.add_argument('--comm_round', type=int, default=50, help='number of maximum communication roun')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="./models/", help='Model directory path')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--mu', type=float, default=5, help='the mu parameter for fedprox or moon')
    parser.add_argument('--out_dim', type=int, default=256, help='the output dimension for the projection layer')
    parser.add_argument('--temperature', type=float, default=0.5, help='the temperature parameter for contrastive loss')
    parser.add_argument('--temper', type=float, default=2.0, help='')
    parser.add_argument('--local_max_epoch', type=int, default=100,
                        help='the number of epoch for local optimal training')
    parser.add_argument('--model_buffer_size', type=int, default=1,
                        help='store how many previous models for contrastive loss')
    parser.add_argument('--pool_option', type=str, default='FIFO', help='FIFO or BOX')
    parser.add_argument('--sample_fraction', type=float, default=1.0, help='how many clients are sampled in each round')
    parser.add_argument('--load_model_file', type=str, default=None, help='the model to load as global model')
    parser.add_argument('--load_pool_file', type=str, default=None, help='the old model pool path to load')
    parser.add_argument('--load_model_round', type=int, default=None,
                        help='how many rounds have executed for the loaded model')
    parser.add_argument('--load_first_net', type=int, default=1, help='whether load the first net as old net or not')
    parser.add_argument('--resnet_model', type=int, default=0, help='use normal model or aggregate model')
    parser.add_argument('--loss', type=str, default='contrastive')
    parser.add_argument('--save_model', type=int, default=0)
    parser.add_argument('--use_project_head', type=int, default=1)
    parser.add_argument('--server_momentum', type=float, default=0, help='the server momentum (FedAvgM)')
    parser.add_argument('--step', type=int, default=3, help='')
    args = parser.parse_args()
    return args



def init_nets(net_configs, n_parties, args, device='cpu'):
    nets = {net_i: None for net_i in range(n_parties)}
    if args.dataset in {'mnist', 'cifar10', 'svhn', 'fmnist', 'cinic10'}:
        n_classes = 10
    elif args.dataset == 'celeba':
        n_classes = 2
    elif args.dataset == 'cifar100':
        n_classes = 100
    elif args.dataset == 'tinyimagenet':
        n_classes = 200
    elif args.dataset == 'femnist':
        n_classes = 26
    elif args.dataset == 'emnist':
        n_classes = 47
    elif args.dataset == 'xray':
        n_classes = 2
    if args.dataset in {'cifar10', 'cifar100', 'tinyimagenet', 'cinic10'}:
        input_channel = 3
    if args.resnet_model:
        for net_i in range(n_parties):
            if args.alg == 'mine':
                net = nn.ModuleDict()
                net['ec'] = ResNet18_imagenet(input_dim=512, hidden_dims=[400,400,256], output_dim=n_classes)
                net['gen'] = Generator(args.noise_dim, args.feature_num, n_classes)
                frozen_net(net, ['ec', 'gen'], frozen=True)
            else:
                net = ResNet18_imagenet(input_dim=512, hidden_dims=[400,400,256], output_dim=n_classes)
            if device == 'cpu':
                net.to(device)
            else:
                net = net.to(device)
            nets[net_i] = net
    else:
        for net_i in range(n_parties):
            if args.alg == 'mine':
                net = nn.ModuleDict()
                # net = ModelFedCon(args.model, args.out_dim, n_classes, net_configs)
                net['ec'] = ModelFedMine(input_channel=input_channel, input_dim=400, hidden_dims=[120, 84, 84, 256], output_dim=n_classes)
                net['gen'] = Generator(args.noise_dim, args.feature_num, n_classes)
                frozen_net(net, ['ec', 'gen'], frozen=True)
            else:
                net = ModelFedMine(input_channel=input_channel, input_dim=400, hidden_dims=[120, 84, 84, 256], output_dim=n_classes)
            if device == 'cpu':
                net.to(device)
            else:
                net = net.to(device)
            nets[net_i] = net

    model_meta_data = []
    layer_type = []
    for (k, v) in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)

    return nets, model_meta_data, layer_type

def frozen_net(model, models_name, frozen=False):
    for name in models_name:
        for param in model[name].parameters():
            param.requires_grad = not frozen
        if frozen:
            model[name].eval()
        else:
            model[name].train()

def get_params(model, models_name):
    params = []
    for name in models_name:
        params.append({"params": model[name].parameters()})
    return params
    
def ema_update_locals(w1, w2):
    w = copy.deepcopy(w1)
    for key in w1.keys():
        w[key] = args.alpha * w1[key] + (1 - args.alpha) * w2[key]
    return w

def fedavging(weight_buffer, freqs):
    # print(weight_buffer)
    w = None
    flag = True
    for net_id, net in enumerate(weight_buffer.values()):
        net_para = net['ec'].state_dict()
        if flag:
            w = net['ec'].state_dict()
            flag = False
        if net_id == 0:
            for key in net_para:
                w[key] = net_para[key] * freqs[net_id]
        else:
            for key in net_para:
                w[key] += net_para[key] * freqs[net_id]
    
    return w

def train_net(net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, args, t_net=None, round=None, device="cpu"):
    # if len(args.gpu.split(',')) > 1:
    #     net = nn.DataParallel(net, device_ids=[0, 1])
    
    net.to(device)
    if t_net is None or round < 5:
        t_net.load_state_dict(net.state_dict())
    t_net.to(device)
    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))

    # train_acc, _ = compute_accuracy(net, train_dataloader, device=device, algo='fedavg')

    # test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device, algo='fedavg')

    # logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    # logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().to(device)
    kl_div = nn.KLDivLoss(reduction='batchmean').to(device)

    cnt = 0

    for epoch in range(epochs):
        epoch_loss_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device)

            optimizer.zero_grad()
            x.requires_grad = False
            target.requires_grad = False
            target = target.long()
            _, _, out = net(x)
            loss = criterion(out, target)
            if args.gkd:
                sharpen = F.softmax(out / args.temper, dim=1)
                with torch.no_grad():
                    _, _, output = t_net(x)
                    soft_target = F.softmax(output / args.temper, dim=1) 
                reg_loss = (args.mu / 2) * criterion(sharpen, soft_target)
                
                loss += reg_loss

            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))


    # train_acc, _ = compute_accuracy(net, train_dataloader, device=device, algo='fedavg')
    test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device, algo='fedavg')

    # logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)
    net.to('cpu')

    logger.info(' ** Training complete **')
    return test_acc


def train_net_fedprox(net_id, net, global_net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, mu, args,
                      device="cpu"):
    # global_net.to(device)
    # if len(args.gpu.split(',')) > 1:
    #     net = nn.DataParallel(net, device_ids=[0, 1])
    net.to(device)
    # else:
    #     net.to(device)
    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))

    # train_acc, _ = compute_accuracy(net, train_dataloader, device=device, algo='fedprox')
    # test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device, algo='fedprox')

    # logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    # logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)

    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    global_weight_collector = list(global_net.to(device).parameters())

    for epoch in range(epochs):
        epoch_loss_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device)

            optimizer.zero_grad()
            x.requires_grad = False
            target.requires_grad = False
            target = target.long()

            _, _, out = net(x)
            loss = criterion(out, target)

            # for fedprox
            fed_prox_reg = 0.0
            # fed_prox_reg += np.linalg.norm([i - j for i, j in zip(global_weight_collector, get_trainable_parameters(net).tolist())], ord=2)
            for param_index, param in enumerate(net.parameters()):
                fed_prox_reg += ((mu / 2) * torch.norm((param - global_weight_collector[param_index])) ** 2)
            loss += fed_prox_reg

            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    # train_acc, _ = compute_accuracy(net, train_dataloader, device=device, algo='fedprox')
    test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device, algo='fedprox')

    # logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)
    net.to('cpu')
    logger.info(' ** Training complete **')
    return test_acc


def train_net_fedmine(net_id, net, t_net, train_dataloader, test_dataloader, round, lr, args_optimizer, mu,
                      args,
                      device="cpu"):
    epochs, rounds = args.epochs, args.comm_round
    # global_net.to(device)
    # if len(args.gpu.split(',')) > 1:
    #     net = nn.DataParallel(net, device_ids=[0,1])
    net.to(device)
    frozen_net(net, ['ec'], frozen=False)
    frozen_net(net, ['gen'], frozen=True)
    frozen_net(t_net, ['ec'], frozen=True)

    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))


    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)

    criterion = nn.CrossEntropyLoss().to(device)
    kl_div = nn.KLDivLoss(reduction='batchmean').to(device)

    N = 0
    eps = 1e-4
    f = False
 
    for batch_idx, (x, target) in enumerate(train_dataloader):
        N += x.size(0)
            
    M = N
    if N <= args.batch_size:
        N = N * epochs
        f = True
    y_ = torch.randint(0, args.num_classes, (N,))
    
    print("*"*20, M)

    for epoch in range(epochs):
        epoch_loss_collector = []
        if f:
            y = y_[epoch*M:(epoch+1)*M].to(device)
            # print(y_.shape)
        else:
            counter = 0
            indices = torch.randperm(N)
            y_ = y_[indices].to(device)
        print(y_.shape)
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device)
            
            optimizer.zero_grad()
            x.requires_grad = False
            target.requires_grad = False
            target = target.long()

            _, _, output = net['ec'](x, withgen=False)
            sharpen = F.softmax(output / args.temper, dim=1)
            
            with torch.no_grad():
                _, _, t_output = t_net['ec'](x, withgen=False)
                soft_target = F.log_softmax(t_output / args.temper, dim=1)
            
            if not f:
                y = y_[counter:counter+x.size(0)]
                counter += x.size(0)

            z = torch.randn(x.size(0), args.noise_dim, 1, 1).to(device)
            gc = net['gen'](z, y).detach()
            _, _, output_ = net['ec'](gc, withgen=True)
            
            loss = criterion(output, target)
            reg_loss = args.mu * kl_div(soft_target, sharpen)
            gen_loss = 1.0 * criterion(output_, y)

            # for fedmine
            loss += (reg_loss + gen_loss)

            loss.backward()
            optimizer.step()

            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))
    frozen_net(net, ['ec'], frozen=True)
    
    # train_acc, _ = compute_accuracy(net, train_dataloader, device=device, algo='mine')
    test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device, algo='mine')

    # logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)
    net.to('cpu')
    logger.info(' ** Training complete **')
    return test_acc


def train_net_fedcon(net_id, net, global_net, previous_nets, train_dataloader, test_dataloader, epochs, lr,
                     args_optimizer, mu, temperature, args,
                     round, device="cpu"):
    # if len(args.gpu.split(',')) > 1:
    #     net = nn.DataParallel(net, device_ids=[0, 1])
    net.to(device)
    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))

    # train_acc, _ = compute_accuracy(net, train_dataloader, device=device, algo='moon')

    # test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device, algo='moon')

    # logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    # logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)

    criterion = nn.CrossEntropyLoss().to(device)
    # global_net.to(device)

    for previous_net in previous_nets:
        previous_net.to(device)
    global_w = global_net.state_dict()

    cnt = 0
    cos = torch.nn.CosineSimilarity(dim=-1)
    # mu = 0.001

    for epoch in range(epochs):
        epoch_loss_collector = []
        epoch_loss1_collector = []
        epoch_loss2_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device)

            optimizer.zero_grad()
            x.requires_grad = False
            target.requires_grad = False
            target = target.long()

            _, pro1, out = net(x)
            _, pro2, _ = global_net(x)

            posi = cos(pro1, pro2)
            logits = posi.reshape(-1, 1)

            for previous_net in previous_nets:
                previous_net.to(device)
                _, pro3, _ = previous_net(x)
                nega = cos(pro1, pro3)
                logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)

                previous_net.to('cpu')

            logits /= temperature
            labels = torch.zeros(x.size(0)).to(device).long()

            loss2 = mu * criterion(logits, labels)

            loss1 = criterion(out, target)
            loss = loss1 + loss2

            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())
            epoch_loss1_collector.append(loss1.item())
            epoch_loss2_collector.append(loss2.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        epoch_loss1 = sum(epoch_loss1_collector) / len(epoch_loss1_collector)
        epoch_loss2 = sum(epoch_loss2_collector) / len(epoch_loss2_collector)
        logger.info('Epoch: %d Loss: %f Loss1: %f Loss2: %f' % (epoch, epoch_loss, epoch_loss1, epoch_loss2))

    for previous_net in previous_nets:
        previous_net.to('cpu')
    # train_acc, _ = compute_accuracy(net, train_dataloader, device=device, algo='moon')
    test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device, algo='moon')

    # logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)
    net.to('cpu')
    logger.info(' ** Training complete **')
    return test_acc

def train_net_scaffold(net_id, net, global_model, c_local, c_global, train_dataloader, test_dataloader, epochs, lr,
                     args_optimizer, args, device="cpu"):
    if len(args.gpu.split(',')) > 1:
        net = nn.DataParallel(net, device_ids=[0,1])
    
    net.to(device)
    c_local.to(device)
    c_global.to(device)
    old_net_para = net.state_dict()
    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))
    
    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)
        
    criterion = nn.CrossEntropyLoss().to(device)
    
    cnt = 0
    c_global_para = c_global.state_dict()
    c_local_para = c_local.state_dict()

    for epoch in range(epochs):
        epoch_loss_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.cuda(), target.cuda()
            optimizer.zero_grad()
            x.requires_grad = False
            target.requires_grad = False
            target = target.long()

            _,_,out = net(x)
            loss = criterion(out, target)

            loss.backward()
            optimizer.step()

            net_para = net.state_dict()
            for name, param in net.named_parameters():
                if param.grad is not None:
                    g = param.grad
                    net_para[name] = net_para[name] - args.lr * (g + c_global_para[name] - c_local_para[name])
                    
            net.load_state_dict(net_para)
            cnt += 1
            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))
        
    test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device, algo='scaffold')
    logger.info('>> Test accuracy: %f' % test_acc)

    c_new_para = c_local.state_dict()
    c_delta_para = copy.deepcopy(c_local.state_dict())
    # global_model_para = global_model.state_dict()
    for key in net_para:
        c_new_para[key] = c_new_para[key] - c_global_para[key] + (old_net_para[key] - net_para[key]) / (cnt * args.lr)
        c_delta_para[key] = c_new_para[key] - c_local_para[key]
        net_para[key] = net_para[key] - old_net_para[key]
    c_local.load_state_dict(c_new_para)

    net.to('cpu')
    c_local.to('cpu')
    c_global.to('cpu')
    return test_acc, c_delta_para, net_para

def local_train_net(nets, args, net_dataidx_map, train_dl=None, test_dl=None, global_model=None, teacher_model=None,
                    prev_model_pool=None, server_c=None, clients_c=None, c_local=None, c_global=None, round=None, device="cpu"):
    avg_acc = 0.0
    acc_list = []
    if args.alg == 'scaffold':
        c_total_delta = global_model.state_dict()
        net_total_delta = global_model.state_dict()
        net_global_para = global_model.state_dict()
        for key in c_total_delta:
            c_total_delta[key] = 0.0
            net_total_delta[key] = 0.0
    if global_model:
        global_model.to(device)
    if server_c:
        server_c.to(device)
        server_c_collector = list(server_c.to(device).parameters())
        new_server_c_collector = copy.deepcopy(server_c_collector)
    # local_distribution = torch.empty([len(nets), args.num_classes])
    for idx, (net_id, net) in enumerate(nets.items()):
        dataidxs = net_dataidx_map[net_id]
        # print(len(dataidxs))
        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, beta=args.beta)
        train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        n_epoch = args.epochs
        # print(train_dl_local)
        if args.alg == 'fedavg':
            testacc = train_net(net_id, net, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, args,
                                          t_net=teacher_model, round=round, device=device)
        elif args.alg == 'fedprox':
            testacc = train_net_fedprox(net_id, net, global_model, train_dl_local, test_dl, n_epoch, args.lr,
                                                  args.optimizer, args.mu, args, device=device)
        elif args.alg == 'moon':
            prev_models = []
            for i in range(len(prev_model_pool)):
                prev_models.append(prev_model_pool[i][net_id])
            testacc = train_net_fedcon(net_id, net, global_model, prev_models, train_dl_local, test_dl,
                                                 n_epoch, args.lr,
                                                 args.optimizer, args.mu, args.temperature, args, round, device=device)

        elif args.alg == 'local_training':
            testacc = train_net(net_id, net, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, args,
                                          device=device)
        elif args.alg == 'mine':
            testacc = train_net_fedmine(net_id, net, teacher_model, train_dl_local, test_dl, round, args.lr,
                                                  args.optimizer, args.mu, args, device=device)
        elif args.alg == 'scaffold':
            testacc, c_delta_para, net_delta_para = train_net_scaffold(net_id, net, global_model, c_local[net_id], c_global, train_dl_local, test_dl, n_epoch, args.lr, 
                                                  args.optimizer, args, device=device)
            for key in c_total_delta:  
                c_delta_para[key] = c_delta_para[key].to('cpu') 
                net_delta_para[key] = net_delta_para[key].to('cpu')
                c_total_delta[key] += c_delta_para[key] / args.n_parties
                net_total_delta[key] += net_delta_para[key] / len(nets)

        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc
        acc_list.append(testacc)
    avg_acc /= args.n_parties
    
    if args.alg == 'scaffold':
        c_global_para = c_global.state_dict()
        for key in c_global_para:
            if c_global_para[key].type() == 'torch.LongTensor':
                c_global_para[key] += c_total_delta[key].type(torch.LongTensor)
                net_global_para[key] += net_total_delta[key].type(torch.LongTensor)
            elif c_global_para[key].type() == 'torch.cuda.LongTensor':
                c_global_para[key] += c_total_delta[key].type(torch.cuda.LongTensor)
                net_global_para[key] += net_total_delta[key].type(torch.cuda.LongTensor)
            else:
                c_global_para[key] += c_total_delta[key]
                net_global_para[key] += net_total_delta[key]
                
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)
        logger.info("std acc %f" % np.std(acc_list))
    if global_model:
        global_model.to('cpu')
    if server_c:
        for param_index, param in enumerate(server_c.parameters()):
            server_c_collector[param_index] = new_server_c_collector[param_index]
        server_c.to('cpu')
    return nets if args.alg != 'scaffold' else (nets, c_global_para, net_global_para)


if __name__ == '__main__':
    args = get_args()
    mkdirs(args.logdir)
    mkdirs(args.modeldir)
    if args.log_file_name is None:
        argument_path = 'experiment_arguments-{}_set={}_beta={}_client={}_frac={}_remark:{}.json'.format(args.alg, args.dataset, args.beta, args.n_parties, args.sample_fraction, args.remark)
    else:
        argument_path = args.log_file_name + '.json'
    with open(os.path.join(args.logdir, argument_path), 'w') as f:
        json.dump(str(args), f)
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu') 
    # print('*****************', device)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    if args.log_file_name is None:
        args.log_file_name = 'experiment_log-{}_set={}_beta={}_client={}_frac={}_remark:{}'.format(args.alg, args.dataset, args.beta, args.n_parties, args.sample_fraction, args.remark)
    log_path = args.log_file_name + '.log'
    logging.basicConfig(
        filename=os.path.join(args.logdir, log_path),
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.info(device)

    seed = args.init_seed
    logger.info("#" * 100)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False

    logger.info("Partitioning data")
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        args.dataset, args.datadir, args.logdir, args.partition, args.n_parties, beta=args.beta)
    
    global_dist = np.ones(args.num_classes) / args.num_classes
    all_data_points = sum([len(net_dataidx_map[r]) for r in range(args.n_parties)])
    all_freqs = [len(net_dataidx_map[r]) / all_data_points for r in range(args.n_parties)]
    n_party_per_round = int(args.n_parties * args.sample_fraction)
    party_list = [i for i in range(args.n_parties)]
    party_list_rounds = []
    if n_party_per_round != args.n_parties:
        for i in range(args.comm_round):
            # new_list = [idx for idx in party_list if idx != (i / args.n_parties) % args.n_parties]
            # party_list_rounds.append(new_list)
            party_list_rounds.append(random.sample(party_list, n_party_per_round))
    else:
        for i in range(args.comm_round):
            party_list_rounds.append(party_list)

    n_classes = len(np.unique(y_train))

    train_dl_global, test_dl, train_ds_global, test_ds_global = get_dataloader(args.dataset,
                                                                               args.datadir,
                                                                               args.batch_size,
                                                                               32)

    print("len train_dl_global:", len(train_ds_global))
    train_dl = None
    data_size = len(test_ds_global)

    logger.info("Initializing nets")
    nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.n_parties, args, device='cpu')
    
    global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 1, args, device='cpu')
    global_model = global_models[0]
    n_comm_rounds = args.comm_round
    if args.load_model_file and args.alg != 'plot_visual':
        global_model.load_state_dict(torch.load(args.load_model_file))
        n_comm_rounds -= args.load_model_round
    # print(global_model.device)
    if args.server_momentum != 0:
        if args.alg == 'mine':
            moment_v = copy.deepcopy(global_model['ec'].state_dict())
        else:
            moment_v = copy.deepcopy(global_model.state_dict())
        for key in moment_v:
            moment_v[key] = 0
    if args.alg == 'moon':
        old_nets_pool = []
        if args.load_pool_file:
            for nets_id in range(args.model_buffer_size):
                old_nets, _, _ = init_nets(args.net_config, args.n_parties, args, device='cpu')
                checkpoint = torch.load(args.load_pool_file)
                for net_id, net in old_nets.items():
                    net.load_state_dict(checkpoint['pool' + str(nets_id) + '_' + 'net' + str(net_id)])
                old_nets_pool.append(old_nets)
        elif args.load_first_net:
            if len(old_nets_pool) < args.model_buffer_size:
                old_nets = copy.deepcopy(nets)
                for _, net in old_nets.items():
                    net.eval()
                    for param in net.parameters():
                        param.requires_grad = False

        for round in range(n_comm_rounds):
            logger.info("in comm round:" + str(round))
            party_list_this_round = party_list_rounds[round]

            global_model.eval()
            for param in global_model.parameters():
                param.requires_grad = False
            global_w = global_model.state_dict()

            if args.server_momentum:
                old_w = copy.deepcopy(global_model.state_dict())

            nets_this_round = {k: nets[k] for k in party_list_this_round}
            for net in nets_this_round.values():
                net.load_state_dict(global_w)

            local_train_net(nets_this_round, args, net_dataidx_map, train_dl=train_dl, test_dl=test_dl,
                            global_model=global_model, prev_model_pool=old_nets_pool, round=round, device=device)

            total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]

            for net_id, net in enumerate(nets_this_round.values()):
                net_para = net.state_dict()
                if net_id == 0:
                    for key in net_para:
                        global_w[key] = net_para[key] * fed_avg_freqs[net_id]
                else:
                    for key in net_para:
                        global_w[key] += net_para[key] * fed_avg_freqs[net_id]

            if args.server_momentum:
                delta_w = copy.deepcopy(global_w)
                for key in delta_w:
                    delta_w[key] = old_w[key] - global_w[key]
                    moment_v[key] = args.server_momentum * moment_v[key] + (1 - args.server_momentum) * delta_w[key]
                    global_w[key] = old_w[key] - moment_v[key]

            global_model.load_state_dict(global_w)
            # summary(global_model.to(device), (3, 32, 32))

            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl))
            global_model.to(device)
            # train_acc, train_loss = compute_accuracy(global_model, train_dl_global, device=device, algo='moon')
            test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device, algo='moon')
            global_model.to('cpu')
            # logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)
            # logger.info('>> Global Model Train loss: %f' % train_loss)

            if len(old_nets_pool) < args.model_buffer_size:
                old_nets = copy.deepcopy(nets)
                for _, net in old_nets.items():
                    net.eval()
                    for param in net.parameters():
                        param.requires_grad = False
                old_nets_pool.append(old_nets)
            elif args.pool_option == 'FIFO':
                old_nets = copy.deepcopy(nets)
                for _, net in old_nets.items():
                    net.eval()
                    for param in net.parameters():
                        param.requires_grad = False
                for i in range(args.model_buffer_size - 2, -1, -1):
                    old_nets_pool[i] = old_nets_pool[i + 1]
                old_nets_pool[args.model_buffer_size - 1] = old_nets

            mkdirs(args.modeldir + 'fedcon/')
            if args.save_model:
                torch.save(global_model.state_dict(),
                           args.modeldir + 'fedcon/global_model_' + args.log_file_name + '.pth')
                torch.save(nets[0].state_dict(), args.modeldir + 'fedcon/localmodel0' + args.log_file_name + '.pth')
                for nets_id, old_nets in enumerate(old_nets_pool):
                    torch.save({'pool' + str(nets_id) + '_' + 'net' + str(net_id): net.state_dict() for net_id, net in
                                old_nets.items()},
                               args.modeldir + 'fedcon/prev_model_pool_' + args.log_file_name + '.pth')


    elif args.alg == 'fedavg':
        if args.gkd:
            weight_buffer = [copy.deepcopy(global_model) for _ in range(5)]
            freq = 1 / len(weight_buffer)
        t_net = copy.deepcopy(global_model)
        for round in range(n_comm_rounds):
            logger.info("in comm round:" + str(round))
            party_list_this_round = party_list_rounds[round]

            global_w = global_model.state_dict()
            if args.server_momentum:
                old_w = copy.deepcopy(global_model.state_dict())

            nets_this_round = {k: nets[k] for k in party_list_this_round}
            for net in nets_this_round.values():
                net.load_state_dict(global_w)

            local_train_net(nets_this_round, args, net_dataidx_map, teacher_model=copy.deepcopy(t_net), train_dl=train_dl, test_dl=test_dl, device=device, round=round)

            total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]
            # print(traindata_cls_counts)
            if args.disco:
                distribution_difference = get_distribution_difference(traindata_cls_counts, participation_clients=party_list_this_round, metric=args.measure_difference, hypo_distribution=global_dist)
                fed_avg_freqs = disco_weight_adjusting(fed_avg_freqs, distribution_difference, args.disco_a, args.disco_b)
                if round==0 or args.sample_fraction<1.0:
                    print(f'Distribution_difference : {distribution_difference}\nDisco Aggregation Weights : {fed_avg_freqs}')
                
            for net_id, net in enumerate(nets_this_round.values()):
                net_para = net.state_dict()
                if net_id == 0:
                    for key in net_para:
                        global_w[key] = net_para[key] * fed_avg_freqs[net_id]
                else:
                    for key in net_para:
                        global_w[key] += net_para[key] * fed_avg_freqs[net_id]

            if args.server_momentum:
                delta_w = copy.deepcopy(global_w)
                for key in delta_w:
                    delta_w[key] = old_w[key] - global_w[key]
                    moment_v[key] = args.server_momentum * moment_v[key] + (1 - args.server_momentum) * delta_w[key]
                    global_w[key] = old_w[key] - moment_v[key]

            global_model.load_state_dict(global_w)
            if args.gkd:
                weight = copy.deepcopy(global_w)
                weight_buffer[round%5].load_state_dict(global_w)
                for net_id, net in enumerate(weight_buffer):
                    net_para = net.state_dict()
                    if net_id == 0:
                        for key in net_para:
                            weight[key] = net_para[key] * freq
                    else:
                        for key in net_para:
                            weight[key] += net_para[key] * freq
                t_net.load_state_dict(weight)
                
            logger.info('global n_test: %d' % len(test_dl))
            global_model.to(device)
            # train_acc, train_loss = compute_accuracy(global_model, train_dl_global, device=device, algo='fedavg')
            test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device, algo='fedavg')  

            logger.info('>> Global Model Test accuracy: %f' % test_acc)
            mkdirs(args.modeldir + 'fedavg/')
            global_model.to('cpu')

            torch.save(global_model.state_dict(),
                       args.modeldir + 'fedavg/' + 'globalmodel' + args.log_file_name + '.pth')
            torch.save(nets[0].state_dict(), args.modeldir + 'fedavg/' + 'localmodel0' + args.log_file_name + '.pth')
    elif args.alg == 'fedprox':

        for round in range(n_comm_rounds):
            logger.info("in comm round:" + str(round))
            party_list_this_round = party_list_rounds[round]
            global_w = global_model.state_dict()
            nets_this_round = {k: nets[k] for k in party_list_this_round}
            for net in nets_this_round.values():
                net.load_state_dict(global_w)

            local_train_net(nets_this_round, args, net_dataidx_map, train_dl=train_dl, test_dl=test_dl,
                            global_model=global_model, device=device)
            global_model.to('cpu')

            # update global model
            total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]

            for net_id, net in enumerate(nets_this_round.values()):
                net_para = net.state_dict()
                if net_id == 0:
                    for key in net_para:
                        global_w[key] = net_para[key] * fed_avg_freqs[net_id]
                else:
                    for key in net_para:
                        global_w[key] += net_para[key] * fed_avg_freqs[net_id]
            global_model.load_state_dict(global_w)

            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl))

            global_model.to(device)
            # train_acc, train_loss = compute_accuracy(global_model, train_dl_global, device=device, algo='fedprox')
            test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device, algo='fedprox')

            # logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)
            # logger.info('>> Global Model Train loss: %f' % train_loss)
            
            mkdirs(args.modeldir + 'fedprox/')
            global_model.to('cpu')
            torch.save(global_model.state_dict(), args.modeldir + 'fedprox/' + args.log_file_name + '.pth')
            
    elif args.alg == 'scaffold':
        c_global = copy.deepcopy(global_model)
        c_nets = copy.deepcopy(nets)
        for round in range(n_comm_rounds):
            logger.info("in comm round:" + str(round))
            party_list_this_round = party_list_rounds[round]

            global_w = global_model.state_dict()

            nets_this_round = {k: nets[k] for k in party_list_this_round}
            for net in nets_this_round.values():
                net.load_state_dict(global_w)

            _, c_global_para, net_global_para = local_train_net(nets_this_round, args, net_dataidx_map, train_dl=train_dl, test_dl=test_dl, 
                                            global_model=global_model, c_local=c_nets, c_global=c_global, device=device)
                
            c_global.load_state_dict(c_global_para)
            global_model.load_state_dict(net_global_para)
            logger.info('global n_test: %d' % len(test_dl))
            global_model.to(device)
            test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device, algo='scaffold')

            logger.info('>> Global Model Test accuracy: %f' % test_acc)

            mkdirs(args.modeldir + 'scaffold/')
            global_model.to('cpu')

            torch.save(global_model.state_dict(),
                       args.modeldir + 'scaffold/' + 'globalmodel' + args.log_file_name + '.pth')
            torch.save(nets[0].state_dict(), args.modeldir + 'scaffold/' + 'localmodel0' + args.log_file_name + '.pth')

    elif args.alg == 'local_training':
        logger.info("Initializing nets")
        local_train_net(nets, args, net_dataidx_map, train_dl=train_dl, test_dl=test_dl, device=device)
        mkdirs(args.modeldir + 'localmodel/')
        for net_id, net in nets.items():
            torch.save(net.state_dict(),
                       args.modeldir + 'localmodel/' + 'model' + str(net_id) + args.log_file_name + '.pth')

    elif args.alg == 'all_in':
        nets, _, _ = init_nets(args.net_config, 1, args, device='cpu')
        # nets[0].to(device)
        trainacc, testacc = train_net(0, nets[0], train_dl_global, test_dl, args.epochs, args.lr,
                                      args.optimizer, args, device=device)
        logger.info("All in test acc: %f" % testacc)
        mkdirs(args.modeldir + 'all_in/')

        torch.save(nets[0].state_dict(), args.modeldir + 'all_in/' + args.log_file_name + '.pth')

    elif args.alg == 'mine':
        t_net = copy.deepcopy(global_model.to(device))
        criterion = nn.CrossEntropyLoss().to(device)
        all_nets = copy.deepcopy(nets)
        num_participate, r_participate = [0] * args.n_parties, [-1] * args.n_parties
        total_participate, each_participate = 0, args.n_parties * args.sample_fraction
        for round in range(n_comm_rounds):
            logger.info("in comm round:" + str(round))
            party_list_this_round = party_list_rounds[round]
            global_ec_w, global_gen_w = global_model['ec'].state_dict(), global_model['gen'].state_dict()
            
            nets_this_round = {k: nets[k] for k in party_list_this_round}
            for k in party_list_this_round:
                num_participate[k] += 1
                r_participate[k] = round
            pitch_participate = [np.exp(-(round - r_participate[k])) for k in range(args.n_parties)]
            # print(nets_this_round)
            for net in nets_this_round.values():
                net['ec'].load_state_dict(global_ec_w)
                net['gen'].load_state_dict(global_gen_w)

            local_train_net(nets_this_round, args, net_dataidx_map, train_dl=train_dl, test_dl=test_dl,
                            teacher_model=copy.deepcopy(t_net), device=device, round=round)

            global_model.to('cpu')

            # avgfreqs
            total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]
            # trifreqs
            total_participate += each_participate
            participate_freqs = [num_participate[k] / total_participate for k in range(args.n_parties)]
            pitch_freqs = [pitch_participate[k] / sum(pitch_participate) for k in range(args.n_parties)]
            mix_freqs = [math.pow(all_freqs[k] * participate_freqs[k] * pitch_freqs[k], 1/3) for k in range(args.n_parties)]
            # mix_freqs = [(all_freqs[k] + participate_freqs[k] + pitch_freqs[k]) / 3 for k in range(args.n_parties)]
            # each_freqs = [mix_freqs[k] for k in range(party_list_this_round)]
            mix_freqs = [freqs / sum(mix_freqs) for freqs in mix_freqs]

            # update global model
            for idx, (net_id, net) in enumerate(nets_this_round.items()):
                # print(net.keys())
                net_para = net['ec'].state_dict()
                all_nets[net_id]['ec'].load_state_dict(net_para)
                if idx == 0:
                    for key in net_para:
                        global_ec_w[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        global_ec_w[key] += net_para[key] * fed_avg_freqs[idx]
            global_model['ec'].load_state_dict(global_ec_w)
            # update teacher model
            extra_class_w = copy.deepcopy(global_ec_w)
            for net_id, net in enumerate(all_nets.values()):
                net_para = net['ec'].state_dict()
                if net_id == 0:
                    for key in net_para:
                        extra_class_w[key] = net_para[key] * mix_freqs[net_id]
                else:
                    for key in net_para:
                        extra_class_w[key] += net_para[key] * mix_freqs[net_id]
            
            t_net['ec'].load_state_dict(extra_class_w)
            global_model.to(device)

            logger.info('>> Start training Generator !!! <<')
            
            G_optim = torch.optim.Adam(global_model['gen'].parameters(), lr=args.gen_lr, weight_decay=args.reg)
            frozen_net(global_model, ['ec'], frozen=True)
            frozen_net(global_model, ['gen'], frozen=False)
            # frozen_net(t_net, ['ec'], frozen=False)
            # frozen_net(t_net, ['gen'], frozen=False)
            y_ = torch.randint(0, args.num_classes, (args.global_iter_per_epoch * args.gen_bs,)).to(device)
        
            for epoch in range(args.global_gen_epoch):
                G_epoch_loss, reg_epoch_loss, div_epoch_loss = [], [], []
                cnt = 0
                indices = torch.randperm(y_.size(0))
                y_ = y_[indices]
                for batch in range(args.global_iter_per_epoch):
                    z = torch.randn(args.gen_bs, args.noise_dim, 1, 1).to(device)
                    y = y_[cnt:cnt + args.gen_bs]
                    # print(y.shape)
                    cnt += args.gen_bs
                    G_optim.zero_grad()

                    G = global_model['gen'](z, y)
                    logit = 0
                    for idx, net in enumerate(nets_this_round.values()):
                        net.to(device)
                        _, _, GC = net['ec'](G, withgen=True)
                        # print(len(GC))
                        logit += fed_avg_freqs[idx] * GC
                        net.to('cpu')
                    # _, z2, logit_global = t_net['ec'](G, withgen=True)
    
                    G_loss = criterion(logit, y)
                    # reg_loss = (round / n_comm_rounds) * torch.norm((z1 - z2), dim=1).mean()

                    split_size = int(args.gen_bs / 2)
                    z1, z2 = torch.split(z, split_size, dim=0)
                    G1, G2 = torch.split(G, split_size, dim=0)
                    div = torch.mean(torch.abs(G1 - G2)) / torch.mean(torch.abs(z1 - z2)).to(device)
                    eps = 1e-5
                    div_loss = 1 / (div + eps)
                    # print(div_loss)
                    # reg_loss = 1 / (reg + eps)
                    # print((G_loss + div_loss).device)
                    (G_loss + div_loss).backward()
                    G_optim.step()
                    G_epoch_loss.append(G_loss)
                    # reg_epoch_loss.append(reg_loss)
                    div_epoch_loss.append(div_loss)
                logger.info("Epoch {}: G loss:{:.4f}, div loss: {:.4f}".format(epoch, sum(G_epoch_loss) / len(
                    G_epoch_loss), sum(div_epoch_loss) / len(div_epoch_loss)))
            
            frozen_net(global_model, ['gen'], frozen=True)
            logger.info(">> End generator training !!! <<")
            
            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl))
            
            # train_acc, train_loss = compute_accuracy(global_model, train_dl_global, device=device, algo='mine')
            test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device, algo='mine')
            tnet_test_acc, _, _ = compute_accuracy(t_net, test_dl, get_confusion_matrix=True, device=device, algo='mine')

            # logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)
            # logger.info('>> Global Model Train loss: %f' % train_loss)
            logger.info('>> Teacher Model Test accuracy: %f' % tnet_test_acc)
            mkdirs(args.modeldir + 'fedmine/')
            global_model.to('cpu')
            torch.save(global_model.state_dict(), args.modeldir + 'fedmine/' + args.log_file_name + '.pth')
