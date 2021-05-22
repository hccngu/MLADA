import os
import time
import datetime

import torch
import numpy as np

from train.utils import grad_param, get_norm
from dataset.sampler import ParallelSampler, ParallelSampler_Test, task_sampler
from tqdm import tqdm
from termcolor import colored
from train.test import test
import torch.nn.functional as F


def train(train_data, val_data, model, args):
    '''
        Train the model
        Use val_data to do early stopping
    '''
    # creating a tmp directory to save the models
    out_dir = os.path.abspath(os.path.join(
                                  os.path.curdir,
                                  "tmp-runs",
                                  str(int(time.time() * 1e7))))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    best_acc = 0
    sub_cycle = 0
    best_path = None

    optG = torch.optim.Adam(grad_param(model, ['G', 'clf']), lr=args.lr_g)
    optD = torch.optim.Adam(grad_param(model, ['D']), lr=args.lr_d)

    if args.lr_scheduler == 'ReduceLROnPlateau':
        schedulerG = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optG, 'max', patience=args.patience//2, factor=0.1, verbose=True)
        schedulerD = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optD, 'max', patience=args.patience // 2, factor=0.1, verbose=True)

    elif args.lr_scheduler == 'ExponentialLR':
        schedulerG = torch.optim.lr_scheduler.ExponentialLR(optG, gamma=args.ExponentialLR_gamma)
        schedulerD = torch.optim.lr_scheduler.ExponentialLR(optD, gamma=args.ExponentialLR_gamma)



    print("{}, Start training".format(
        datetime.datetime.now()), flush=True)

    # train_gen = ParallelSampler(train_data, args, args.train_episodes)
    train_gen_val = ParallelSampler_Test(train_data, args, args.val_episodes)
    val_gen = ParallelSampler_Test(val_data, args, args.val_episodes)

    # sampled_classes, source_classes = task_sampler(train_data, args)
    for ep in range(args.train_epochs):

        sampled_classes, source_classes = task_sampler(train_data, args)

        train_gen = ParallelSampler(train_data, args, sampled_classes, source_classes, args.train_episodes)

        sampled_tasks = train_gen.get_epoch()

        grad = {'clf': [], 'G': [], 'D': []}

        if not args.notqdm:
            sampled_tasks = tqdm(sampled_tasks, total=train_gen.num_episodes,
                    ncols=80, leave=False, desc=colored('Training on train',
                        'yellow'))
        d_acc = 0
        for task in sampled_tasks:
            if task is None:
                break
            d_acc += train_one(task, model, optG, optD, args, grad)

        d_acc = d_acc / args.train_episodes

        print("---------------ep:" + str(ep) + " d_acc:" + str(d_acc) + "-----------")

        if ep % 10 == 0:

            acc, std, _ = test(train_data, model, args, args.val_episodes, False,
                            train_gen_val.get_epoch())
            print("{}, {:s} {:2d}, {:s} {:s}{:>7.4f} ± {:>6.4f} ".format(
                datetime.datetime.now(),
                "ep", ep,
                colored("train", "red"),
                colored("acc:", "blue"), acc, std,
                ), flush=True)

        # Evaluate validation accuracy
        cur_acc, cur_std, _ = test(val_data, model, args, args.val_episodes, False,
                                val_gen.get_epoch())
        print(("{}, {:s} {:2d}, {:s} {:s}{:>7.4f} ± {:>6.4f}, "
               "{:s} {:s}{:>7.4f}, {:s}{:>7.4f}").format(
               datetime.datetime.now(),
               "ep", ep,
               colored("val  ", "cyan"),
               colored("acc:", "blue"), cur_acc, cur_std,
               colored("train stats", "cyan"),
               colored("G_grad:", "blue"), np.mean(np.array(grad['G'])),
               colored("clf_grad:", "blue"), np.mean(np.array(grad['clf'])),
               ), flush=True)

        # Update the current best model if val acc is better
        if cur_acc > best_acc:
            best_acc = cur_acc
            best_path = os.path.join(out_dir, str(ep))

            # save current model
            print("{}, Save cur best model to {}".format(
                datetime.datetime.now(),
                best_path))

            torch.save(model['G'].state_dict(), best_path + '.G')
            torch.save(model['D'].state_dict(), best_path + '.D')
            torch.save(model['clf'].state_dict(), best_path + '.clf')

            sub_cycle = 0
        else:
            sub_cycle += 1

        # Break if the val acc hasn't improved in the past patience epochs
        if sub_cycle == args.patience:
            break

        if args.lr_scheduler == 'ReduceLROnPlateau':
            schedulerG.step(cur_acc)
            schedulerD.step(cur_acc)

        elif args.lr_scheduler == 'ExponentialLR':
            schedulerG.step()
            schedulerD.step()

    print("{}, End of training. Restore the best weights".format(
            datetime.datetime.now()),
            flush=True)

    # restore the best saved model
    model['G'].load_state_dict(torch.load(best_path + '.G'))
    model['D'].load_state_dict(torch.load(best_path + '.D'))
    model['clf'].load_state_dict(torch.load(best_path + '.clf'))

    if args.save:
        # save the current model
        out_dir = os.path.abspath(os.path.join(
                                      os.path.curdir,
                                      "saved-runs",
                                      str(int(time.time() * 1e7))))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        best_path = os.path.join(out_dir, 'best')

        print("{}, Save best model to {}".format(
            datetime.datetime.now(),
            best_path), flush=True)

        torch.save(model['G'].state_dict(), best_path + '.G')
        torch.save(model['D'].state_dict(), best_path + '.D')
        torch.save(model['clf'].state_dict(), best_path + '.clf')

        with open(best_path + '_args.txt', 'w') as f:
            for attr, value in sorted(args.__dict__.items()):
                f.write("{}={}\n".format(attr, value))

    return


def train_one(task, model, optG, optD, args, grad):
    '''
        Train the model on one sampled task.
    '''
    model['G'].train()
    model['D'].train()
    model['clf'].train()

    support, query, source = task
    for _ in range(args.k):
        # ***************update D**************
        optD.zero_grad()

        # Embedding the document
        XS, XS_inputD, _ = model['G'](support, flag='support')
        YS = support['label']
        # print('YS', YS)

        XQ, XQ_inputD, _ = model['G'](query, flag='query')
        YQ = query['label']
        YQ_d = torch.ones(query['label'].shape, dtype=torch.long).to(query['label'].device)
        # print('YQ', set(YQ.numpy()))

        XSource, XSource_inputD, _ = model['G'](source, flag='query')
        YSource_d = torch.zeros(source['label'].shape, dtype=torch.long).to(source['label'].device)

        XQ_logitsD = model['D'](XQ_inputD)
        XSource_logitsD = model['D'](XSource_inputD)

        d_loss = F.cross_entropy(XQ_logitsD, YQ_d) + F.cross_entropy(XSource_logitsD, YSource_d)
        d_loss.backward(retain_graph=True)
        grad['D'].append(get_norm(model['D']))
        optD.step()

        # *****************update G****************
        optG.zero_grad()
        XQ_logitsD = model['D'](XQ_inputD)
        XSource_logitsD = model['D'](XSource_inputD)
        d_loss = F.cross_entropy(XQ_logitsD, YQ_d) + F.cross_entropy(XSource_logitsD, YSource_d)

        acc, d_acc, loss, _ = model['clf'](XS, YS, XQ, YQ, XQ_logitsD, XSource_logitsD, YQ_d, YSource_d)

        g_loss = loss - d_loss
        if args.ablation == "-DAN":
            g_loss = loss
            print("%%%%%%%%%%%%%%%%%%%This is ablation mode: -DAN%%%%%%%%%%%%%%%%%%%%%%%%%%")
        g_loss.backward(retain_graph=True)
        grad['G'].append(get_norm(model['G']))
        grad['clf'].append(get_norm(model['clf']))
        optG.step()

    return d_acc