import datetime

import numpy as np
import torch
from termcolor import colored
from tqdm import tqdm

from dataset.sampler import ParallelSampler_Test


def test_one(task, model, args):
    '''
        Evaluate the model on one sampled task. Return the accuracy.
    '''
    support, query = task
    # print("query_text.shape:", query['text'].shape)

    if args.embedding != 'mlada':

        # Embedding the document
        XS = model['ebd'](support)
        YS = support['label']

        XQ = model['ebd'](query)
        YQ = query['label']

        # Apply the classifier
        acc, _ = model['clf'](XS, YS, XQ, YQ)

        return acc

    else:
        # Embedding the document
        XS, XS_inputD, XS_avg = model['G'](support, flag='support')
        YS = support['label']
        # print('YS', YS)

        XQ, XQ_inputD, XQ_avg = model['G'](query, flag='query')
        YQ = query['label']
        YQ_d = torch.ones(query['label'].shape, dtype=torch.long).to(query['label'].device)
        # print('YQ', set(YQ.numpy()))

        XSource, XSource_inputD, _ = model['G'](query, flag='query')
        YSource_d = torch.zeros(query['label'].shape, dtype=torch.long).to(query['label'].device)

        XQ_logitsD = model['D'](XQ_inputD)
        XSource_logitsD = model['D'](XSource_inputD)

        query_data = query['text']
        if query_data.shape[1] < 50:
            zero = torch.zeros((query_data.shape[0], 50 - query_data.shape[1]))
            if args.cuda != -1:
                zero = zero.cuda(args.cuda)
            query_data = torch.cat((query_data, zero), dim=-1)
            # print('reverse_feature.shape[1]', reverse_feature.shape[1])
        else:
            query_data = query_data[:, :50]
            # print('reverse_feature.shape[1]', reverse_feature.shape[1])

        # Apply the classifier
        acc, d_acc, loss, x_hat = model['clf'](XS, YS, XQ, YQ, XQ_logitsD, XSource_logitsD, YQ_d, YSource_d, query_data)

        all_sentence_ebd = XQ
        all_avg_sentence_ebd = XQ_avg
        all_label = YQ
        # print(all_sentence_ebd.shape, all_avg_sentence_ebd.shape, all_label.shape)

        return acc, d_acc, all_sentence_ebd.cpu().detach().numpy(), all_avg_sentence_ebd.cpu().detach().numpy(), all_label.cpu().detach().numpy(), XQ_inputD.cpu().detach().numpy(), query_data.cpu().detach().numpy(), x_hat.cpu().detach().numpy()


def test(test_data, model, args, num_episodes, verbose=True, sampled_tasks=None):
    '''
        Evaluate the model on a bag of sampled tasks. Return the mean accuracy
        and its std.
    '''
    model['G'].eval()
    model['D'].eval()
    model['clf'].eval()

    if sampled_tasks is None:
        sampled_tasks = ParallelSampler_Test(test_data, args,
                                        num_episodes).get_epoch()

    acc = []
    d_acc = []
    all_sentence_ebd = None
    all_avg_sentence_ebd = None
    all_sentence_label = None
    all_word_weight = None
    all_query_data = None
    all_x_hat = None
    all_drawn_data = {}
    if not args.notqdm:
        sampled_tasks = tqdm(sampled_tasks, total=num_episodes, ncols=80,
                             leave=False,
                             desc=colored('Testing on val', 'yellow'))
    count = 0
    for task in sampled_tasks:
        if args.embedding == 'mlada':
            acc1, d_acc1, sentence_ebd, avg_sentence_ebd, sentence_label, word_weight, query_data, x_hat = test_one(task, model, args)
            if count < 20:
                if all_sentence_ebd is None:
                    all_sentence_ebd = sentence_ebd
                    all_avg_sentence_ebd = avg_sentence_ebd
                    all_sentence_label = sentence_label
                    all_word_weight = word_weight
                    all_query_data = query_data
                    all_x_hat = x_hat
                else:
                    all_sentence_ebd = np.concatenate((all_sentence_ebd, sentence_ebd), 0)
                    all_avg_sentence_ebd = np.concatenate((all_avg_sentence_ebd, avg_sentence_ebd), 0)
                    all_sentence_label = np.concatenate((all_sentence_label, sentence_label))
                    all_word_weight = np.concatenate((all_word_weight, word_weight), 0)
                    all_query_data = np.concatenate((all_query_data, query_data), 0)
                    all_x_hat = np.concatenate((all_x_hat, x_hat), 0)
            count = count + 1
            acc.append(acc1)
            d_acc.append(d_acc1)
        else:
            acc.append(test_one(task, model, args))

    acc = np.array(acc)
    d_acc = np.array(d_acc)
    # all_drawn_data["sentence_ebd"] = all_sentence_ebd.tolist()
    # all_drawn_data["avg_sentence_ebd"] = all_avg_sentence_ebd.tolist()
    # all_drawn_data["label"] = all_sentence_label.tolist()
    # all_drawn_data["word_weight"] = all_word_weight.tolist()
    # all_drawn_data["query_data"] = all_query_data.tolist()
    # all_x = []
    # for _x in all_x_hat.tolist():
    #     x_ = []
    #     for x_x in _x:
    #         x_.append(args.id2word[x_x])
    #     all_x.append(x_)
    # all_drawn_data["x_hat"] = all_x


    if verbose:
        if args.embedding != 'mlada':
            print("{}, {:s} {:>7.4f}, {:s} {:>7.4f}".format(
                datetime.datetime.now(),
                colored("test acc mean", "blue"),
                np.mean(acc),
                colored("test std", "blue"),
                np.std(acc),
                ), flush=True)
        else:
            print("{}, {:s} {:>7.4f}, {:s} {:>7.4f}".format(
                datetime.datetime.now(),
                colored("test acc mean", "blue"),
                np.mean(acc),
                colored("test std", "blue"),
                np.std(acc),
                colored("test d_acc mean", "blue"),
                np.mean(d_acc),
                colored("test d_acc std", "blue"),
                np.std(d_acc),
            ), flush=True)

    return np.mean(acc), np.std(acc), all_drawn_data