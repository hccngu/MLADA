import time
import numpy as np

from multiprocessing import Process, Queue, cpu_count
import dataset.utils as utils


class ParallelSampler():

    def __init__(self, data, args, sampled_classes, source_classes, num_episodes=None):
        self.data = data
        self.args = args
        self.num_episodes = num_episodes
        self.sampled_classes = sampled_classes
        self.source_classes = source_classes

        self.all_classes = np.unique(self.data['label'])
        self.num_classes = len(self.all_classes)
        if self.num_classes < self.args.way:
            raise ValueError("Total number of classes is less than #way.")

        self.idx_list = []
        for y in self.all_classes:
            self.idx_list.append(
                np.squeeze(np.argwhere(self.data['label'] == y)))

        self.count = 0
        self.done_queue = Queue()

        self.num_cores = cpu_count() if args.n_workers is 0 else args.n_workers

        self.p_list = []
        for i in range(self.num_cores):
            self.p_list.append(
                Process(target=self.worker, args=(self.done_queue, self.sampled_classes, self.source_classes)))

        for i in range(self.num_cores):
            self.p_list[i].start()

    def get_epoch(self):

        for _ in range(self.num_episodes):
            # wait until self.thread finishes
            support, query, source = self.done_queue.get()

            # convert to torch.tensor
            support = utils.to_tensor(support, self.args.cuda, ['raw'])
            query = utils.to_tensor(query, self.args.cuda, ['raw'])
            source = utils.to_tensor(source, self.args.cuda, ['raw'])

            support['is_support'] = True
            query['is_support'] = False
            source['is_support'] = False

            yield support, query, source

    def worker(self, done_queue, sampled_classes, source_classes):
        '''
            Generate one task (support and query).
            Store into self.support[self.cur] and self.query[self.cur]
        '''
        while True:
            if done_queue.qsize() > 100:
                time.sleep(1)
                continue

            # sample examples
            support_idx, query_idx, source_idx = [], [], []
            for y in sampled_classes:
                tmp = np.random.permutation(len(self.idx_list[y]))
                support_idx.append(
                    self.idx_list[y][tmp[:self.args.shot]])
                query_idx.append(
                    self.idx_list[y][
                        tmp[self.args.shot:self.args.shot + self.args.query]])

            for z in source_classes:
                tmp = np.random.permutation(len(self.idx_list[z]))
                source_idx.append(
                    tmp[:self.args.query]
                )

            support_idx = np.concatenate(support_idx)
            query_idx = np.concatenate(query_idx)
            source_idx = np.concatenate(source_idx)

            # aggregate examples
            max_support_len = np.max(self.data['text_len'][support_idx])
            max_query_len = np.max(self.data['text_len'][query_idx])
            max_source_len = np.max(self.data['text_len'][source_idx])

            support = utils.select_subset(self.data, {}, ['text', 'text_len', 'label'],
                                          support_idx, max_support_len)
            query = utils.select_subset(self.data, {}, ['text', 'text_len', 'label'],
                                        query_idx, max_query_len)
            source = utils.select_subset(self.data, {}, ['text', 'text_len', 'label'],
                                         source_idx, max_source_len)

            done_queue.put((support, query, source))

    def __del__(self):
        '''
            Need to terminate the processes when deleting the object
        '''
        for i in range(self.num_cores):
            self.p_list[i].terminate()

        del self.done_queue


class ParallelSampler_Test():

    def __init__(self, data, args, num_episodes=None):
        self.data = data
        self.args = args
        self.num_episodes = num_episodes

        self.all_classes = np.unique(self.data['label'])
        self.num_classes = len(self.all_classes)
        if self.num_classes < self.args.way:
            raise ValueError("Total number of classes is less than #way.")

        self.idx_list = []
        for y in self.all_classes:
            self.idx_list.append(
                np.squeeze(np.argwhere(self.data['label'] == y)))

        self.count = 0
        self.done_queue = Queue()

        self.num_cores = cpu_count() if args.n_workers is 0 else args.n_workers

        self.p_list = []
        for i in range(self.num_cores):
            self.p_list.append(
                Process(target=self.worker, args=(self.done_queue,)))

        for i in range(self.num_cores):
            self.p_list[i].start()

    def get_epoch(self):
        for _ in range(self.num_episodes):
            # wait until self.thread finishes
            support, query = self.done_queue.get()

            # convert to torch.tensor
            support = utils.to_tensor(support, self.args.cuda, ['raw'])
            query = utils.to_tensor(query, self.args.cuda, ['raw'])

            support['is_support'] = True
            query['is_support'] = False

            yield support, query

    def worker(self, done_queue):
        '''
            Generate one task (support and query).
            Store into self.support[self.cur] and self.query[self.cur]
        '''
        while True:
            if done_queue.qsize() > 100:
                time.sleep(1)
                continue
            # sample ways
            sampled_classes = np.random.permutation(
                self.num_classes)[:self.args.way]

            source_classes = []
            for j in range(self.num_classes):
                if j not in sampled_classes:
                    source_classes.append(self.all_classes[j])
            source_classes = sorted(source_classes)

            # sample examples
            support_idx, query_idx = [], []
            for y in sampled_classes:
                tmp = np.random.permutation(len(self.idx_list[y]))
                support_idx.append(
                    self.idx_list[y][tmp[:self.args.shot]])
                query_idx.append(
                    self.idx_list[y][
                        tmp[self.args.shot:self.args.shot + self.args.query]])

            support_idx = np.concatenate(support_idx)
            query_idx = np.concatenate(query_idx)
            if self.args.mode == 'finetune' and len(query_idx) == 0:
                query_idx = support_idx

            # aggregate examples
            max_support_len = np.max(self.data['text_len'][support_idx])
            max_query_len = np.max(self.data['text_len'][query_idx])

            support = utils.select_subset(self.data, {}, ['text', 'text_len', 'label'],
                                          support_idx, max_support_len)
            query = utils.select_subset(self.data, {}, ['text', 'text_len', 'label'],
                                        query_idx, max_query_len)

            done_queue.put((support, query))

    def __del__(self):
        '''
            Need to terminate the processes when deleting the object
        '''
        for i in range(self.num_cores):
            self.p_list[i].terminate()

        del self.done_queue


def task_sampler(data, args):
    all_classes = np.unique(data['label'])
    num_classes = len(all_classes)

    # sample classes
    temp = np.random.permutation(num_classes)
    sampled_classes = temp[:args.way]

    source_classes = temp[args.way:args.way + args.way]

    return sampled_classes, source_classes
