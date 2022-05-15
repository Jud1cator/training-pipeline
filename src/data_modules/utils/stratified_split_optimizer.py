import copy
import logging
import math
import random

import numpy as np


class StratifiedSplitOptimizer:
    def __init__(self, dataset, n_classes, split_ratios):
        self.class_occurrence_index = np.zeros((len(dataset), n_classes))
        for i in range(len(dataset)):
            _, mask = dataset[i]
            classes, counts = np.unique(mask, return_counts=True)
            self.class_occurrence_index[:, classes] += counts

        self.split_ratios = split_ratios
        self.normalized_class_occurrence_index = \
            np.nan_to_num(self.class_occurrence_index / self.class_occurrence_index.sum(axis=0))

    def mse_to_target_split(self, splits):
        distance = 0
        for split_id in range(len(splits)):
            distance += np.mean(
                (
                    self.normalized_class_occurrence_index[splits[split_id]].sum(axis=0) -
                    self.split_ratios[split_id]
                )**2
            )
        return distance

    @staticmethod
    def swap_image_ids(splits, split1, split2, id1, id2):
        new_splits = copy.deepcopy(splits)
        new_splits[split1].remove(id1)
        new_splits[split1].append(id2)
        new_splits[split2].remove(id2)
        new_splits[split2].append(id1)
        return new_splits

    @staticmethod
    def insert_image_id(splits, split1, split2, id):
        new_splits = copy.deepcopy(splits)
        new_splits[split1].remove(id)
        new_splits[split2].append(id)
        return new_splits

    def swap(self, splits):
        new_splits = copy.deepcopy(splits)
        split1, split2 = random.sample(list(range(len(new_splits))), 2)
        idx1 = random.choice(new_splits[split1])
        idx2 = random.choice(new_splits[split2])
        return self.swap_image_ids(new_splits, split1, split2, idx1, idx2)

    def greedy_swap(self, splits):
        new_splits = copy.deepcopy(splits)
        split1, split2 = random.sample(list(range(len(new_splits))), 2)
        idx1 = random.choice(new_splits[split1])
        idx2 = random.choice(new_splits[split2])
        swap = self.swap_image_ids(new_splits, split1, split2, idx1, idx2)
        insert = self.insert_image_id(new_splits, split1, split2, idx1)
        return swap if self.mse_to_target_split(swap) < self.mse_to_target_split(insert) else insert

    def find_approximately_optimal_split(
            self,
            splits,
            initial_temp=1000,
            cooling_rate=0.001
    ):
        initial_splits = copy.deepcopy(splits)
        initial_error = self.mse_to_target_split(splits)
        temp = initial_temp

        logging.info('Searching for a better stratified split...')

        while temp > 1:
            current_error = self.mse_to_target_split(splits)

            new_splits = self.greedy_swap(splits)
            new_error = self.mse_to_target_split(new_splits)
            if new_error < current_error or \
                    random.uniform(0, 1) < math.exp((current_error - new_error) / temp):
                splits = new_splits
            temp *= 1 - cooling_rate

        final_error = self.mse_to_target_split(splits)
        if final_error > initial_error:
            logging.info('Failed to optimize a stratified split, initial split will be used')
            splits = initial_splits
        else:
            logging.info('Successfully found better stratified split')
        return splits
