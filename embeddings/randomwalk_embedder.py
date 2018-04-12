import urllib.request
import collections
import math
import os
import random
import zipfile
from collections import deque
import datetime as dt

import numpy as np
import tensorflow as tf

class BatchGenerator:
    def generate_batch(self, batch_size, num_skips):
        raise NotImplementedError()

class LongWalkBatchGenerator(BatchGenerator):

    def __init__(self, walk_list, skip_window):
        self.walk_list = walk_list
        self.skip_window = skip_window
        self.iterator = self.global_window_iterator(self.walk_list, self.skip_window)

    def global_window_iterator(self, walk_list, skip_window):
        span_size = 2 * skip_window + 1  # [ skip_window input_word skip_window ]
        for walk in walk_list:
            prepared_walk_list = self.prepare_edge_for_window(walk, skip_window)
            for window in self.sliding_window(prepared_walk_list, span_size):
                yield window

    def generate_batch(self, batch_size, num_skips):
        assert batch_size % num_skips == 0
        assert num_skips <= 2 * self.skip_window
        batch = np.ndarray(shape=(batch_size), dtype=np.int32)
        batch_index = 0
        context = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

        for i in range(batch_size // num_skips):
            window = next(self.iterator, None)

            # if iterator has finished restart it
            if window is None:
                self.iterator = self.global_window_iterator(self.walk_list, self.skip_window)
                window = next(self.iterator, None)

            window[0], window[len(window) / 2] = window[len(window) / 2], window[
                0]  # swap middle with first element
            focus_node = window[0]

            selected_context_nodes = random.sample(range(1, len(window)), num_skips)
            for selection in selected_context_nodes:
                if window[selection] is None:
                    continue
                batch[batch_index] = focus_node
                context[batch_index, 0] = window[selection]
                batch_index += 1

        return batch, context

    @staticmethod
    def sliding_window(seq, span_size):
        it = iter(seq)
        win = deque((next(it, None) for _ in range(span_size)), maxlen=span_size)
        yield win
        append = win.append
        for e in it:
            append(e)
            yield win

    @staticmethod
    def prepare_edge_for_window(seq, edge_size):
        dummy = [None] * edge_size
        prepared = dummy + seq + dummy
        return prepared


# generate batch data from short walks: the whole walk is taken as the context for the start node
class ShortWalkBatchGenerator(BatchGenerator):

    def __init__(self, walk_list, skip_window):
        self.walk_list = walk_list
        self.data_index = 0

    # TODO should nodes at the beginning of a walk have a higher selection probabilty?
    def generate_batch(self, batch_size, num_skips):
        assert batch_size % num_skips == 0
        batch = np.ndarray(shape=(batch_size), dtype=np.int32)
        context = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        for i in range(batch_size // num_skips):
            current_walk = self.walk_list[self.data_index]
            focus_node = current_walk[0]

            assert len(current_walk) >= num_skips, "{} >= {}".format(len(current_walk), num_skips)

            selected_context_nodes = random.sample(range(1, len(current_walk)), num_skips)
            for counter, selection in enumerate(selected_context_nodes):
                batch[i * num_skips + counter] = focus_node
                context[i * num_skips + counter, 0] = current_walk[selection]

            self.data_index = (self.data_index + 1) % len(self.walk_list)
        return batch, context


# Walks are given like [2314, 123123, 4324, 2344, 2344]; the first line contains the column names and is therefore skipped
def read_walks(file_name):
    walk_list = []
    available_nodes = set()
    with open(file_name) as file:
        file.__next__()  # skip first line
        for line in file:
            content = line[line.find("[") + 1:line.find("]")]
            node_ids = [int(id) for id in content.split(", ")]

            walk_list.append(node_ids)
            available_nodes |= set(node_ids)
    return walk_list, available_nodes


class BatchGeneratorWrapper:

    def __init__(self, walk_list, skip_window, available_nodes, batch_generator_class: BatchGenerator.__class__, id_mapping=None):
        self.id_mapping = id_mapping
        if self.id_mapping is None:
            self.id_mapping = self.create_id_mapping(available_nodes=available_nodes)

        self.inverse_mapping = {v: k for k, v in self.id_mapping.items()}
        converted_walks = self.convert_walks(walk_list, self.id_mapping)

        self.batch_generator = batch_generator_class(converted_walks, skip_window)

    @staticmethod
    def create_id_mapping(available_nodes):
        available_nodes_counter = range(len(available_nodes))
        return dict(zip(available_nodes, available_nodes_counter))

    @staticmethod
    def convert_walks(walk_list, id_mapping):
        converted_walks = []
        for walk in walk_list:
            converted_path = []
            for id in walk:
                converted_path.append(id_mapping[id])
            converted_walks.append(converted_path)
        return converted_walks

    def generate_batch(self, batch_size, num_skips):
        return self.batch_generator.generate_batch(batch_size, num_skips)

    def get_original_id(self, own_id):
        return self.inverse_mapping[own_id]

    def get_translated_id(self, original_id):
        return self.id_mapping[original_id]



