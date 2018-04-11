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


class LongWalkBatchGenerator:

    def __init__(self, walk_list, available_nodes, skip_window):
        self.walk_list = walk_list
        self.available_nodes = available_nodes
        self.skip_window = skip_window
        self.iterator = self.global_window_iterator(self.walk_list, self.skip_window)
      
    @classmethod
    def fromfilename(cls, file_name, skip_window):
        walk_list, available_nodes = self.read_walks(file_name)
        return cls(walk_list, available_nodes, skip_window)
    
    @staticmethod
    def global_window_iterator(self, walk_list, skip_window):
        span_size = 2 * skip_window + 1  # [ skip_window input_word skip_window ]
        for walk in walk_list:
            prepared_walk_list = self.prepare_edge_for_window(walk, skip_window)
            for window in self.sliding_window(prepared_walk_list, span_size):
                yield window

    # generate batch data from long walks: the walk is read as a "sentence" using a sliding window
    def generate_batch(self, batch_size, num_skips):
        assert batch_size % num_skips == 0
        assert num_skips <= 2 * self.skip_window
        batch = np.ndarray(shape=(batch_size), dtype=np.int32)
        batch_index = 0
        context = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

        for i in range(batch_size // num_skips):
            window  = next(self.iterator, None)

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
    def sliding_window(self, seq, span_size):
        it = iter(seq)
        win = deque((next(it, None) for _ in range(span_size)), maxlen=span_size)
        yield win
        append = win.append
        for e in it:
            append(e)
            yield win

    @staticmethod
    def prepare_edge_for_window(self, seq, edge_size):
        dummy = [None] * edge_size
        prepared = dummy + seq + dummy
        return prepared

    @staticmethod
    def read_walks(self, file_name):
        walk_list = []
        available_nodes = set()
        with open(file_name) as file:
            for line in file:
                node_ids = [int(id) for id in line.split()]

                walk_list.append(node_ids)
                available_nodes |= set(node_ids)
        return walk_list, available_nodes


class ShortWalkBatchGenerator:

    def __init__(self, walk_list, available_nodes):
        self.walk_list = walk_list
        self.available_nodes = available_nodes
        self.data_index = 0
        
    @classmethod
    def fromfilename(cls, file_name, skip_window):
        walk_list, available_nodes = self.read_walks(file_name)
        return cls(walk_list, available_nodes, skip_window)

    @staticmethod
    def read_walks(file_name):
        walk_list = []
        available_nodes = set()
        with open(file_name) as file:
            for line in file:
                node_ids = line.split()

                start_id = int(node_ids[0])
                walk_id_list = [int(id) for id in node_ids[1:]]

                walk_list.append([start_id, walk_id_list])
                available_nodes.add(start_id)
        return walk_list, available_nodes

    # generate batch data from short walks: the whole walk is taken as the context for the start node
    # TODO should nodes at the beginning of a walk have a higher selection probabilty?
    def generate_batch(self, batch_size, num_skips):
        assert batch_size % num_skips == 0
        batch = np.ndarray(shape=(batch_size), dtype=np.int32)
        context = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        for i in range(batch_size // num_skips):
            current_walk = self.walk_list[self.data_index]
            focus_key = int(len(current_walk)/2)
            focus_node = current_walk[focus_key]
            walk = np.delete(np.array(current_walk), focus_key)

            assert len(current_walk) >= num_skips, "{} >= {}".format(len(current_walk), num_skips)

            selected_context_nodes = random.sample(range(0, len(walk)), num_skips)
            for counter, selection in enumerate(selected_context_nodes):
                batch[i * num_skips + counter] = focus_node
                context[i * num_skips + counter, 0] = walk[selection]

                self.data_index = (self.data_index + 1) % len(self.walk_list)
        return batch, context


