import tensorflow as tf
import os
import time
from util.config import LOG_DIR

TF_LOG_DIR = os.path.join(LOG_DIR, 'tensorboard')

loggers = {}


class TensorboardLogger:
    """
    Logs variables visibly to the tensorboard interface.
    """

    _is_in_setup_phase = True
    _writer = None
    _sess = None
    _iteration = 0
    _tracked_vars = {}
    _graph = None
    _logs_path = None

    def __init__(self, logs_path):
        self._graph = tf.Graph()
        self._logs_path = logs_path + str(int(time.time()))
        self._sess = tf.Session(graph=self._graph)
        print("Start this tensorboard with \n\ttensorboard --logdir={}".format(self._logs_path))

    # Functions for setting up which variables should be tracked

    def is_in_setup_phase(self):
        if not self._is_in_setup_phase:
            raise RuntimeError("Logger has been already written something. Can't use this action.")

    def name_not_used(self, name):
        if name in self._tracked_vars.keys():
            raise RuntimeError("You tried to create the variable {}, which has already been assigned.".format(name))

    def track_scalar(self, name):
        self.is_in_setup_phase()
        self.name_not_used(name)

        with self._graph.as_default():
            self._tracked_vars[name] = {}
            self._tracked_vars[name]['node'] = tf.placeholder(tf.float32, shape=(), name=name)
            tf.summary.scalar(name, self._tracked_vars[name]['node'])
            self._tracked_vars[name]['value'] = None

    def track_histogram(self, name):
        self.is_in_setup_phase()
        self.name_not_used(name)

        with self._graph.as_default():
            self._tracked_vars[name] = {}
            self._tracked_vars[name]['node'] = tf.placeholder(tf.float32, shape=None, name=name)
            tf.summary.histogram(name, self._tracked_vars[name]['node'])
            self._tracked_vars[name]['value'] = None

    # TODO mabe track metadata as well (run parameters)

    # Functions for updating and writing the summaries

    def start_writer(self):

        self._writer = tf.summary.FileWriter(self._logs_path, graph=self._graph,flush_secs=1)
        with self._graph.as_default():
            self._sess.run(tf.global_variables_initializer())
        self._is_in_setup_phase = False

    def update(self, name, value):
        self._tracked_vars[name]['value'] = value

    def write_summary(self):
        with self._graph.as_default():
            summary_op = tf.summary.merge_all()
            summary = self._sess.run(summary_op, feed_dict=self._feed_dict())
            print(summary)
            self._writer.add_summary(summary, self._iteration)
            self._iteration = 1 + self._iteration

    def close(self):
        self._sess.close()

    # Getters and Setters

    def _feed_dict(self):
        return {var['node']: var['value'] for var in self._tracked_vars.values()}


### Functions for retrieving loggers of a specific context

def new_logger(name, logs_path=None):
    if logs_path is None:
        logs_path = os.path.join(TF_LOG_DIR, name)
    loggers[name] = TensorboardLogger(logs_path)
    return loggers[name]


def get_logger(name):
    if name in loggers.keys():
        return loggers[name]
    else:
        return new_logger(name)
