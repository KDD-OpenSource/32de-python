from typing import Tuple
import tensorflow as tf


class Input:
    @classmethod
    def from_json(cls, data):
        raise NotImplementedError()

    def input(self) -> Tuple:
        raise NotImplementedError()


class MetaPathsInput(Input):

    def __init__(self, dataset):
        self.dataset = self.apply_transformation(dataset)

    @classmethod
    def from_json(cls, data):
        dataset = tf.data.Dataset()

        return cls(dataset)

    def apply_transformation(self, dataset):
        pass

    @classmethod
    def parse_meta_paths(json, min_size=5, seperator=" | "):
        walk_list = []
        available_nodes = set()
        for meta_paths in json.keys():
            node_ids = [int(id) for id in meta_paths.split(seperator)]
            if (len(node_ids) < min_size):
                continue
            walk_list.append(node_ids)
            available_nodes |= set(node_ids)
        return walk_list, available_nodes

    def input(self):
        return self.dataset


class NodeInput(Input):
    pass


def model_word2vec_skipgram():
    pass


def model_word2vec_cbow(mode, size_of_vocabulary: int, loss: str):
    labels = None
    embeddings = None
    ids = None

    # Look up embedding for all words
    embedded_words = tf.nn.embedding_lookup(embeddings, ids)
    # Concatenate vectors
    concatenated_embeddings = embedded_words
    # Transform embeddings linearly
    hidden_layer = tf.layers.dense(inputs=concatenated_embeddings, units=size_of_vocabulary, activation=None,
                                   use_bias=True,
                                   name="linear_transformation",
                                   kernel_initializer=tf.random_uniform_initializer(minval=-1, maxval=1))

    # Apply softmax and calulate loss
    if loss == 'cross_entropy':
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hidden_layer, labels=labels),
                              name="cross_entropy_loss")
    else:
        raise NotImplementedError("This loss isn't implemented yet. Implement it!")

    optimizer = tf.train.AdamOptimizer()
    total_train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    metrics = {'loss': loss}
    # Loss for tensorboard
    tf.summary.scalar('Loss', loss)

    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=total_train_op)
    else:
        raise NotImplementedError("This mode isn't implemented yet")


def model_paragraph_vectors_skipgram():
    pass


def model_paragraph_vectors_cbow():
    pass


def create_estimator():
    pass


def parse_arguments():
    pass


if __name__ == "__main__":
    args = parse_arguments()
    classifier = create_estimator()

    if args.mode == 'train':
        # TODO: Create Dataset
        dataset = tf.data.Dataset()
        classifier.train(input_fn=MetaPathsInput(dataset).input())
    elif args.mode == 'predict':
        raise NotImplementedError()
    elif args.mode == 'eval':
        raise NotImplementedError()
