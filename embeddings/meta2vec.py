import tensorflow as tf
import argparse
import json
from typing import List, Tuple
from embeddings.input import *


# def calculate_embeddings(meta_paths: List[List[str]]) -> List[Tuple(List[str], List[float])]:
#    pass

def model_word2vec(features, labels, mode, params):
    """
    Word2vec model from "Efficient Estimation of Word Representations in Vector Space" (Mikolov et al.)
    TODO: Add documentation
    :return:
    """
    assert features is not None, 'features is {}'.format(features)
    assert labels is not None, 'labels is {}'.format(labels)
    input = tf.feature_column.input_layer(features, params['feature_columns'])
    size_of_vocabulary = input.shape[1].value

    word_embeddings = tf.Variable(
        initial_value=tf.random_uniform(shape=[size_of_vocabulary, params['embedding_size']], minval=-1, maxval=1),
        name='word_embeddings')

    # Look up embedding for all words
    embedded_words = tf.nn.embedding_lookup(word_embeddings, tf.argmax(input, axis=1))
    assert embedded_words.shape == (
        input.shape[0].value, params['embedding_size']), 'Shape expected ({}, {}), but was {}'.format(
        input.shape[0].value, params['embedding_size'], embedded_words.shape)

    # Concatenate vectors
    concatenated_embeddings = tf.reshape(tf.concat(tf.unstack(embedded_words, axis=0), axis=0), shape=[1, -1])
    assert concatenated_embeddings.shape == (
        1, embedded_words.shape[0].value * embedded_words.shape[1].value), 'Shape expected ({}), but was {}'.format(
        1, embedded_words.shape[0].value * embedded_words.shape[1].value, concatenated_embeddings.shape)

    return _model_word2vec(mode, size_of_vocabulary, params['loss'], labels, concatenated_embeddings, params['optimizer'])


def _model_word2vec(mode, size_of_vocabulary, loss: str, labels, concatenated_embeddings, optimizer: str):
    # Transform embeddings linearly
    hidden_layer = tf.layers.dense(inputs=concatenated_embeddings, units=size_of_vocabulary, activation=None,
                                   use_bias=True,
                                   name="linear_transformation",
                                   kernel_initializer=tf.random_uniform_initializer(minval=-1, maxval=1))
    assert hidden_layer.shape == (
        1, size_of_vocabulary), 'Shape expected ({}, {}), but was {}'.format(
        1, size_of_vocabulary, hidden_layer.shape)

    # Apply softmax and calulate loss
    if loss == 'cross_entropy':
        # TODO: Reduce to one vector when using skip-grams (so shape is (1, 14) and not (1, 56))
        labels = tf.reshape(tf.one_hot(indices=labels, depth=size_of_vocabulary), shape=[1, -1])
        assert labels.shape == (
            1, size_of_vocabulary), 'Shape expected ({}, {}), but was {}'.format(
            1, size_of_vocabulary, labels.shape)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hidden_layer, labels=labels),
                              name="cross_entropy_loss")
    else:
        raise NotImplementedError("This loss isn't implemented yet. Implement it!")

    total_train_op = __build_train_op(loss, optimizer)

    # Loss for tensorboard
    tf.summary.scalar('Loss', loss)

    metrics = {'loss': loss}
    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=total_train_op)
    else:
        raise NotImplementedError("This mode isn't implemented yet")


def __build_train_op(loss, optimizer):
    if optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer()
    elif optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer()
    else:
        raise NotImplementedError("This optimizer isn't implemented yet. Implement it!")
    total_train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return total_train_op


def model_paragraph_vectors_skipgram(features, labels, mode, params):
    """
    Distributed Memory Model of Paragraph Vectors (PV-DM) from "Distributed Representations of Sentences and Documents" (Mikolov et al.)
    :return:
    """
    assert features is not None, 'features is {}'.format(features)
    assert labels is not None, 'labels is {}'.format(labels)
    context = tf.feature_column.input_layer(features, params['feature_columns'])
    paragraph = tf.feature_column.input_layer(features, params['paragraph_columns'])
    size_of_node_vocabulary = context.shape[1].value
    size_of_paragraph_vocabulary = paragraph.shape[1].value

    node_embeddings = tf.Variable(
        initial_value=tf.random_uniform(shape=[size_of_node_vocabulary, params['embedding_size'][0]], minval=-1, maxval=1),
        name='word_embeddings')
    paragraph_embedding = tf.Variable(
        initial_value=tf.random_uniform(shape=[size_of_paragraph_vocabulary, params['embedding_size'][1]], minval=-1, maxval=1),
        name='paragraph_embeddings')

    # Look up embedding for all words
    embedded_words = tf.nn.embedding_lookup(node_embeddings, tf.argmax(context, axis=1))
    embedded_paragraph = tf.nn.embedding_lookup(paragraph_embedding, tf.argmax(paragraph, axis=1))
    assert embedded_words.shape == (
        context.shape[0].value, params['embedding_size'][0]), 'Shape expected ({}, {}), but was {}'.format(
        context.shape[0].value, params['embedding_size'][0], embedded_words.shape)
    assert embedded_paragraph.shape == (
        paragraph.shape[0].value, params['embedding_size'][1]), 'Shape expected ({}, {}), but was {}'.format(
        paragraph.shape[0].value, params['embedding_size'][1], embedded_paragraph.shape)

    # Concatenate vectors
    concatenated_embeddings = tf.reshape(tf.concat([embedded_paragraph, tf.unstack(embedded_words, axis=0)], axis=0), shape=[1, -1])
    assert concatenated_embeddings.shape == (
        1, embedded_words.shape[0].value * embedded_words.shape[1].value + embedded_paragraph.shape[0].value), 'Shape expected ({}), but was {}'.format(
        1, embedded_words.shape[0].value * embedded_words.shape[1].value + embedded_paragraph.shape[0].value, concatenated_embeddings.shape)

    return _model_word2vec(mode, size_of_node_vocabulary, params['loss'], labels, embedded_words, params['optimizer'])


def model_paragraph_vectors_dbow(features, labels, mode, params):
    """
    Distributed Bag of Words version of Paragraph Vector (PV-DBOW) from "Distributed Representations of Sentences and Documents" (Mikolov et al.)
    :return:
    """
    pass


def create_estimator(model_dir, model_fn, input: Input, embedding_size: int, loss: str, gpu_memory: float):
    features = tf.feature_column.categorical_column_with_hash_bucket('features',
                                                                     input.get_vocab_size(),
                                                                     dtype=tf.int32)
    indicator_column = tf.feature_column.indicator_column(features)
    session_config = tf.ConfigProto()
    session_config.gpu_options.per_process_gpu_memory_fraction = gpu_memory
    run_config = tf.estimator.RunConfig(tf_random_seed=42,
                                        save_summary_steps=500,
                                        save_checkpoints_steps=1000,
                                        keep_checkpoint_max=5,
                                        keep_checkpoint_every_n_hours=0.25,
                                        log_step_count_steps=50,
                                        session_config=session_config)
    classifier = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=model_dir,
        params={'feature_columns': [indicator_column],
                'embedding_size': embedding_size,
                'loss': loss},
        config=run_config)
    return classifier


def create_paragraph_estimator(model_dir, model_fn, input: Input, embedding_size: int, loss: str, gpu_memory: float):
    context = tf.feature_column.categorical_column_with_hash_bucket('features',
                                                                     input.get_vocab_size(),
                                                                     dtype=tf.int32)

    paragraph = tf.feature_column.categorical_column_with_hash_bucket('paragraphs',
                                                                     input.get_paths_count(),
                                                                     dtype=tf.int32)

    context_indicator = tf.feature_column.indicator_column(context)
    paragraph_indicator = tf.feature_column.indicator_column(paragraph)
    session_config = tf.ConfigProto()
    session_config.gpu_options.per_process_gpu_memory_fraction = gpu_memory
    run_config = tf.estimator.RunConfig(tf_random_seed=42,
                                        save_summary_steps=500,
                                        save_checkpoints_steps=1000,
                                        keep_checkpoint_max=5,
                                        keep_checkpoint_every_n_hours=0.25,
                                        log_step_count_steps=50,
                                        session_config=session_config)
    classifier = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=model_dir,
        params={'feature_columns': [context_indicator],
                'paragraph_columns': [paragraph_indicator],
                'embedding_size': embedding_size,
                'loss': loss},
        config=run_config)
    return classifier


def parse_arguments():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--mode",
                        choices=["train", "predict", "eval"],
                        default="train",
                        help="Specify what you want to do: 'train', 'predict' or 'eval'.")
    parser.add_argument('--input_type',
                        choices=["meta-paths", "node-types", "nodes"],
                        help='Choose input type of data in json',
                        type=str,
                        required=True)
    parser.add_argument('--json_path',
                        help='Specify path of json with input data',
                        type=str,
                        required=True)
    parser.add_argument('--model_dir',
                        help='Specify directory where checkpoints etc. should be saved',
                        type=str,
                        required=True)
    parser.add_argument('--model',
                        choices=["word2vec", "paragraph_vectors"],
                        help='Choose which model should be used',
                        type=str,
                        required=True)
    parser.add_argument('--model_type',
                        choices=["bag-of-words", "skip-gram"],
                        help='Choose which model type should be used',
                        type=str,
                        required=True)
    parser.add_argument('--embedding_size',
                        help='Specify the size of the node embedding, paragraph embedding or both.'
                             'If you want to specify both, you need to specify the node embedding first.',
                        type=int,
                        nargs='+',
                        required=True)
    parser.add_argument('--gpu_memory',
                        help='Specify amount of GPU memory this process is allowed to use',
                        type=float,
                        required=True)
    parser.add_argument('--loss',
                        choices=["cross_entropy"],
                        help='Choose which loss should be used',
                        type=str,
                        required=True)
    parser.add_argument('--optimizer',
                        choices=['adam', 'sgd'],
                        help='Choose which optimizer should be used',
                        type=str,
                        default='adam')
    return parser.parse_args()


def choose_function(model: str, model_type: str, input_type: str, json_path: str):
    json_file = open(json_path, mode='r')
    json_data = json.load(json_file)
    if input_type == "node-types":
        input = NodeEdgeTypeInput.from_json(json_data)
        if model_type == "bag-of-words":
            input_fn = input.bag_of_words_input
        elif model_type == "skip-gram":
            input_fn = input.skip_gram_input
    elif input_type == "nodes":
        input = NodeInput.from_json(json_data)
        if model_type == "bag-of-words":
            input_fn = input.bag_of_words_input
        elif model_type == "skip-gram":
            input_fn = input.skip_gram_input
    elif input_type == "meta-paths":
        input = MetaPathsInput.from_json(json_data)
        if model_type == "bag-of-words":
            input_fn = input.skip_gram_input
        elif model_type == "skip-gram":
            input_fn = input.bag_of_words_input

    if model == "word2vec":
        model_fn = model_word2vec
    elif model == "paragraph_vectors":
        if model_type == "bag-of-words":
            model_fn = model_paragraph_vectors_dbow
        elif model_type == "skip-gram":
            model_fn = model_paragraph_vectors_skipgram
    return input, model_fn, input_fn


if __name__ == "__main__":
    args = parse_arguments()
    input, model_fn, input_fn = choose_function(model=args.model, model_type=args.model_type,
                                                input_type=args.input_type,
                                                json_path=args.json_path)
    print("chose function")
    if args.model == "word2vec":
        classifier = create_estimator(model_dir=args.model_dir, model_fn=model_fn, input=input,
                                  embedding_size=args.embedding_size, loss=args.loss, gpu_memory=args.gpu_memory)
    elif args.model == "paragraph_vectors":
        classifier = create_paragraph_estimator(model_dir=args.model_dir, model_fn=model_fn, input=input,
                                      embedding_size=args.embedding_size, loss=args.loss, gpu_memory=args.gpu_memory)

    print("Created estimator")
    if args.mode == 'train':
        print("Training")
        classifier.train(input_fn=input_fn)
    elif args.mode == 'predict':
        raise NotImplementedError()
    elif args.mode == 'eval':
        raise NotImplementedError()
