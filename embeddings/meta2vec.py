import argparse
import json
from typing import Tuple
import logging
logging.getLogger('MetaExp.meta2vec')

from embeddings.estimators import create_word2vec_estimator, create_paragraph_estimator
from embeddings.input import *
from embeddings.models import model_word2vec, model_paragraph_vectors_skipgram, model_paragraph_vectors_dbow


def calculate_metapath_embeddings(metapaths: List[List[int]], model_dir: str = './model_dir', gpu_memory: float = 0.3,
                                  loss: str = "cross_entropy", optimizer: str = "adam",
                                  metapath_embedding_size: int = None,
                                  node_embedding_size=4, model_type='skip-gram') -> List[Tuple[List[str], List[float]]]:
    """

    :param metapath_embedding_size:
    :param metapaths: The meta-paths to be embedded.
    :return: The embedding of the meta-paths in the same order as the given meta-paths.
             Every list represents a vector.
    """
    input = MetaPathsInput.from_paths_list(metapaths)

    if metapath_embedding_size is None:
        metapath_embedding_size = int(len(metapaths) / 100)  # TODO: there's some formula in the literatur

    model_fn = choose_model_function(model='paragraph_vectors', model_type=model_type)
    classifier = create_paragraph_estimator(model_dir=model_dir, model_fn=model_fn,
                                            node_count=input.get_vocab_size(), paths_count=input.paths_count(),
                                            sentence_embedding_size=metapath_embedding_size,
                                            word_embedding_size=node_embedding_size, optimizer=optimizer,
                                            loss=loss, gpu_memory=gpu_memory)

    logging.info("Beginning training...")
    classifier.train(input_fn=choose_input_function(input=input, model_type=model_type))
    logging.info("Finished training.")

    # Get trained embeddings
    trained_embeddings = classifier.get_variable_value(name='paragraph_embeddings')

    embedded_metapaths = []
    for id, metapath in enumerate(trained_embeddings):
        embedded_metapaths.append((metapaths[id], metapath.tolist()))

    logging.debug("Returning embedded meta-paths")
    return embedded_metapaths


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
    json_file.close()
    if input_type == "node-types":
        input = NodeEdgeTypeInput.from_json(json_data)
    elif input_type == "nodes":
        input = NodeInput.from_json(json_data)
    elif input_type == "meta-paths":
        input = MetaPathsInput.from_json(json_data)
    else:
        input = Input(paths=[], vocabulary=[])

    input_fn = choose_input_function(input, model_type)

    model_fn = choose_model_function(model, model_type)
    return input, model_fn, input_fn


def choose_input_function(input, model_type):
    if model_type == "bag-of-words":
        input_fn = input.bag_of_words_input
    elif model_type == "skip-gram":
        input_fn = input.skip_gram_input
    return input_fn


def choose_model_function(model, model_type):
    if model == "word2vec":
        model_fn = model_word2vec
    elif model == "paragraph_vectors":
        if model_type == "bag-of-words":
            model_fn = model_paragraph_vectors_dbow
        elif model_type == "skip-gram":
            model_fn = model_paragraph_vectors_skipgram
    return model_fn


if __name__ == "__main__":
    args = parse_arguments()
    input, model_fn, input_fn = choose_function(model=args.model, model_type=args.model_type,
                                                input_type=args.input_type,
                                                json_path=args.json_path)
    print("chose function")
    if args.model == "word2vec":
        classifier = create_word2vec_estimator(vocab_size=input.get_vocab_size(), model_fn=model_fn,
                                               model_dir=args.model_dir,
                                               embedding_size=args.embedding_size, loss=args.loss,
                                               gpu_memory=args.gpu_memory)
    elif args.model == "paragraph_vectors":
        classifier = create_paragraph_estimator(model_dir=args.model_dir, model_fn=model_fn,
                                                node_count=input.get_vocab_size(), paths_count=input.paths_count(),
                                                sentence_embedding_size=args.embedding_size[1],
                                                word_embedding_size=args.embedding_size[0], optimizer=args.optimizer,
                                                loss=args.loss, gpu_memory=args.gpu_memory)

    print("Created estimator")
    if args.mode == 'train':
        print("Training")
        classifier.train(input_fn=input_fn)
    elif args.mode == 'predict':
        raise NotImplementedError("Predict mode isn't implemented")
    elif args.mode == 'eval':
        raise NotImplementedError("Evaluate mode isn't implemented")
