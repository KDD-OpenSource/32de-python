from flask import Flask, jsonify, request, abort, session
from flask_session import Session
from flask_cors import CORS
import json
import os
import time
import datetime
from flask_ask import Ask, statement
import logging
from typing import Dict

import embeddings.meta2vec
from util.config import *
from active_learning.active_learner import UncertaintySamplingAlgorithm
from explanation.explanation import SimilarityScore, Explanation
from api.redis_own import Redis
from util.metapaths_database_importer import RedisImporter


app = Flask(__name__)
ask = Ask(app, '/alexa')
set_up_logger()
logger = logging.getLogger('MetaExp.Server')

# TODO: Change if we have a database in the background
SESSION_TYPE = 'filesystem'
SESSION_FILE_DIR = SESSION_CACHE_DIR
SESSION_FILE_THRESHOLD = SESSION_THRESHOLD
# We need leading zeros on the modifier
SESSION_FILE_MODE = int(SESSION_MODE, 8)
SESSION_PERMANENT = True
app.config.from_object(__name__)
# TODO: Change for deployment, e.g. use environment variable
app.config["SECRET_KEY"] = "37Y,=i9.,U3RxTx92@9j9Z[}"
Session(app)

CORS(app, supports_credentials=True, resources={r"/*": {
    "origins": ["https://hpi.de/mueller/metaexp-demo-api/", "http://172.20.14.22:3000", "http://localhost",
                "http://localhost:3000", "http://metaexp.herokuapp.com"]}})


def run(port, hostname, debug_mode):
    app.run(host=hostname, port=port, debug=debug_mode, threaded=True)


@app.route('/redis-import', methods=['GET'])
def redis_import():
    RedisImporter(enable_existence_check=True).import_all()
    return jsonify({'status': 200})


@app.route('/test-import', methods=['GET'])
def test_import():
    RedisImporter(enable_existence_check=False).import_data_set(
        {'name': 'Helmholtz', 'bolt-url': 'bolt://172.20.14.22:7697', 'username': 'neo4j',
         'password': ''})
    return jsonify({'status': 200})


@app.route('/train-embeddings/<string:database>', methods=['GET'])
def train_embedding(database):
    redis = Redis(database)
    logger.debug("Start computation of embeddings...")
    meta_path_list_embeddings = embeddings.meta2vec.calculate_metapath_embeddings(redis.get_all_meta_paths(),
                                                                                  metapath_embedding_size=30)
    logger.debug("Received embeddings {}".format(meta_path_list_embeddings))
    redis.store_embeddings(meta_path_list_embeddings)
    return jsonify({'status': 200})


@app.route('/login', methods=["POST"])
def login():
    session.clear()
    data = request.get_json()

    # retrieve data from login
    logger.debug("Login route received data: {}".format(data))
    session['username'] = data['username']
    session['dataset'] = data['dataset']
    session['purpose'] = data['purpose']
    chosen_dataset = None
    for dataset in AVAILABLE_DATA_SETS:
        if dataset['name'] == session['dataset']:
            chosen_dataset = dataset
    if not chosen_dataset:
        logger.error('Dataset {} not available'.format(data['dataset']))
    session['dataset'] = chosen_dataset

    # setup data
    session['meta_path_id'] = 1
    session['rated_meta_paths'] = []

    redis = Redis(session['dataset']['name'])

    # TODO feed this selection to the ALgorithms
    session['selected_node_types'] = [(node_type.decode(), True) for node_type in redis.node_type_to_id_map().keys()]
    session['selected_edge_types'] = [(edge_type.decode(), True) for edge_type in redis.edge_type_to_id_map().keys()]
    logger.debug(session)
    return jsonify({'status': 200})


@app.route('/logout')
def logout():
    rated_meta_paths = {
        'meta_paths': session['active_learning_algorithm'].create_output(),
        'dataset': session['dataset']['name'],
        'node_type_selection': session['selected_node_types'],
        'edge_type_selection': session['selected_edge_types'],
        'username': session['username'],
        'purpose': session['purpose']
    }
    filename = '{}_{}_{}.json'.format(session['dataset']['name'], session['username'], time.time())
    logger.info("Writing results to file {}...".format(filename))
    path = os.path.join(RATED_DATASETS_PATH, filename)
    json.dump(rated_meta_paths, open(path, "w", encoding="utf8"))
    session.clear()
    return jsonify({'status': 200})


@app.route("/stop-rating", methods=["GET"])
def stop_meta_path_rating():
    session['similarity_score'].refresh()

    return jsonify({'status': 200})


@app.route("/node-types", methods=["POST"])
def receive_meta_path_start_and_end_label():
    redis = Redis(session['dataset']['name'])

    json_response = request.get_json()
    start_type = json_response['start_label']
    end_type = json_response['end_label']
    start_node_ids = json_response['start_node_ids']
    end_node_ids = json_response['end_node_ids']

    logger.debug("Recieved following meta-paths from redis: {}".format(redis.meta_paths(start_type, end_type)))
    session['active_learning_algorithm'] = UncertaintySamplingAlgorithm(
        redis.meta_paths(start_type, end_type),
        hypothesis='Gaussian Process')
    session['similarity_score'] = SimilarityScore(session['active_learning_algorithm'].get_complete_rating,
                                                  session['dataset'],
                                                  start_node_ids,
                                                  end_node_ids)

    return jsonify({'status': 200})


@app.route("/next-meta-paths/<int:batch_size>", methods=["GET"])
def send_next_metapaths_to_rate(batch_size):
    """
        Returns the next `batchsize` meta-paths to rate.

        Metapaths are formatted like this:
        {'id': 3,
        'metapath': ['Phenotype', 'HAS', 'Association', 'HAS', 'SNP', 'HAS', 'Phenotype'],
        'rating': 0.5}
    """

    next_metapaths, is_last_batch, reference_paths = session['active_learning_algorithm'].get_next(
        batch_size=batch_size)
    logger.debug("Received meta paths from active learner {}".format(next_metapaths))

    paths = {'meta_paths': next_metapaths,
             'next_batch_available': not is_last_batch}
    if reference_paths:
        logger.info("Appending reference paths to response...")
        paths['min_path'] = reference_paths['min_path']
        paths['max_path'] = reference_paths['max_path']

    logger.debug("Responding to server: {}".format(paths))
    if "time" in session.keys():
        session['time_old'] = session['time']
    session['time'] = datetime.datetime.now()

    return jsonify(paths)


@app.route("/get-available-datasets", methods=["GET"])
def get_available_datasets():
    """
    :return:  all data sets registered on the server and a dataset access properties of each
    """

    return jsonify(AVAILABLE_DATA_SETS)


def transform_rating(data: Dict) -> Dict:
    logger.info("Transforming ratings")

    new_min_path_rating = data['min_path']['rating']
    new_max_path_rating = data['max_path']['rating']
    if new_max_path_rating < new_min_path_rating:
        logger.error("The modified rating of the min_path must always be smaller then the one of max_path!")
        abort(400)
    # Extract ids of meta_paths, which received a smaller rating than min_path
    new_min_paths = [mp for mp in data['meta_paths'] if mp['rating'] < new_min_path_rating]
    logger.debug("Found meta paths, which are rated less than the min path: {}".format(new_min_paths))
    # Extract ids of meta_pats, which received a higher rating than max_path
    new_max_paths = [mp for mp in data['meta_paths'] if mp['rating'] > new_max_path_rating]
    logger.debug("Found meta paths, which are rated better than the max path: {}".format(new_max_paths))
    # Transform rating of new_min_paths meta paths
    for min_path in new_min_paths:
        rating_diff_to_min_path = abs(min_path['rating'] - data['min_path']['rating'])
        logger.debug(rating_diff_to_min_path)
        min_ref_path = session['active_learning_algorithm'].get_min_ref_path()
        logger.debug(min_ref_path)
        min_path['rating'] = min_ref_path['rating'] - rating_diff_to_min_path

    # Transform rating of new_max_paths meta paths
    for max_path in new_max_paths:
        rating_diff_to_min_path = abs(max_path['rating'] - data['max_path']['rating'])
        logger.debug(rating_diff_to_min_path)
        min_ref_path = session['active_learning_algorithm'].get_max_ref_path()
        logger.debug(min_ref_path)
        max_path['rating'] = min_ref_path['rating'] + rating_diff_to_min_path

    logger.debug("Rating was transformed: {}".format(data['meta_paths']))
    return data


# TODO: Maybe post each rated meta-path
@app.route("/rate-meta-paths", methods=["POST"])
def receive_rated_metapaths():
    """
    Receives the rated meta-paths.

    Format:
    'meta_paths': [{'id': 3,
                   'metapath': ['Phenotype', 'HAS', 'Association', 'HAS', 'SNP', 'HAS', 'Phenotype'],
                   'rating': 0.75},...]
    'min_path':{}
    'max_path':{}
    """
    time_results_received = datetime.datetime.now()
    if not request.is_json:
        logger.error("Aborting, because request is not in json format")
        abort(400)

    data = request.get_json()
    logger.debug("Login route received data: {}".format(data))

    expected_keys = ['id', 'metapath', 'rating']
    for datapoint in data['meta_paths']:
        if not all(key in datapoint for key in expected_keys):
            logger.error("Aborting, because keys {} are misssing in this part of json: {}".format(
                [key for key in expected_keys if key not in datapoint], datapoint))
            abort(400)

    if not session['active_learning_algorithm'].is_first_batch():
        data = transform_rating(data)

    logger.info("Updating active learning algorithm...")
    session['active_learning_algorithm'].update(data['meta_paths'])
    if "time_old" in session.keys():
        data['time_to_rate'] = (time_results_received - session['time_old']).total_seconds()
    else:
        if "time" in session.keys():
            data['time_to_rate'] = (time_results_received - session['time']).total_seconds()

    return jsonify({'status': 200})


@app.route("/get-similarity-score", methods=["GET"])
def send_similarity_score():
    """
    :return: float, that is a similarity score between both node sets
    """
    return jsonify({'similarity_score': session['similarity_score'].get_similarity_score()})


@app.route("/contributing-meta-paths", methods=["GET"])
def send_contributing_meta_paths():
    """
    :return: Array of dictionaries, that hold necessary information for a pie chart visualization
            about k-most contributing meta-paths to overall similarity score
    """
    return jsonify({'contributing_meta_paths': session['similarity_score'].get_contributing_meta_paths()})


@app.route("/contributing-meta-path/<int:meta_path_id>", methods=["GET"])
def send_contributing_meta_path(meta_path_id):
    """
    :param meta_path_id: Integer, that is a unique identifier for a meta-path
    :return: Dictionary, that holds detailed information about the belonging meta-path
    """
    return jsonify({'meta_path': session['similarity_score'].get_contributing_meta_path(meta_path_id)})


@app.route("/similar-nodes", methods=["GET"])
def send_similar_nodes():
    """
    :return: Array of dictionaries, that hold a 1-neighborhood query and properties about
             k-similar nodes regarding both node sets
    """

    explanation = Explanation()
    return jsonify({'similar_nodes': explanation.get_similar_nodes()})

if __name__ == '__main__':
    app.run(port=API_PORT, threaded=True, debug=True)
