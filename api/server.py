from flask import Flask, jsonify, request, abort, session
from flask_session import Session
from flask_cors import CORS
import json
import os
import time
import datetime
from flask_ask import Ask
import logging
from typing import Dict

from util.config import *
from util.meta_path_loader_dispatcher import MetaPathLoaderDispatcher
from util.graph_stats import GraphStats
from active_learning.active_learner import UncertaintySamplingAlgorithm
from explanation.explanation import SimilarityScore, Explanation

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

# TODO: Fix CORS origins specification
# Configure Cross Site Scripting
if "METAEXP_DEV" in os.environ.keys() and os.environ["METAEXP_DEV"] == "true":
    if REACT_PORT == 80:
        CORS(app, supports_credentials=True, resources={r"/*": {"origins": "http://{}".format(SERVER_PATH)}})
    else:
        CORS(app, supports_credentials=True,
             resources={r"/*": {"origins": "http://{}:{}".format(SERVER_PATH, REACT_PORT)}})
else:
    CORS(app, supports_credentials=True, resources={r"/*": {"origins": "*"}})


def run(port, hostname, debug_mode):
    app.run(host=hostname, port=port, debug=debug_mode, threaded=True)


@app.route('/login', methods=["POST"])
def login():
    session.clear()
    data = request.get_json()

    # retrieve data from login
    logger.debug("Login route received data: {}".format(data))
    session['username'] = data['username']
    session['dataset'] = data['dataset']
    session['purpose'] = data['purpose']

    # setup data
    # TODO use key from dataset to select data
    meta_path_loader = MetaPathLoaderDispatcher().get_loader(session['dataset'])
    meta_paths = meta_path_loader.load_meta_paths()
    # TODO get Graph stats for current dataset
    graph_stats = GraphStats()
    session['active_learning_algorithm'] = UncertaintySamplingAlgorithm(meta_paths, hypothesis='Gaussian Process')
    session['meta_path_id'] = 1
    session['rated_meta_paths'] = []
    # TODO feed this selection to the ALgorithms
    session['selected_node_types'] = build_selection(graph_stats.get_node_types())
    session['selected_edge_types'] = build_selection(graph_stats.get_edge_types())

    return jsonify({'status': 200})


@app.route('/logout')
def logout():
    rated_meta_paths = {
        'meta_paths': session['active_learning_algorithm'].create_output(),
        'dataset': session['dataset'],
        'node_type_selection': session['selected_node_types'],
        'edge_type_selection': session['selected_edge_types'],
        'username': session['username'],
        'purpose': session['purpose']
    }
    filename = '{}_{}_{}.json'.format(session['dataset'], session['username'], time.time())
    logger.info("Writing results to file {}...".format(filename))
    path = os.path.join(RATED_DATASETS_PATH, filename)
    json.dump(rated_meta_paths, open(path, "w", encoding="utf8"))
    session.clear()
    return 'OK'


# TODO: If functionality "meta-paths for node set A and B" will be written in Java, team alpha will need this information in Java
@app.route("/node-sets", methods=["POST"])
def receive_node_sets():
    """
    Receives the node sets from the "Setup" page which the user selects.
    This endpoint is called for each new added node.

    The repeated calling enables us to start the following computations as early as possible
    so that we can return information on the next pages faster.
    For example on the first call we already know the type of the whole node set and
    therefore can begin to retrieve the corresponding node sets.
    """
    # TODO: Check if necessary information is in request object
    if not request.json:
        abort(400)
    raise NotImplementedError("This API endpoint isn't implemented in the moment")


@app.route("/first-node -set-query", methods=["GET"])
def send_first_node_set():
    """
    :return: first node set as a cypher query as input for the neo4jGraphRenderer
    TODO: Build query dynamically depending on selected node-IDs
    """
    return jsonify({'node_set_query': 'MATCH (n)-[r]->(m) RETURN n,r,m'})


@app.route("/second-node-set-query", methods=["GET"])
def send_second_node_set():
    """
    :return: second node set as a cypher query as input for the neo4jGraphRenderer
    TODO: Build query dynamically depending on selected node-IDs
    """
    return jsonify({'node_set_query': 'MATCH (n)-[r]->(m) RETURN n,r,m'})


@app.route("/set-edge-types", methods=["POST"])
def receive_edge_types():
    """
    Receives edge types which are selected on the Config page
    """

    # TODO: Check if necessary information is in request object
    if not request.json:
        abort(400)

    edge_types = request.get_json()
    session['selected_edge_types'] = edge_types

    return jsonify({'edge_types': edge_types})


@app.route("/set-node-types", methods=["POST"])
def receive_node_types():
    """
    Receives node types which are selected on the Config page
    """

    # TODO: Check if necessary information is in request object
    if not request.json:
        abort(400)

    node_types = request.get_json()
    session['selected_node_types'] = node_types

    return jsonify({'node_types': node_types})


@app.route("/get-edge-types", methods=["GET"])
def send_edge_types():
    """
    :return: Array of available edge types for the Config page
    """
    return jsonify(session['selected_edge_types'])


@app.route("/get-node-types", methods=["GET"])
def send_node_types():
    """
    :return: Array of available node types for the Config page
    """
    return jsonify(session['selected_node_types'])


def build_selection(types):
    return [(element, True) for element in types]


@app.route("/next-meta-paths/<int:batch_size>", methods=["GET"])
def send_next_metapaths_to_rate(batch_size):
    """
        Returns the next `batchsize` meta-paths to rate.

        Metapaths are formatted like this:
        {'id': 3,
        'metapath': ['Phenotype', 'HAS', 'Association', 'HAS', 'SNP', 'HAS', 'Phenotype'],
        'rating': 0.5}
    """
    next_metapaths, is_last_batch, reference_paths = session['active_learning_algorithm'].get_next(batch_size=batch_size)

    for i in range(len(next_metapaths)):
        next_metapaths[i]['metapath'] = next_metapaths[i]['metapath'].as_list()

    paths = {'meta_paths': next_metapaths,
             'next_batch_available': not is_last_batch}
    if reference_paths:
        logger.info("Appending reference paths to response...")
        min_path = reference_paths['min_path']
        max_path = reference_paths['max_path']
        paths['min_path']= min_path
        paths['max_path']= max_path

    logger.debug("Responding to server: {}".format(paths))
    if "time" in session.keys():
        session['time_old'] = session['time']
    session['time'] = datetime.datetime.now()

    return jsonify(paths)


@app.route("/save-new-dataset", methods=["POST"])
def add_new_dataset():
    if not request.json:
        abort(400)

    data = request.get_json()

    if 'url' not in data or 'name' not in data or 'username' not in data or 'password' not in data:
        abort(422)

    return jsonify({'status': 200})


@app.route("/get-available-datasets", methods=["GET"])
def get_available_datasets():
    """
        Returns all data sets registered on the server and a short description of each
    """
    return jsonify(MetaPathLoaderDispatcher().get_available_datasets())


def transform_rating(data:Dict) -> Dict:
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

    return 'OK'


@app.route("/get-similarity-score", methods=["GET"])
def send_similarity_score():
    """
    :return: float, that is a similarity score between both node sets
    """
    similarity_score = SimilarityScore()
    return jsonify({'similarity_score': similarity_score.get_similarity_score()})


@app.route("/contributing-meta-paths", methods=["GET"])
def send_contributing_meta_paths():
    """
    :return: Array of dictionaries, that hold necessary information for a pie chart visualization
            about k-most contributing meta-paths to overall similarity score
    """
    similarity_score = SimilarityScore()
    return jsonify({'contributing_meta_paths': similarity_score.get_contributing_meta_paths()})


@app.route("/contributing-meta-path/<int:meta_path_id>", methods=["GET"])
def send_contributing_meta_path(meta_path_id):
    """
    :param meta_path_id: Integer, that is a unique identifier for a meta-path
    :return: Dictionary, that holds detailed information about the belonging meta-path
    """

    similarity_score = SimilarityScore()
    return jsonify({'meta_path': similarity_score.get_contributing_meta_path(meta_path_id)})


@app.route("/similar-nodes", methods=["GET"])
def send_similar_nodes():
    """
    :return: Array of dictionaries, that hold a 1-neighborhood query and properties about
             k-similar nodes regarding both node sets
    """

    explanation = Explanation()
    return jsonify({'similar_nodes': explanation.get_similar_nodes()})


# Self defined intents
@ask.intent('ChooseDataset')
def choose_dataset(dataset):
    raise NotImplementedError()


@ask.intent('RateMetapath')
def rate_metapath():
    raise NotImplementedError()


@ask.intent('ExcludeEdgeType')
def exclude_edge_type():
    raise NotImplementedError()


@ask.intent('ExcludeNodeType')
def exclude_node_type():
    raise NotImplementedError()


@ask.intent('ShowMoreMetapaths')
def show_more_metapaths():
    raise NotImplementedError()


@ask.intent('ShowResults')
def show_results():
    raise NotImplementedError()


# Built-in intents
@ask.intent('AMAZON.CancelIntent')
def cancel():
    raise NotImplementedError()


@ask.intent('AMAZON.HelpIntent')
def help():
    raise NotImplementedError()


@ask.intent('AMAZON.StopIntent')
def stop():
    raise NotImplementedError()


@ask.intent('AMAZON.MoreIntent')
def more():
    raise NotImplementedError()


@ask.intent('AMAZON.NavigateHomeIntent')
def navigate_home():
    raise NotImplementedError()


@ask.intent('AMAZON.NavigateSettingsIntent')
def navigate_settings():
    raise NotImplementedError()


@ask.intent('AMAZON.NextIntent')
def next():
    raise NotImplementedError()


@ask.intent('AMAZON.PageUpIntent')
def page_up():
    raise NotImplementedError()


@ask.intent('AMAZON.PageDownIntent')
def page_down():
    raise NotImplementedError()


@ask.intent('AMAZON.PreviousIntent')
def previous():
    raise NotImplementedError()


@ask.intent('AMAZON.ScrollRighIntent')
def scroll_right():
    raise NotImplementedError()


@ask.intent('AMAZON.ScrollDownIntent')
def scroll_down():
    raise NotImplementedError()


@ask.intent('AMAZON.ScrollLeftIntent')
def scroll_left():
    raise NotImplementedError()


@ask.intent('AMAZON.ScrollUpIntent')
def scroll_up():
    raise NotImplementedError()


if __name__ == '__main__':
    app.run(port=API_PORT, threaded=True, debug=True)
