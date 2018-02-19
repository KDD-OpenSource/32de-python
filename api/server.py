from flask import Flask, jsonify, request, abort, session
from flask_session import Session
from flask_cors import CORS
from util.config import REACT_PORT, API_PORT, SESSION_CACHE_DIR, SESSION_MODE, SESSION_THRESHOLD, RATED_DATASETS_PATH
from util.meta_path_loader_dispatcher import MetaPathLoaderDispatcher
from util.graph_stats import GraphStats
from active_learning.meta_path_selector import RandomMetaPathSelector
import json
import os
import time

app = Flask(__name__)

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

CORS(app, supports_credentials=True, resources={r"/*": {"origins": "http://localhost:{}".format(REACT_PORT)}})


def run(port, hostname, debug_mode):
    app.run(host=hostname, port=port, debug=debug_mode, threaded=True)

@app.route('/login', methods=["POST", "GET"])
def login():
    if request.method == 'POST' and 'username' not in session:
        data = request.get_json()

        # retrieve data from login
        print("Login route received data: {}".format(data))
        session['username'] = data['username']
        session['dataset'] = data['dataset']
        session['purpose'] = data['purpose']

        # setup dataset
        # TODO use key from dataset to select data
        meta_path_loader = MetaPathLoaderDispatcher().get_loader(session['dataset'])
        meta_paths = meta_path_loader.load_meta_paths()
        # TODO get Graph stats for current dataset
        graph_stats = GraphStats()
        session['meta_path_distributor'] = RandomMetaPathSelector(meta_paths=meta_paths)
        session['meta_path_id'] = 1
        session['rated_meta_paths'] = []
        session['selected_node_types'] = build_selection(graph_stats.get_node_types())
        session['selected_edge_types'] = build_selection(graph_stats.get_edge_types())

    return jsonify({'status': 200})


@app.route('/logout')
def logout():
    rated_meta_paths = {
        'meta_paths': session['rated_meta_paths'],
        'dataset': session['dataset'],
        'node_type_selection': session['selected_node_types'],
        'edge_type_selection': session['selected_edge_types'],
        'username': session['username'],
        'purpose': session['purpose']
    }
    filename = '{}_{}_{}.json'.format(session['dataset'], session['username'], time.time())
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


@app.route("/node-sets", methods=["GET"])
def send_node_sets():
    """
    Returns the node sets which the user previously selected on the "Setup" page.
    """
    # TODO: Does active_learning really needs this endpoint? Does someone needs this endpoint?
    # TODO: Call fitting method in active_learning
    # TODO: Check if necessary information is in request object
    raise NotImplementedError("This API endpoint isn't implemented in the moment")


# TODO: If functionality "meta-paths for node set A and B" will be written in Java, team alpha will need this information in Java
@app.route("/set-edge-types", methods=["POST"])
def receive_edge_types():
    """
    Receives the node and edge types which are selected (types which are active) on the "Config" page.
    """

    # TODO: Check if necessary information is in request object
    if not request.json:
        abort(400)

    edge_types = request.get_json()
    session['selected_edge_types'] = edge_types
    return 'OK'


# TODO: If functionality "meta-paths for node set A and B" will be written in Java, team alpha will need this information in Java
@app.route("/set-node-types", methods=["POST"])
def receive_node_types():
    """
    Receives the node and edge types which are selected (types which are active) on the "Config" page.
    """

    # TODO: Check if necessary information is in request object
    if not request.json:
        abort(400)

    node_types = request.get_json()
    session['selected_node_types'] = node_types
    return 'OK'


@app.route("/get-edge-types", methods=["GET"])
def send_edge_types():
    """
    Returns the available edge types for the "Config" page
    """
    return jsonify(session['selected_edge_types'])


@app.route("/get-node-types", methods=["GET"])
def send_node_types():
    """
    Returns the available node types for the "Config" page
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
    meta_path_id = session['meta_path_id']
    next_batch = session['meta_path_distributor'].get_next(size=batch_size)
    next_metapaths, last_batch = next_batch[0], next_batch[1]
    paths = {'meta_paths': [{'id': meta_id,
                             'metapath': meta_path.as_list(),
                             'rating': 0.5} for meta_id, meta_path in
                            zip(range(meta_path_id, meta_path_id + batch_size), next_metapaths)],
             'next_batch_available': not last_batch}

    meta_path_id += batch_size
    session['meta_path_id'] = meta_path_id
    return jsonify(paths)


@app.route("/get-available-datasets", methods=["GET"])
def get_available_datasets():
    """
        Deliver all available data sets for rating and a short description of each.
    """
    return jsonify(MetaPathLoaderDispatcher().get_available_datasets())


# TODO: Maybe post each rated meta-path
@app.route("/rate-meta-paths", methods=["POST"])
def receive_rated_metapaths():
    """
    Receives the rated meta-paths.

    Meta-paths are formated like this:
    {'id': 3,
    'metapath': ['Phenotype', 'HAS', 'Association', 'HAS', 'SNP', 'HAS', 'Phenotype'],
    'rating': 0.75}
    """
    # TODO: Check if necessary information is in request object
    if not request.is_json:
        abort(400)
    rated_metapaths = request.get_json()
    for datapoint in rated_metapaths:
        if not all(key in datapoint for key in ['id', 'metapath', 'rating']):
            abort(400)  # malformed input
    session['rated_meta_paths'] = session['rated_meta_paths'] + rated_metapaths
    return 'OK'


@app.route("/results", methods=["GET"])
def send_results():
    """
    TODO: Endpoint needs to be specified by team delta
    """
    # TODO: Call fitting method in explanation
    raise NotImplementedError("This API endpoint isn't implemented in the moment")


if __name__ == '__main__':
    app.run(port=API_PORT, threaded=True)
