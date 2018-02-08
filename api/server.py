from flask import Flask, jsonify, request, abort, session
from flask_cors import CORS
from flask_session import Session
from util.config import REACT_PORT, API_PORT
from util.config import SESSION_CACHE_DIR, SESSION_MODE, SESSION_THRESHOLD
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

CORS(app, resources={r"/*": {"origins": "http://localhost:{}".format(REACT_PORT)}})


def run(port, hostname, debug_mode):
    app.run(host=hostname, port=port, debug=debug_mode)


# TODO: If functionality "meta-paths for node set A and B" will be written in Java, team alpha will need this information in Java
@app.route("/node-sets", methods=["POST"])
def receive_node_sets():
    """
    Endpoint where the node sets from the "Setup" page which the user selects are posted.
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
    Endpoint where the node sets which the user previously selected on the "Setup" page can be retrieved.
    """
    # TODO: Does active_learning really needs this endpoint? Does someone needs this endpoint?
    # TODO: Call fitting method in active_learning
    # TODO: Check if necessary information is in request object
    raise NotImplementedError("This API endpoint isn't implemented in the moment")
    return jsonify("Hello world")


# TODO: If functionality "meta-paths for node set A and B" will be written in Java, team alpha will need this information in Java
@app.route("/types", methods=["POST"])
def receive_edge_node_types():
    """
    Endpoint where the node and edge types which are selected (types which are active) on the "Config" page are posted
    """
    # TODO: Check if necessary information is in request object
    if not request.json:
        abort(400)
    raise NotImplementedError("This API endpoint isn't implemented in the moment")

mock_id = 1
@app.route("/next-meta-paths", methods=["GET"])
def send_next_metapaths_to_rate():
    """
    Endpoint which returns the next `batchsize` meta-paths to rate.

    Metapaths are formated like this:
    {'id': 3,
    'path': ['Phenotype', 'HAS', 'Association', 'HAS', 'SNP', 'HAS', 'Phenotype'],
    'rating': 0.6}
    """
    # TODO: Check if necessary information is in request object
    global mock_id
    batchsize = 5
    paths = [{'id': i,
              'path': ['Phenotype', 'HAS', 'Association', 'HAS', 'SNP', 'HAS', 'Phenotype'],
              'rating': 0.5} for i in range(mock_id, mock_id + batchsize)]
    mock_id += batchsize
    # TODO: Call fitting method in active_learning
    return jsonify(paths)


# TODO: Maybe post each rated meta-path
@app.route("/rate-meta-paths", methods=["POST"])
def receive_rated_metapaths():
    """
    Endpoint where the rated meta-paths are posted.

    Meta-paths are formated like this:
    {'id': 3,
    'rating': 0.75}
    """
    # TODO: Check if necessary information is in request object
    if not request.json:
        abort(400)
    rated_metapaths = request.get_json()
    return 'OK'


@app.route("/results", methods=["GET"])
def send_results():
    """
    TODO: Endpoint needs to be specified by team delta
    """
    # TODO: Call fitting method in explanation
    raise NotImplementedError("This API endpoint isn't implemented in the moment")
    return jsonify("Hello world")


if __name__ == '__main__':
    app.run(port=API_PORT)
