from flask import Flask, jsonify, request, abort
from flask_cors import CORS
from util.config import REACT_PORT, API_PORT

app = Flask(__name__)
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


@app.route("/node-sets", methods=["GET"])
def send_node_sets():
    """
    Endpoint where the node sets which the user previously selected on the "Setup" page can be retrieved.
    """
    # TODO: Does active_learning really needs this endpoint? Does someone needs this endpoint?
    # TODO: Call fitting method in active_learning
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
    return jsonify("Hello world")


if __name__ == '__main__':
    app.run(port=API_PORT)
