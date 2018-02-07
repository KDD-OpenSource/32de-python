from flask import Flask, jsonify, request, abort, session
from flask_cors import CORS
from flask.ext.session import Session
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
app.config["SECRET_KEY"] = "grgrersg346879468"
Session(app)

CORS(app, resources={r"/*": {"origins": "http://localhost:{}".format(REACT_PORT)}})


def run(port, hostname, debug_mode):
    app.run(host=hostname, port=port, debug=debug_mode)


# TODO: If meta-paths for A and B will be written in Java, they will need this information in Java
@app.route("/node-sets", methods=["POST"])
def receive_node_sets():
    if 'id' not in session.keys():
        # TODO: Create user id
        pass
    if 'node-sets' in session.keys():
        # TODO: Append necessary information to session entry
        id_following_node_set = 0
        session['node-sets'] = session['node-sets'].append(id_following_node_set)
    else:
        # TODO: Add necessary information to first session entry
        id_first_node_set = 0
        session['node-sets'] = id_first_node_set
    # TODO: Check if necessary information is in request object
    if not request.json:
        abort(400)


@app.route("/node-sets", methods=["GET"])
def send_node_sets():
    # TODO: See lines of 'receive_node_sets()' regarding session for how to use the session variables
    # TODO: Call fitting method in active_learning
    if 'id' not in session.keys():
        # TODO: Create user id
        session['id'] = time.time()
    if 'node-sets' in session.keys():
        # TODO: Append necessary information to session entry
        id_following_node_set = 1
        session['node-sets'] = session['node-sets'] + (id_following_node_set)
    else:
        # TODO: Add necessary information to first session entry
        id_first_node_set = 0
        session['node-sets'] = id_first_node_set
    # TODO: Check if necessary information is in request object
    return jsonify("Hello world")


# TODO: If meta-paths for A and B will be written in Java, they will need this information in Java
@app.route("/types", methods=["POST"])
def receive_edge_node_types():
    # TODO: See lines of 'receive_node_sets()' regarding session for how to use the session variables
    # TODO: Check if necessary information is in request object
    if not request.json:
        abort(400)

mock_id = 1
@app.route("/next-meta-paths", methods=["GET"])
def send_next_metapaths_to_rate():
    # TODO: See lines of 'receive_node_sets()' regarding session for how to use the session variables
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
    # TODO: See lines of 'receive_node_sets()' regarding session for how to use the session variables
    # TODO: Check if necessary information is in request object
    if not request.json:
        abort(400)
    rated_metapaths = request.get_json()
    return 'OK'


@app.route("/results", methods=["GET"])
def send_results():
    # TODO: See lines of 'receive_node_sets()' regarding session for how to use the session variables
    # TODO: Call fitting method in explanation
    return jsonify("Hello world")


if __name__ == '__main__':
    app.run(port=API_PORT)
