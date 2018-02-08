from flask import Flask, jsonify, request, abort, session
from flask_session import Session
from flask_cors import CORS
from util.config import REACT_PORT, API_PORT, SESSION_CACHE_DIR, SESSION_MODE, SESSION_THRESHOLD
from util.meta_path_loader import MetaPathLoaderDispatcher
from active_learning.meta_path_selector import RandomMetaPathSelector
import json
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

@app.route('/login', methods=["POST", "GET"])
def login():
    if request.method == 'POST':
        session['username'] = request.form['username']
        meta_path_loader = MetaPathLoaderDispatcher().get_loader("Rotten Tomato")
        meta_paths = meta_path_loader.load_meta_paths()
        session['meta_path_distributor'] = RandomMetaPathSelector(meta_paths=meta_paths)
        session['meta_path_id'] = 1
        print(session['username'])
        print(session['meta_path_distributor'])
    return '''
        <form method="post">
            <p><input type=text name=username>
            <p><input type=submit value=Login>
        </form>
    '''

@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('meta_path_distributer', None)


# TODO: If meta-paths for A and B will be written in Java, they will need this information in Java
@app.route("/node-sets", methods=["POST"])
def receive_node_sets():
    # TODO: Check if necessary information is in request object
    if not request.json:
        abort(400)


@app.route("/node-sets", methods=["GET"])
def send_node_sets():
    # TODO: Call fitting method in active_learning
    return jsonify("Hello world")


# TODO: If meta-paths for A and B will be written in Java, they will need this information in Java
@app.route("/types", methods=["POST"])
def receive_edge_node_types():
    # TODO: See lines of 'receive_node_sets()' regarding session for how to use the session variables
    # TODO: Check if necessary information is in request object
    if not request.json:
        abort(400)

@app.route("/next-meta-paths", methods=["GET"])
def send_next_metapaths_to_rate():
    # TODO: See lines of 'receive_node_sets()' regarding session for how to use the session variables
    # TODO: Check if necessary information is in request object
    batch_size = 5
    meta_path_id = session['meta_path_id']
    print(session['meta_path_distributor'])
    print(session['username'])
    next_batch = session['meta_path_distributor'].get_next(size=batch_size)
    paths = [{'id': id,
              'meta_path': meta_path.as_list(),
              'rating': 0.5} for id, meta_path in zip(range(meta_path_id, meta_path_id + batch_size), next_batch)]
    meta_path_id += batch_size
    return jsonify(paths)

@app.route("/get-available-datasets", methods=["GET"])
def get_available_datasets():
    return jsonify(MetaPathLoaderDispatcher().get_available_datasets())

# TODO: Maybe post each rated meta-path
@app.route("/rate-meta-paths", methods=["POST"])
def receive_rated_metapaths():
    if not request.is_json:
        abort(400)
    rated_metapaths = request.get_json()
    if not all(key in rated_metapaths for key in ['id', 'meta_path', 'rating']):
        abort(400)
    # TODO: BUGGGG
    json.dump('rated_data_{}.json'.format(session.get('username', 'fail')), rated_metapaths)
    return 'OK'


@app.route("/results", methods=["GET"])
def send_results():
    # TODO: See lines of 'receive_node_sets()' regarding session for how to use the session variables
    # TODO: Call fitting method in explanation
    return jsonify("Hello world")


if __name__ == '__main__':
    app.run(port=API_PORT)
