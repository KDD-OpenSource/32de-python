from flask import Flask, jsonify, request, abort
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

#TODO CORS
def run(port, hostname, debug_mode):
    app.run(host=hostname, port=port, debug=debug_mode)


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
    # TODO: Check if necessary information is in request object
    if not request.json:
        abort(400)


@app.route("/next-meta-paths", methods=["GET"])
def send_next_metapaths_to_rate():
    # TODO: Call fitting method in active_learning
    return jsonify([{'id': 32,
                    'path': ['Phenotype', 'HAS', 'Association', 'HAS', 'SNP', 'HAS', 'Phenotype'],
                    'rating': 0.5}])


# TODO: Maybe post each rated meta-path
@app.route("/next-meta-paths", methods=["POST"])
def receive_rated_metapaths():
    # TODO: Check if necessary information is in request object
    if not request.json:
        abort(400)


@app.route("/results", methods=["GET"])
def send_results():
    # TODO: Call fitting method in explanation
    return jsonify("Hello world")

if __name__ == '__main__':
    app.run(port=8000)