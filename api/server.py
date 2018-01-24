from flask import Flask, jsonify, request, abort

app = Flask(__name__)


def run(port, hostname, debug_mode):
    app.run(host=hostname, port=port, debug=debug_mode)


@app.route("/node-sets", options="POST")
def receive_node_sets():
    # TODO: Check if necessary information is in request object
    if not request.json:
        abort(400)


@app.route("/types", options="POST")
def receive_edge_node_types():
    # TODO: Check if necessary information is in request object
    if not request.json:
        abort(400)


@app.route("/next-meta-paths", options="GET")
def send_next_metapaths_to_rate():
    # TODO: Call fitting method in active_learning
    return jsonify("Hello world")


@app.route("/next-meta-paths", options="POST")
def receive_rated_metapaths():
    # TODO: Check if necessary information is in request object
    if not request.json:
        abort(400)


@app.route("/results", options="GET")
def send_results():
    # TODO: Call fitting method in explanation
    return jsonify("Hello world")
