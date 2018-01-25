from flask import Flask, jsonify

app = Flask(__name__)


def run(port, hostname, debug_mode):
    app.run(host=hostname, port=port, debug=debug_mode)


@app.route("/")
def hello_world():
    return jsonify("Hello world")
