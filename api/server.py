from flask import Flask, jsonify, request
from util.config import REACT_PORT

app = Flask(__name__)

whitelist = ['http://localhost:' + REACT_PORT]


def run(port, hostname, debug_mode):
    app.run(host=hostname, port=port, debug=debug_mode)


@app.route("/")
def hello_world():
    return jsonify({'world':"Hello world"})


@app.after_request
def add_cors_headers(response):
    if request.referrer is not None:
        r = request.referrer[:-1]
        if r in whitelist:
            response.headers.add('Access-Control-Allow-Origin', r)
            response.headers.add('Access-Control-Allow-Credentials', 'true')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
            response.headers.add('Access-Control-Allow-Headers', 'Cache-Control')
            response.headers.add('Access-Control-Allow-Headers', 'X-Requested-With')
            response.headers.add('Access-Control-Allow-Headers', 'Authorization')
            response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS, PUT, DELETE')
    return response