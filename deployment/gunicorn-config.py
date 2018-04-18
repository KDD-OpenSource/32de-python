import os

bind = "0.0.0.0:8000"
if not os.path.exists("./log"):
    os.mkdir("./log")
# Only for development purpose
if "METAEXP_DEV" in os.environ.keys() and os.environ["METAEXP_DEV"] == "true":
    accesslog = "./log/access.log"
    errorlog = "./log/errors.log"
    loglevel = "debug"

if "METAEXP_HTTPS" in os.environ.keys() and os.environ["METAEXP_HTTPS"] == "true":
    certfile = "/32de-python/https/api.crt"
    keyfile = "/32de-python/https/api.key"
timeout = 99999
workers = 36
