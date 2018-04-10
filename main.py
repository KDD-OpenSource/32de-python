import active_learning.active_learner as active_learning
import domain_scoring.domain_scoring as domain_scoring
import explanation.explanation as explanation
from api import server
import argparse
import util.config as config


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",
                        choices=[config.BASELINE_MODE, config.RESEARCH_MODE],
                        help="Run the baseline or our research prototype")
    parser.add_argument("--port",
                        type=int,
                        default=5000,
                        help="Port the flask server should listen on")
    parser.add_argument("--host",
                        type=str,
                        default='127.0.0.1',
                        help="Hostname the flask server should listen on")
    parser.add_argument("--debug",
                        action="store_true",
                        help="Debug mode of flask server")

    return parser.parse_args()


if __name__ == '__main__':
    print("Starting system...")

    args = parse_arguments()

    server.run(port=args.port, hostname=args.host, debug_mode=args.debug)