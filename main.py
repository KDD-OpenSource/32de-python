from active_learning.active_learning import ActiveLearner
from domain_scoring.domain_scoring import DomainScoring
from explanation.explanation import Explanation
import argparse

RESEARCH_MODE = "research"
BASELINE_MODE = "baseline"

if __name__ == '__main__':
    print("Starting system...")

    args = parse_arguments()

    # beta: active learning of relevant meta-paths
    active_learner = ActiveLearner()
    rated_paths = active_learner.retrieve_user_rating()

    # gamma: score the learned paths
    domain_score = DomainScoring(rated_paths, mode=args.mode)

    # delta
    explanation = Explanation()

    print("...did everything.")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",
                        choices=[BASELINE_MODE, RESEARCH_MODE],
                        help="Run the baseline or our research prototype")

    return parser.parse_args()
