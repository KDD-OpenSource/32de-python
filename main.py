import active_learning
import domain_scoring
import explanation
import argparse

RESEARCH_MODE = "research"
BASELINE_MODE = "baseline"

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",
                        choices=[BASELINE_MODE, RESEARCH_MODE],
                        help="Run the baseline or our research prototype")

    return parser.parse_args()


if __name__ == '__main__':
    print("Starting system...")

    args = parse_arguments()

    # beta: active learning of relevant meta-paths
    active_learner = active_learning.ActiveLearner()
    rated_paths = active_learner.retrieve_user_rating()

    # gamma: score the learned paths
    domain_score = domain_scoring.DomainScoring(rated_paths, mode=args.mode)

    # delta
    explanation = explanation.Explanation()

    print("...did everything.")