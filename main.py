from active_learning.active_learning import ActiveLearner
from domain_scoring.domain_scoring import DomainScoring
from explanation.explanation import Explanation

if __name__ == '__main__':
    print("Starting system...")

    # beta: active learning of relevant meta-paths
    active_learner = ActiveLearner()
    rated_paths = active_learner.retrieve_user_rating()

    # gamma: score the learned paths
    domain_score = DomainScoring(rated_paths)

    # delta
    explanation = Explanation()

    print("...did everything.")
