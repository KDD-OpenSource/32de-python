from domain_scoring.domain_scoring import DomainScoring


class ActiveLearner():
    def __init__(self):
        raise NotImplementedError()

    def gain_information(self):
        scoring = DomainScoring()
