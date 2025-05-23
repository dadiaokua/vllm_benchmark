from experiment.base_experiment import BaseExperiment


class LFSExperiment(BaseExperiment):
    def __init__(self, client):
        super().__init__(client)
        self.exp_type = 'LFS'
