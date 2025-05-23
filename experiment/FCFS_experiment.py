from experiment.base_experiment import BaseExperiment


class FCFSExperiment(BaseExperiment):
    def __init__(self, client, config):
        super().__init__(client, config)
        self.exp_type = 'FCFS'
