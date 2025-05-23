from experiment.base_experiment import BaseExperiment


class FCFSExperiment(BaseExperiment):
    def __init__(self, client):
        super().__init__(client)
        self.exp_type = 'FCFS'
