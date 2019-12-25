from datascience.ml.neural.checkpoints.checkpoints import load_checkpoint
from engine.logging import print_notification, print_debug
from engine.parameters.special_parameters import restart_experiment

import pickle


class Statistics(object):
    def __init__(self, last_model, statistics_path=None, min_epochs=0):
        self.last_model = last_model
        self.final_statistics = None
        self.best_statistics = None
        self.best_model_id = None

        self.min_epochs = min_epochs

        self.cv_metric = None
        self.statistics_path = statistics_path
        if restart_experiment:
            self.load()

    def update_statistics(self, metrics, validation_id):
        if len(metrics) > 0:
            if self.cv_metric is None:
                for m in metrics:
                    if m.cv_metric:
                        self.cv_metric = m
                        break
                better_validation = True
            else:
                better_validation = self.cv_metric.is_better(self.best_statistics[self.cv_metric.__class__.__name__][0])

            if better_validation or validation_id < self.min_epochs:
                best_statistics = {}
                for m in metrics:
                    best_statistics[m.__class__.__name__] = (m.metric_score(), str(m))

                self.best_statistics = best_statistics
                self.best_model_id = validation_id

    def set_final_statistics(self, metrics):
        self.best_statistics = {}
        for m in metrics:
            self.best_statistics[m.__class__.__name__] = (m.metric_score(), str(m))

    def switch_to_best_model(self):
        if self.best_model_id is not None:
            load_checkpoint(self.last_model, validation_id=self.best_model_id)
            if self.cv_metric is not None:
                print_notification(
                    '*Best validation: \''+str(self.best_model_id) + '\', with a score of ' +
                    str(self.best_statistics[self.cv_metric.__class__.__name__][0]) + '*'
                )
            else:
                print_notification(
                    '*Best validation: \'' + str(self.best_model_id) + '\'*'
                )

    def save(self):
        statistic = {
            'best_statistics': self.best_statistics,
            'best_model_id': self.best_model_id
        }
        print_debug('Saving statistics at ' + self.statistics_path)
        with open(self.statistics_path, 'wb') as f:
            pickle.dump(statistic, f)

    def load(self):
        print_debug('Loading statistics from ' + self.statistics_path)
        with open(self.statistics_path, 'rb') as f:
            statistics = pickle.load(f)
        self.best_statistics = statistics['best_statistics']
        self.best_model_id = statistics['best_model_id']

    def __repr__(self):
        return str(self)

    def __str__(self):
        r = ''
        for k in self.best_statistics.keys():
            r += str(self.best_statistics[k][1]) + '\n'
        return r
