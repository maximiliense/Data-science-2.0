from datascience.ml.neural.checkpoints.checkpoints import load_checkpoint
from engine.logging import print_notification, print_debug
from engine.parameters.special_parameters import restart_experiment

import pickle


class Statistics(object):
    def __init__(self, last_model, statistics_path=None, min_epochs=0):
        self._last_model = last_model
        self._final_statistics = None
        self._best_statistics = None
        self._best_model_id = None

        self._min_epochs = min_epochs

        self._cv_final_metric = None
        self._cv_metric = None
        self._statistics_path = statistics_path
        if restart_experiment:
            self.load()

    def update_statistics(self, metrics, validation_id):
        if len(metrics) > 0:
            if self._cv_metric is None:
                for m in metrics:
                    if m.cv_metric:
                        self._cv_metric = m
                        break
                better_validation = True
            else:
                better_validation = self._cv_metric.is_better(self._best_statistics[self._cv_metric.__class__.__name__][0])

            if better_validation or int(validation_id) < self._min_epochs:
                best_statistics = {}
                for m in metrics:
                    best_statistics[m.__class__.__name__] = (m.metric_score(), str(m))

                self._best_statistics = best_statistics
                self._best_model_id = validation_id

    def set_final_statistics(self, metrics):
        if len(metrics) > 0:
            for m in metrics:
                if m.cv_metric:
                    self._cv_final_metric = m
                    break

            self._final_statistics = {}
            for m in metrics:
                self._final_statistics[m.__class__.__name__] = (m.metric_score(), str(m))

    def best_metric(self):
        return self._cv_metric

    def best_metrics(self):
        return self._best_statistics

    def final_metrics(self):
        return self._final_statistics

    def final_metric(self):
        return self._cv_final_metric

    def switch_to_best_model(self):
        if self._best_model_id is not None:
            load_checkpoint(self._last_model, validation_id=self._best_model_id)
            if self._cv_metric is not None:
                print_notification(
                    '*Best validation: \'' + str(self._best_model_id) + '\', with a score of ' +
                    str(self._best_statistics[self._cv_metric.__class__.__name__][0]) + '*'
                )
            else:
                print_notification(
                    '*Best validation: \'' + str(self._best_model_id) + '\'*'
                )

    def save(self):
        statistic = {
            'best_statistics': self._best_statistics,
            'best_model_id': self._best_model_id,
            'final_statistics': self._final_statistics
        }
        print_debug('Saving statistics at ' + self._statistics_path)
        with open(self._statistics_path, 'wb') as f:
            pickle.dump(statistic, f)

    def load(self):
        print_debug('Loading statistics from ' + self._statistics_path)
        with open(self._statistics_path, 'rb') as f:
            statistics = pickle.load(f)
        self._best_statistics = statistics['best_statistics']
        self._final_statistics = statistics['final_statistics']
        self._best_model_id = statistics['best_model_id']

    def __repr__(self):
        return str(self)

    def __str__(self):
        r = ''
        if self._best_statistics is not None:
            for k in self._best_statistics.keys():
                r += str(self._best_statistics[k][1]) + '\n'
        return r
