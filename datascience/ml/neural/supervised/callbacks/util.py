from abc import ABC, abstractmethod

from engine.logging import print_logs


def init_callbacks(cbs, val_modulo, max_cb, dataset, model_z):
    for cb in cbs:
        cb.initial_call(val_modulo, max_cb, dataset, model_z)


def run_callbacks(cbs, cb_id):
    for cb in cbs:
        cb(cb_id)


def finish_callbacks(cbs):
    for cb in cbs:
        cb.last_call()


class Callback(ABC):
    def __init__(self):
        self.modulo = None
        self.nb_calls = None
        self.dataset = None
        self.model = None

    def initial_call(self, modulo, nb_calls, dataset, model):
        print_logs('Init Callback: ' + str(self))
        self.modulo = modulo
        self.nb_calls = nb_calls
        self.dataset = dataset
        self.model = model

    @abstractmethod
    def __call__(self, validation_id):
        pass

    @abstractmethod
    def last_call(self):
        pass

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return self.__class__.__name__


