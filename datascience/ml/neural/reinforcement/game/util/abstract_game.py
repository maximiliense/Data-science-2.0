from abc import abstractmethod, ABC


class AbstractGame(ABC):
    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def get_state(self):
        pass

    @abstractmethod
    def score(self):
        pass

    @abstractmethod
    def action(self, action):
        pass
