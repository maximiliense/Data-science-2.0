import random
from torch.utils.data import Dataset


class ReplayMemoryBalanced(Dataset):
    def __init__(self, replay_memory_size):
        self.replay_memory_size = replay_memory_size
        self.memory = {
            'finished': [],
            'notfinished': []
        }

        self.proba_finished = 0.1

    def add(self, other):

        if other[4]:
            self.memory['finished'].append(other)
            if len(self.memory['finished']) > self.replay_memory_size:
                self.memory['finished'].pop(0)
        else:
            self.memory['notfinished'].append(other)
            if len(self.memory['notfinished']) > self.replay_memory_size:
                self.memory['notfinished'].pop(0)

    def sample(self, batch_size=1):
        if random.random > self.proba_finished and len(self.memory['finished']) > batch_size:
            return random.sample(self.memory['finished'], min(len(self.memory['finished']), batch_size))
        else:
            return random.sample(self.memory['notfinished'], min(len(self.memory['notfinished']), batch_size))

    def save(self, path):
        raise NotImplemented()

    def load(self, path):
        raise NotImplemented()

    def current_memory_size(self):
        return len(self.memory)

    def __len__(self):
        return self.replay_memory_size

    def __getitem__(self, idx):
        if random.random() > self.proba_finished and len(self.memory['finished']) > 0:
            return self.memory['finished'][idx % len(self.memory['finished'])]
        else:
            return self.memory['notfinished'][idx % len(self.memory['notfinished'])]


class ReplayMemory(Dataset):
    def __init__(self, replay_memory_size):
        self.replay_memory_size = replay_memory_size
        self.memory = []

    def add(self, other):
        self.memory.append(other)
        if len(self.memory) > self.replay_memory_size:
            self.memory.pop(0)

    def sample(self, batch_size=1):
        return random.sample(self.memory, min(len(self.memory), batch_size))

    def save(self, path):
        raise NotImplemented()

    def load(self, path):
        raise NotImplemented()

    def current_memory_size(self):
        return len(self.memory)

    def __len__(self):
        return self.replay_memory_size

    def __getitem__(self, idx):
        return self.memory[idx % len(self.memory)]
