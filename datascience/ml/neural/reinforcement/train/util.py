import random
import torch
import numpy as np

from engine.hardware import use_gpu


def construct_action(epsilon, model_z, state, output_size, xp_gpu):
    random_action = random.random() <= epsilon
    action_index = [
        torch.randint(output_size, torch.Size([]), dtype=torch.int)
        if random_action else torch.argmax(model_z(*state)[0])
    ][0]

    action = torch.zeros([output_size], dtype=torch.float32)

    if xp_gpu:
        action = action.cuda()

    action[action_index] = 1

    return action


def init_game(game, replay_memory, action_size, init_size):
    """
    initialize the game and the memory with random events
    :param game:
    :param replay_memory:
    :param action_size:
    :param init_size:
    :return: the last state
    """
    game.start()
    state = game.get_state()
    # initializing some random elements in the memory
    for _ in range(init_size):
        action_index = random.randint(0, action_size - 1)
        new_state, reward, finish = game.action(action_index)
        action = np.zeros(action_size).astype(np.float32)
        action[action_index] = 1

        replay_memory.add((process_state_back(state), action, process_state_back(new_state), reward, finish))
        state = new_state

    # print(color.RED + 'Model and memory initialized...' + color.END)
    return new_state


def process_state(state):
    if type(state) is tuple or type(state) is list:

        if use_gpu():
            return tuple(s.cuda() for s in state)
        else:
            return tuple(s for s in state)
    else:
        return (state.cuda(),) if use_gpu() else (state,)


def process_state_back(state):
    if type(state) is torch.Tensor:
        return state.squeeze().cpu()
    else:
        return tuple(s.squeeze().cpu() for s in state)


def unsqueeze(state):
    if type(state) is tuple or type(state) is list:
        return tuple(s.unsqueeze(0) for s in state)
    else:
        return state.unsqueeze(0)
