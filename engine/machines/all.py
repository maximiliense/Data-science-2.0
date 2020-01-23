import socket

from engine.machines.special_clusters import clusters
from engine.parameters import special_parameters
from engine.logging import print_info


def detect_machine():
    hostname = socket.gethostname()
    found_machine = False
    for k, v in clusters.items():
        for h in v:
            if hostname.startswith(h):
                special_parameters.machine = k
                found_machine = True
                break
            if found_machine:
                break
    if not found_machine:
        special_parameters.machine = hostname
    print_info('The machine was identified as ' + special_parameters.machine)
