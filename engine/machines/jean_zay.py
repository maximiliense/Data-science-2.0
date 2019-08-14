from engine.core import module

import os

from engine.logging import print_info


@module
def check_machine():
    """
    execute some commands to print specific information about the machine.
    """

    # list of commands to be executed
    commands = ('env', 'module list', 'pwd', 'hostname')

    for c in commands:
        print(('[' + c + ']' + ' ' + '*' * 80)[:80])
        print_info(os.popen(c).read())
