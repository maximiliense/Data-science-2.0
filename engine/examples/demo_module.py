from engine.core import module


@module
def say_whee():
    print("Whee!")


@module
def say_hello(name, title='Sir'):
    print('Hello {} {}!'.format(title, name))
