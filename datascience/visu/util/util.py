import matplotlib
matplotlib.use('Agg')
from engine.logging import print_logs
from engine.parameters.special_parameters import plt_style
from engine.path import output_path
import matplotlib.pyplot
from engine.core import module

figures = {}
figure_id_count = 0
matplotlib.pyplot.style.use(plt_style)


def create_figure(figure_name, *args, **kwargs):
    global figure_id_count
    figure_id = figure_id_count
    figure_id_count += 1
    figure = matplotlib.pyplot.figure(num=figure_id, *args, **kwargs)
    if 'figsize' in kwargs:
        figure.set_figheight(kwargs['figsize'][1])
        figure.set_figwidth(kwargs['figsize'][0])

    figures[figure_name] = (figure_id, figure, kwargs)


def delete_figure(figure_name):
    plt(figure_name).clf()
    figures.pop(figure_name)


def plt(figure_name=None, *args, **kwargs):
    if figure_name is not None:
        if figure_name not in figures:
            create_figure(figure_name, *args, **kwargs)

        matplotlib.pyplot.figure(figures[figure_name][0])
    return matplotlib.pyplot


def get_figure(figure_name):
    if figure_name not in figures:
        create_figure(figure_name)
    return figures[figure_name][1]


def show_fig():
    global figures
    matplotlib.pyplot.show()
    figures = {}


def _save_fig(path_name, figure):

    print_logs('Saving figure at: ' + path_name)
    figure.savefig(path_name)


def save_fig_direct_call(path=None, figure_name=None, extension='jpeg'):
    if '.' not in extension:
        extension = '.' + extension
    global figures
    if figure_name is None:
        for k in figures.keys():
            path_name = output_path(k + extension)
            figure = figures[k][1]
            _save_fig(path_name, figure)
            matplotlib.pyplot.close(figure)
        figures = {}
    else:
        path_name = output_path(figure_name + extension) if path is None else path
        figure = figures[figure_name][1]
        _save_fig(path_name, figure)
        fig = figures.pop(figure_name)
        matplotlib.pyplot.close(fig[1])


@module
def save_fig(path=None, figure_name=None, extension='jpeg'):
    save_fig_direct_call(path, figure_name, extension)
