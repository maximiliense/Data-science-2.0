from datascience.data.rasters.environmental_raster_glc import PatchExtractor
from datascience.data.util.source_management import check_source
from engine.core import module
from engine.logging import print_statistics

from engine.parameters import special_parameters
from datascience.visu.util import plt, get_figure


@module
def pplot(latitude, longitude, source, resolution=1., style=special_parameters.plt_style, nb_cols=5, alpha=1.):
    """
    patch plot
    :param style:
    :param latitude:
    :param longitude:
    :param source:
    :param resolution:
    :return:
    """
    r = check_source(source)
    rasters = r['rasters']
    extractor = PatchExtractor(rasters, resolution=resolution)
    extractor.add_all()
    extractor.plot(item=(latitude, longitude), return_fig=True, style=style, nb_cols=nb_cols, alpha=alpha)


@module
def raster_characteristics(source):
    """
    print infos about the rasters
    :param source:
    :return:
    """
    r = check_source(source)
    rasters = r['rasters']
    extractor = PatchExtractor(rasters)
    extractor.add_all()

    print_statistics(str(extractor))


@module
def pplot_patch(patch, resolution=1., return_fig=True, header=None):
    """
    plot a patch that has already been extracted
    :param header:
    :param patch:
    :param resolution:
    :param return_fig:
    :return:
    """
    if header is None:
        header = ['' for _ in range(len(patch))]

    # computing number of rows and columns...
    nb_rows = (len(patch) + 4) // 5
    nb_cols = 5

    plt('patch_ext', figsize=(nb_cols * 6.4 * resolution, nb_rows * 4.8 * resolution))
    fig = get_figure('patch_ext')
    for i, (t, p) in enumerate(zip(header, patch)):
        plt('patch_ext').subplot(nb_rows, nb_cols, i + 1)
        plt('patch_ext').title(t, fontsize=20)
        plt('patch_ext').imshow(p, aspect='auto')
        if len(p.shape) < 3:
            plt('patch_ext').colorbar()
    fig.tight_layout()
    if return_fig:
        return fig
    else:
        fig.show()
        plt('patch_ext').close(fig)
