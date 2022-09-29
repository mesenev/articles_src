# file should contain things for making pics from 2d functions
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from utilities import default_colormap, mkdir
from dolfin import *
from numpy import meshgrid, linspace, array, random

from utilities.defaults import font


def print_3d_boundaries_on_cube(v, name='solution', folder='results', cmap=default_colormap, colorbar_scalable=True):
    """
    https://stackoverflow.com/questions/36046338/contourf-on-the-faces-of-a-matplotlib-cube
    :param v: function to draw
    :param cmap: cmap
    :param name: target filename
    :param folder: folder name for target
    :param colorbar_scalable: should colorbar scale to actual values or equals to [0,1]
    :return: nothing. Just saves the picture
    """
    plt.close('all')
    fig = plt.figure()
    top = lambda x, y: abs(v(Point(x, y, 1)))
    vertical_left = lambda x, y: abs(v(Point(x, 0, y)))
    vertical_right = lambda x, y: abs(v(Point(1, y, x)))

    x_m, y_m = meshgrid(linspace(0, 1, num=100), linspace(0, 1, num=100))
    left_vals = array(list(map(vertical_left, x_m.reshape(100 ** 2), y_m.reshape(100 ** 2)))).reshape(100, 100)
    right_vals = array(list(map(vertical_right, x_m.reshape(100 ** 2), y_m.reshape(100 ** 2)))).reshape(100, 100)
    top_vals = array(list(map(top, x_m.reshape(100 ** 2), y_m.reshape(100 ** 2)))).reshape(100, 100)
    ax = fig.gca(projection='3d')
    X = linspace(0, 1, 100)
    Y = linspace(0, 1, 100)
    X, Y = meshgrid(X, Y)
    Z = random.rand(100, 100) * 5.0 - 10.0
    cset = [[], [], []]
    mn = min(left_vals.min(), top_vals.min(), right_vals.min())
    mx = max(left_vals.max(), top_vals.max(), right_vals.max())
    if colorbar_scalable and abs(mn - mx) > 0.01:
        levels = linspace(mn, mx, 17)
    else:
        levels = linspace(0, 1, 17)
    cset[0] = ax.contourf(X, Y, top_vals, zdir='z', offset=1, levels=levels, cmap=cmap)
    # now, for the x-constant face, assign the contour to the x-plot-variable:
    cset[1] = ax.contourf(right_vals, y_m, x_m, zdir='x', offset=1, levels=levels, cmap=cmap)
    # likewise, for the y-constant face, assign the contour to the y-plot-variable:
    cset[2] = ax.contourf(x_m, left_vals, y_m, zdir='y', offset=0, levels=levels, cmap=cmap)
    # setting 3D-axis-limits:
    ax.set_xlim3d(0, 1)
    ax.set_ylim3d(0, 1)
    ax.set_zlim3d(0, 1)
    fig.subplots_adjust(right=0.8)
    color_bar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    ticks = [mn] + [mn + (mx - mn) / 4 * i for i in range(1, 4)] + [mx]
    fig.colorbar(cset[0], cax=color_bar_ax,
                 ticks=ticks,
                 extend='both',
                 extendfrac='auto',
                 spacing='uniform',
                 orientation='vertical')
    # fig.colorbar(cset[1], cax=color_bar_ax,
    #              extend='both',
    #              extendfrac='auto',
    #              spacing='uniform',
    #              orientation='vertical')
    # fig.colorbar(cset[2], cax=color_bar_ax,
    #              extend='both',
    #              extendfrac='auto',
    #              spacing='uniform',
    #              orientation='vertical')
    plt.savefig(f'{folder}/{name}.svg', bbox_inches='tight')
    # plt.savefig(f'{folder}/{name}.png', bbox_inches='tight')


def print_all_variations(v, name, folder):
    print_3d_boundaries_on_cube(v, name=name + '_cube', folder=folder)
    print_3d_boundaries_single(v, name=name + '_single', folder=folder)
    try:
        mkdir(folder + '/separate')
    finally:
        print_3d_boundaries_separate(v, name=name, folder=folder + '/separate')


def print_3d_boundaries_single(v, name='solution', folder='results'):
    top = lambda x, y: v(Point(x, y, 1))
    bottom = lambda x, y: v(Point(x, y, 0))

    def vertical(x, y):
        if x <= 1:
            return v(Point(0, x, y))
        if x <= 2:
            return v(Point(x - 1, 1, y))
        if x <= 3:
            return v(Point(1, 3 - x, y))
        if x <= 4:
            return v(Point(4 - x, 0, y))

    fig = plt.figure()
    gs = gridspec.GridSpec(2, 2)
    ax1 = fig.add_subplot(gs[0, :], aspect='auto')
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    x_v, y_v = meshgrid(linspace(0, 1, num=100) * 4, linspace(0, 1, num=100))
    x_m, y_m = meshgrid(linspace(0, 1, num=100), linspace(0, 1, num=100))
    vertical_vals = array(list(map(vertical, x_v.reshape(100 ** 2), y_v.reshape(100 ** 2)))).reshape(100, 100)
    top_vals = array(list(map(top, x_m.reshape(100 ** 2), y_m.reshape(100 ** 2)))).reshape(100, 100)
    bottom_vals = array(list(map(bottom, x_m.reshape(100 ** 2), y_m.reshape(100 ** 2)))).reshape(100, 100)
    mn = min(vertical_vals.min(), top_vals.min(), bottom_vals.min())
    mx = max(vertical_vals.max(), top_vals.max(), bottom_vals.max())
    im1 = ax1.pcolor(vertical_vals, cmap=default_colormap, vmin=mn, vmax=mx)
    im2 = ax2.pcolor(top_vals, cmap=default_colormap, vmin=mn, vmax=mx)
    im3 = ax3.pcolor(bottom_vals, cmap=default_colormap, vmin=mn, vmax=mx)
    ax1.axis('off')
    ax1.set_title('Вертикальные грани', font)
    ax2.axis('off')
    ax2.set_title('Верхняя грань', font)
    ax3.axis('off')
    ax3.set_title('Нижняя грань', font)
    fig.subplots_adjust(right=0.8)
    color_bar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im1, cax=color_bar_ax)
    fig.colorbar(im2, cax=color_bar_ax)
    fig.colorbar(im3, cax=color_bar_ax)
    plt.savefig('{}/{}_full.eps'.format(folder, name, bbox_inches='tight'))
    plt.gcf().clear()


def print_3d_boundaries_separate(v, name='solution', folder='results'):
    top = lambda x, y: v(Point(x, y, 1))
    bottom = lambda x, y: v(Point(x, y, 0))

    def vertical(x, y):
        if x <= 1:
            return v(Point(0, x, y))
        if x <= 2:
            return v(Point(x - 1, 1, y))
        if x <= 3:
            return v(Point(1, 3 - x, y))
        if x <= 4:
            return v(Point(4 - x, 0, y))

    for action in [(top, 'top'), (bottom, 'bottom'), (vertical, 'belt')]:
        X, Y = meshgrid(linspace(0, 1, num=100) * (4 if action[1] == 'belt' else 1),
                        linspace(0, 1, num=100))
        top_vals = array(list(map(action[0],
                                  X.reshape(100 ** 2), Y.reshape(100 ** 2)))).reshape(100, 100)
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')

        SC = ax.pcolor(X, Y, top_vals, cmap=default_colormap, vmin=top_vals.min(), vmax=top_vals.max())

        the_divider = make_axes_locatable(ax)
        color_axis = the_divider.append_axes("right", size="5%", pad=0.1)

        # Colorbar
        plt.colorbar(SC, cax=color_axis)
        plt.savefig('{}/{}_{}.eps'.format(folder, name, action[1]), bbox_inches='tight')
        plt.savefig('{}/{}_{}.svg'.format(folder, name, action[1]), bbox_inches='tight')
        plt.gcf().clear()
    return
