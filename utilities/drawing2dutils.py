# file should contain things for making pics from 2d functions
import codecs
import json

from dolfin import *

from utilities.utils import get_trace, Point, UnitSquareMesh, vectorize
from utilities.defaults import default_colormap
from utilities.drawingsimple import print_simple_graphic

from matplotlib import patches, pyplot as plt


points_2d = [Point(0, 0.5), Point(0.5, 1), Point(1, 0.5), Point(0.5, 0)]


def print_2d_boundaries(v, name=None, folder='results', steps=10, terminal_only=True):
    data = get_trace(v, steps)
    x = list(range(1, len(data) + 1))
    print_simple_graphic(data, name=name)
    if not terminal_only:
        if not name:
            raise Exception
        plt.figure()
        plt.plot(x, data, linewidth=1, color='black')
        plt.xlabel('Граница области')
        plt.ylabel('Значение')
        plt.grid(True)
        plt.savefig(f'{folder}/{name}.png')
        plt.close()

        json.dump(
            dict(x=x, data=data),
            codecs.open(f"{folder}/{name}", 'w', encoding='utf-8'),
            separators=(',', ':'), indent=1
        )
    return


def print_2d(v, name='function', folder='results', colormap=default_colormap, table=False):
    plt.figure()
    if not table:
        mesh = UnitSquareMesh(100, 100)
        V = FunctionSpace(mesh, 'P', 1)
        c = plot(interpolate(v, V), title="function", mode='color')
    else:
        c = plt.imshow(v, cmap=colormap)
    try:
        mx, mn = v.max(), v.min()
        ticks = [mn] + [mn + (mx - mn) / 4 * i for i in range(1, 4)] + [mx]
        plt.colorbar(
            c, ticks=ticks, extend='both', extendfrac='auto',
            spacing='uniform', orientation='vertical'
        )
    except:
        plt.colorbar(c)
    # c.axes.set_xticks([-1, 89])
    # c.axes.set_yticks([0])
    c.axes.yaxis.set_ticklabels(['1', ])
    c.axes.xaxis.set_ticklabels(['0', '1'])
    plt.savefig(f'{folder}/{name}.png')


def print_2d_isolines(v, name='function', folder='results', precision=0.01, table=False):
    if table:
        Z = v
    else:
        import numpy as np
        x = np.arange(0, 1.0, precision)
        y = np.arange(0, 1.0, precision)
        X, Y = np.meshgrid(x, y)
        Z = vectorize(lambda _, __: abs(v(Point(_, __))))(X, Y)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # levels = [
    #     0.4,
    #     0.55,
    #     0.65,
    #     0.7,
    #     0.74,
    #     0.79,
    #     0.84,
    #     0.9,
    # ]
    # a = ax.contour(Z, levels=levels, colors='k', linewidths=0.4, extent=[0, 100, 0, 100])
    a = ax.contour(Z, colors='k', linewidths=0.4, extent=[0, 100, 0, 100])
    fmt = {}
    # for l in levels:
    #     fmt[l] = str(l)[:4]

    # ax.clabel(a, a.levels, fontsize=9, inline=True, fmt=fmt)
    ax.clabel(a, a.levels, fontsize=9, inline=True)
    ax.set_aspect('equal')
    # ax.axes.xaxis.set_ticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1'])
    # ax.axes.yaxis.set_ticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1'])
    extra2 = patches.Patch(color='none', label=f'Максимальное значение: {Z.max()}')
    extra1 = patches.Patch(color='none', label=f'Минимальное значение: {Z.min()}')
    plt.legend([extra1, extra2], [extra1.get_label(), extra2.get_label()], prop={'size': 10})

    fig.savefig(f'{folder}/{name}_equal.png', bbox_inches='tight')
    fig.savefig(f'{folder}/{name}_equal.eps', bbox_inches='tight')
    ax.set_aspect('auto')
    fig.savefig(f'{folder}/{name}_auto.png', bbox_inches='tight')
    fig.savefig(f'{folder}/{name}_auto.eps', bbox_inches='tight')


def print_two_with_colorbar(v1, v2, name, folder='results'):
    import numpy as np

    x = np.arange(0.0, 1.0, 0.01)
    y = np.arange(0.0, 1.0, 0.01)
    X, Y = np.meshgrid(x, y)
    A = X * Y
    B = X * Y
    i1, j1 = 0, 0
    for i in x[::-1]:
        j1 = 0
        for j in y:
            A[i1, j1] = v1(Point(j, i))
            B[i1, j1] = v2(Point(j, i))
            j1 += 1
        i1 += 1

    fig, axes = plt.subplots(nrows=1, ncols=2)
    im = None
    for ax, f in zip(axes.flat, [A, B]):
        im = ax.imshow(f)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    plt.savefig(f'{folder}/{name}.png')
