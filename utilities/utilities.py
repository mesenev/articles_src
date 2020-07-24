import shutil
from os import mkdir

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from dolfin import *
from matplotlib import pyplot as plt, cm, gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
# noinspection PyUnresolvedReferences
from mpl_toolkits.mplot3d import Axes3D
from numpy import array, linspace, meshgrid, random, cross, sqrt, newaxis

from utilities import asciichartpy

font = {'family': 'sans-serif',
        'name': 'Sans',
        'color': 'black',
        'weight': 'ultralight',
        'size': 10,
        }


def clear_dir(folder):
    try:
        shutil.rmtree(folder)
    except OSError:
        print("Deletion of the directory %s failed" % folder)
    finally:
        mkdir(folder)


default_colormap = cm.Greys

points_2d = [Point(0, 0.5), Point(0.5, 1), Point(1, 0.5), Point(0.5, 0)]


def print_all_variations(v, name, folder):
    print_3d_boundaries_on_cube(v, name=name + '_cube', folder=folder)
    print_3d_boundaries_single(v, name=name + '_single', folder=folder)
    try:
        mkdir(folder + '/separate')
    finally:
        print_3d_boundaries_separate(v, name=name, folder=folder + '/separate')


def print_2d_boundaries(v, name=None, folder='results', steps=10, terminal_only=True):
    left = list(map(lambda x: v(Point(0, x)), (1 / (steps - 1) * _ for _ in range(0, steps))))
    top = list(map(lambda x: v(Point(x, 1)), (1 / (steps - 1) * _ for _ in range(0, steps))))
    right = list(map(lambda x: v(Point(1, x)), (1 - 1 / (steps - 1) * _ for _ in range(0, steps))))
    bottom = list(map(lambda x: v(Point(x, 0)), (1 - 1 / (steps - 1) * _ for _ in range(0, steps))))

    # print("Values on left = ", *left)
    # print("Values on top = ", *top)
    # print("Values on right = ", *right)
    # print("Values on bottom = ", *bottom)
    print_simple_graphic(left[:-1] + top[:-1] + right[:-1] + bottom, name=name)
    if not terminal_only:
        if not name:
            raise Exception
        draw_simple_graphic(left[:-1] + top[:-1] + right[:-1] + bottom, target_file=name, folder=folder)
    return


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
    top = lambda x, y: v(Point(x, y, 1))
    vertical_left = lambda x, y: v(Point(x, 0, y))
    vertical_right = lambda x, y: v(Point(1, y, x))

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
    if colorbar_scalable and mn != mx:
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
    plt.savefig(f'{folder}/{name}.eps', bbox_inches='tight')


def print_2d(v, name='function', folder='results'):
    mesh = UnitSquareMesh(100, 100)
    V = FunctionSpace(mesh, 'P', 1)
    plt.figure()
    c = plot(interpolate(v, V), title="function", mode='color')
    plt.colorbar(c)
    plt.savefig(f'{folder}/{name}.png')


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
    for ax, f in zip(axes.flat, [A, B]):
        im = ax.imshow(f)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    plt.savefig(f'{folder}/{name}.png')


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
        plt.gcf().clear()
    return


def draw_simple_graphic(
        data, target_file, logarithmic=False, folder='results', x_label='', y_label='',
):
    x = [i for i in range(0, data.__len__())]
    plt.figure()
    plt.semilogx(x, data, color='black') if logarithmic else plt.plot(x, data, color='black')
    # scale = (max(data) - min(data)) / 8
    # plt.semilogx([-0.01, x.__len__(), min(y) - scale, max(y) + scale])
    # plt.text((x.__len__()+10)*1/3, (max(y) + scale)*4/5, , fontsize=14)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    blue_line = mlines.Line2D([], [], color='black', label=r"")
    extra1 = mpatches.Patch(color='none', label=f'Начальное значение: {data[0]}')
    extra2 = mpatches.Patch(color='none', label=f'Конечное значение: {data[-1]}')
    plt.legend([extra1, extra2], [extra1.get_label(), extra2.get_label()], prop={'size': 10})
    plt.grid(True)
    plt.savefig(f'{folder}/{target_file}.eps')
    plt.close()
    return


def print_simple_graphic(data, name=None):
    print()
    if name:
        print(name.center(60, ' '))
    print(asciichartpy.plot(data, dict(height=10)))
    print()


def checkers():
    # 3d boundary drawer check
    print('3d boundary check')
    mesh = UnitCubeMesh(10, 10, 10)
    V = FunctionSpace(mesh, 'P', 1)
    u = interpolate(Expression('x[0]/2+x[1]/3+x[2]', degree=3), V)
    print_3d_boundaries_on_cube(u, 'test_3d_cube', folder='checker')
    # simple graphic check
    # print('Simple graphic check')
    # draw_simple_graphic([1, 2, 3, 4, 5, 6, 7, 8, 9], 'test_simple_graphic', folder='checker')

    # print('2d boundaries check')
    # mesh = UnitSquareMesh(10, 10)
    # V = FunctionSpace(mesh, 'P', 1)
    # u = interpolate(Expression('x[0]+x[1]', degree=2), V)
    # print_2d_boundaries(u, 'test_2d_boundary', folder='checker')

    return


# checkers()


class Normal(UserExpression):
    def __init__(self, mesh, **kwargs):
        self.mesh = mesh
        super().__init__(**kwargs)

    def eval_cell(self, values, x, ufc_cell):
        values[0] = 0
        values[1] = 0
        if x[0] < DOLFIN_EPS:
            values[0] = -1
        if x[0] + DOLFIN_EPS > 1:
            values[0] = 1
        if x[1] < DOLFIN_EPS:
            values[1] = -1
        if x[1] + DOLFIN_EPS > 1:
            values[1] = 1
        if abs(values[0]) == abs(values[1]) == 1:
            values[0] *= 0.65
            values[1] *= 0.65

    def value_shape(self):
        return (2,)


def get_facet_normal(bmesh):
    # https://bitbucket.org/fenics-project/dolfin/issues/53/dirichlet-boundary-conditions-of-the-form
    '''Manually calculate FacetNormal function'''

    if not bmesh.type().dim() == 2:
        raise ValueError('Only works for 2-D mesh')

    vertices = bmesh.coordinates()
    cells = bmesh.cells()

    vec1 = vertices[cells[:, 1]] - vertices[cells[:, 0]]
    vec2 = vertices[cells[:, 2]] - vertices[cells[:, 0]]

    normals = cross(vec1, vec2)
    normals /= sqrt((normals ** 2).sum(axis=1))[:, newaxis]

    # Ensure outward pointing normal
    bmesh.init_cell_orientations(Expression(('x[0]', 'x[1]', 'x[2]')))
    normals[bmesh.cell_orientations() == 1] *= -1

    V = VectorFunctionSpace(bmesh, 'DG', 0)
    norm = Function(V)
    nv = norm.vector()

    for n in (0, 1, 2):
        dofmap = V.sub(n).dofmap()
        for i in range(dofmap.global_dimension()):
            dof_indices = dofmap.cell_dofs(i)
            assert len(dof_indices) == 1
            nv[dof_indices[0]] = normals[i, n]

    return norm
