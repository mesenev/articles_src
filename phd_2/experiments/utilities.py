from os import mkdir

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib import pyplot as plt, cm, gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import array, linspace, meshgrid, random
from dolfin import *
import numpy as np

font = {'family': 'sans-serif',
    'name': 'Sans',
    'color': 'black',
    'weight': 'ultralight',
    'size': 10,
}


def print_all_variations(v, name, folder):
    print_3d_boundaries_on_cube(v, name=name + '_cube', folder=folder)
    print_3d_boundaries_single(v, name=name + '_single', folder=folder)
    try:
        mkdir(folder + '/separate')
    finally:
        print_3d_boundaries_separate(v, name=name, folder=folder + '/separate')


def print_2d_boundaries(v, name, folder='results'):
    mesh = UnitSquareMesh(10, 10)
    space = FunctionSpace(mesh, 'P', 1)
    v = interpolate(v, space)
    f_s = v.ufl_function_space()
    b_function = Function(f_s)
    boundaries = [
        AutoSubDomain(lambda x, on_bnd: near(x[0], 0) and on_bnd),
        AutoSubDomain(lambda x, on_bnd: near(x[1], 1) and on_bnd),
        AutoSubDomain(lambda x, b: near(x[0], 1) and b),
        AutoSubDomain(lambda x, b: near(x[1], 0) and b),
    ]
    [DirichletBC(f_s, i + 1, _).apply(b_function.vector()) for i, _ in enumerate(boundaries)]
    print("Values on left = ", *v.vector()[b_function.vector() == 1][::-1])
    print("Values on top = ", *v.vector()[b_function.vector() == 2])
    print("Values on right = ", *v.vector()[b_function.vector() == 3])
    print("Values on bottom = ", *v.vector()[b_function.vector() == 4][::-1])
    draw_simple_graphic(list(v.vector()[b_function.vector() == 1][::-1]) +
                        list(v.vector()[b_function.vector() == 2]) +
                        list(v.vector()[b_function.vector() == 3]) +
                        list(v.vector()[b_function.vector() == 4][::-1]),
                        target_file=name, folder=folder)
    return


def print_3d_boundaries_on_cube(v, name='solution', folder='results', cmap='binary'):
    '''
    https://stackoverflow.com/questions/36046338/contourf-on-the-faces-of-a-matplotlib-cube
    :param v: function to draw
    :param name: target filename
    :param folder: folder name for target
    :return: nothing. Just saves the picture
    '''
    plt.close('all')
    fig = plt.figure()
    top = lambda x, y: v(Point(x, y, 1))
    vertical_left = lambda x, y: v(Point(x, 0, y))
    vertical_right = lambda x, y: v(Point(1, x, y))
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
    # this is the example that worked for you:
    levels = linspace(0, 3, 100)
    cset[0] = ax.contourf(X, Y, top_vals, zdir='z', offset=1, levels=levels, cmap=cmap)
    # now, for the x-constant face, assign the contour to the x-plot-variable:
    cset[1] = ax.contourf(top_vals, y_m, x_m, zdir='x', offset=1, levels=levels, cmap=cmap)
    # likewise, for the y-constant face, assign the contour to the y-plot-variable:
    cset[2] = ax.contourf(x_m, top_vals, y_m, zdir='y', offset=0, levels=levels, cmap=cmap)
    # setting 3D-axis-limits:
    ax.set_xlim3d(0, 1)
    ax.set_ylim3d(0, 1)
    ax.set_zlim3d(0, 1)
    fig.subplots_adjust(right=0.8)
    color_bar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    bounds = [0, 1.5, 3]
    fig.colorbar(cset[0], cax=color_bar_ax,
                 boundaries=[-10] + bounds + [10],
                 extend='both',
                 extendfrac='auto',
                 ticks=bounds,
                 spacing='uniform',
                 orientation='vertical').ax.set_yticklabels(['< 0', '1.5', '> 3'])
    fig.colorbar(cset[1], cax=color_bar_ax,
                 boundaries=[-10] + bounds + [10],
                 extend='both',
                 extendfrac='auto',
                 ticks=bounds,
                 spacing='uniform',
                 orientation='vertical').ax.set_yticklabels(['< 0', '1.5', '> 3'])
    fig.colorbar(cset[2], cax=color_bar_ax,
                 boundaries=[-10] + bounds + [10],
                 extend='both',
                 extendfrac='auto',
                 ticks=bounds,
                 spacing='uniform',
                 orientation='vertical').ax.set_yticklabels(['< 0', '1.5', '> 3'])
    plt.savefig('{}/{}_full.eps'.format(folder, name, bbox_inches='tight'))


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
    im1 = ax1.pcolor(vertical_vals, cmap=cm.RdBu, vmin=mn, vmax=mx)
    im2 = ax2.pcolor(top_vals, cmap=cm.RdBu, vmin=mn, vmax=mx)
    im3 = ax3.pcolor(bottom_vals, cmap=cm.RdBu, vmin=mn, vmax=mx)
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

        SC = ax.pcolor(X, Y, top_vals, cmap=cm.RdBu, vmin=top_vals.min(), vmax=top_vals.max())

        the_divider = make_axes_locatable(ax)
        color_axis = the_divider.append_axes("right", size="5%", pad=0.1)

        # Colorbar
        plt.colorbar(SC, cax=color_axis)
        plt.savefig('{}/{}_{}.eps'.format(folder, name, action[1]), bbox_inches='tight')
        plt.gcf().clear()
    return


def draw_simple_graphic(data, target_file, logariphmic=False, folder='results', x_label='', y_label='', ):
    x = [i for i in range(0, data.__len__())]
    plt.figure()
    plt.semilogx(x, data) if logariphmic else plt.plot(x, data)
    # scale = (max(data) - min(data)) / 8
    # plt.semilogx([-0.01, x.__len__(), min(y) - scale, max(y) + scale])
    # plt.text((x.__len__()+10)*1/3, (max(y) + scale)*4/5, , fontsize=14)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    blue_line = mlines.Line2D([], [], color='blue', label=r"")
    extra1 = mpatches.Patch(color='none', label='Начальное значение: {}'.format(data[0]))
    extra2 = mpatches.Patch(color='none', label="Конечное значение: {}".format(data[-1]))
    plt.legend([extra1, extra2], [blue_line.get_label(),
        extra1.get_label(), extra2.get_label()], prop={'size': 10})
    plt.grid(True)
    plt.savefig("{}/{}.eps".format(folder, target_file))
    plt.close()
    return


def checkers():
    # 3d boundary drawer check
    print('3d boundary check')
    mesh = UnitCubeMesh(10, 10, 10)
    V = FunctionSpace(mesh, 'P', 1)
    u = interpolate(Expression('x[0]+x[1]+x[2]', degree=3), V)
    print_3d_boundaries_on_cube(u, 'test_3d_cube', folder='checker', cmap=cm.coolwarm)
    # simple graphic check
    print('Simple graphic check')
    draw_simple_graphic([1, 2, 3, 4, 5, 6, 7, 8, 9], 'test_simple_graphic', folder='checker')

    print('2d boundaries check')
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, 'P', 1)
    u = interpolate(Expression('x[0]+x[1]', degree=2), V)
    print_2d_boundaries(u, 'test_2d_boundary', folder='checker')

    return


def get_facet_normal(bmesh):
    # https://bitbucket.org/fenics-project/dolfin/issues/53/dirichlet-boundary-conditions-of-the-form
    '''Manually calculate FacetNormal function'''

    if not bmesh.type().dim() == 2:
        raise ValueError('Only works for 2-D mesh')

    vertices = bmesh.coordinates()
    cells = bmesh.cells()

    vec1 = vertices[cells[:, 1]] - vertices[cells[:, 0]]
    vec2 = vertices[cells[:, 2]] - vertices[cells[:, 0]]

    normals = np.cross(vec1, vec2)
    normals /= np.sqrt((normals ** 2).sum(axis=1))[:, np.newaxis]

    # Ensure outward pointing normal
    bmesh.init_cell_orientations(Expression(('x[0]', 'x[1]', 'x[2]')))
    normals[bmesh.cell_orientations() == 1] *= -1

    V = VectorFunctionSpace(bmesh, 'DG', 0)
    norm = Function(V)
    nv = norm.vector()

    for n in (0, 1, 2):
        dofmap = V.sub(n).dofmap()
        for i in xrange(dofmap.global_dimension()):
            dof_indices = dofmap.cell_dofs(i)
            assert len(dof_indices) == 1
            nv[dof_indices[0]] = normals[i, n]

    return norm
