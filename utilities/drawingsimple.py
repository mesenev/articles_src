# file should contain things for making pics from 1d-arrays
from matplotlib import pyplot as plt, patches


def draw_simple_graphic(data, name, logarithmic=False, folder='results', x_label='iterations', y_label='', ):
    x = list(range(0, data.__len__()))
    plt.figure()
    if logarithmic:
        plt.semilogx(x, data, linewidth=1, color='black')
    else:
        plt.plot(x, data, linewidth=1, color='black')
    # scale = (max(data) - min(data)) / 8
    # plt.semilogx([-0.01, x.__len__(), min(y) - scale, max(y) + scale])
    # plt.text((x.__len__()+10)*1/3, (max(y) + scale)*4/5, , fontsize=14)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    extra1 = patches.Patch(color='none', label=f'Начальное значение: {data[0]}')
    extra2 = patches.Patch(color='none', label=f'Конечное значение: {data[-1]}')
    plt.legend([extra1, extra2], [extra1.get_label(), extra2.get_label()], prop={'size': 10})
    plt.grid(True)
    plt.savefig(f'{folder}/{name}.eps')
    plt.close()
    return


def print_simple_graphic(data, name=None):
    print()
    if name:
        print(name.center(60, ' '))
    from utilities import asciichartpy
    print(asciichartpy.plot(data, dict(height=10)))
    print()
