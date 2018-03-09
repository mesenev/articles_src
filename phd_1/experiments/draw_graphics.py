from subprocess import call

import numpy as np
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches


print('Launched...')
if len(sys.argv) == 1:
    folder = "calculations"
else:
    folder = sys.argv[1]

with open(folder + "/output_cost.txt") as f:
    y = f.readlines()[4:-1]
y = [float(i.replace(" ", "").replace("!", "")) for i in y]
x = [i for i in range(0, y.__len__())]
yscale = (max(y) - min(y)) / 8
plt.figure(0)
plt.semilogx(x, y)
# plt.semilogx([-0.01, x.__len__(), min(y) - yscale, max(y) + yscale])
plt.xlabel('Итерации')
plt.ylabel('Значение функционала качества')
blue_line = mlines.Line2D([], [], color='blue',
                          label=r"Функционал качества $J$")
extra1 = mpatches.Patch(color='none', label='Начальное значение: {}'.format(y[0]))
extra2 = mpatches.Patch(color='none', label="Конечное значение: {}".format(y[-1]))
plt.legend([blue_line, extra1, extra2], [blue_line.get_label(),
                                         extra1.get_label(), extra2.get_label()], prop={'size': 10})
# plt.text((x.__len__()+10)*1/3, (max(y) + yscale)*4/5, , fontsize=14)
plt.grid(True)
plt.savefig(folder + "/cost_functional_dynamics.eps")
# plt.show()
plt.close()

if len(sys.argv) ==  1:
    folder = "calculations"
else:
    folder = sys.argv[1]

with open(folder + "/control.txt") as f:
    input_data = f.readlines()
input_data = [i.replace("\t", " ").strip() for i in input_data]
y1 = input_data[input_data.index("Optimal control:") + 2: input_data.index("Initial control:") - 1]
y2 = input_data[input_data.index("Initial control:") + 2: input_data.index("Final control:") - 1]
y3 = input_data[input_data.index("Final control:") + 2:-1]
y1 = " ".join(y1).split(" ")
y1 = [i for i in map(lambda elem: float(elem), y1)]
y2 = [i for i in map(lambda elem: float(elem), " ".join(y2).split(" "))]
y3 = [i for i in map(lambda elem: float(elem), " ".join(y3).split(" "))]
x = np.arange(0, 1.0, 1.0 / (y1.__len__() - 1)).tolist() + [1.0]


plt.figure(1)
plt.plot(x, y1, "g-.", x, y2, 'b--', x, y3, 'r-')
plt.axis([0, 1.0, -0.02, 0.52])
plt.xlabel('$x$ координата')
plt.ylabel('Значение')

green_legend = mlines.Line2D([], [], color='green', linestyle='-.', label="Тестовое значение $u$", )
blue_legend = mlines.Line2D([], [], color='blue', linestyle='--', label="Начальное значение $u_0$")
red_legend = mlines.Line2D([], [], color='red', label="Найденное значение $u_{end}$")
plt.legend(handles=[green_legend, blue_legend, red_legend], prop={'size': 10})

plt.grid(True)
plt.savefig(folder + "/control_initial_optimal_final.eps")
plt.close()

plt.figure(2)
plt.grid(True)
with open(folder + "/theta.txt") as f:
    data_new = f.readlines()

input_data = []
for data in data_new:
    data = data.strip().split(" ")
    data = [data[0]] + [float(i) for i in data[1:]]
    input_data += [data]

allvals = input_data[0][1:] + input_data[1][1:] + input_data[2][1:]# + input_data[3][1:]
yscale = (max(allvals) - min(allvals)) / 8
plt.axis([0, 1.0, min(allvals) - yscale, max(allvals) + yscale])

green_legend = mlines.Line2D([], [], color='green', linestyle='-.', label='$\Theta_0$', )
blue_legend = mlines.Line2D([], [], color='blue', linestyle='--', label=r"$\Theta$ начальное значение")
yellow_legend = mlines.Line2D([], [], color='red', linestyle=':', label=r'$\Theta$ на сотой итерации')
red_legend = mlines.Line2D([], [], color='yellow', label=r'$\Theta$ на тысячной итерации')

plt.legend(handles=[green_legend, blue_legend, yellow_legend, red_legend])

y = input_data[0][1:]
x = np.arange(0, 1.0, 1.0 / (y.__len__() - 1)).tolist() + [1.0]
plt.plot(x, y, "g-.", x, input_data[1][1:], 'b--')
plt.plot(x, input_data[2][1:], 'r:')
plt.plot(x, input_data[3][1:], 'y-')
plt.xlabel('$x$ координата')
plt.ylabel('Значение')
plt.savefig(folder + "/theta_funcs.eps")
plt.close()
