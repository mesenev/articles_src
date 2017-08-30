from subprocess import call

import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

if sys.argv:
    folder = "calculations"
else:
    folder = sys.argv[1]

with open(folder + "/output_cost.txt") as f:
    y = f.readlines()[4:-1]
y = [float(i.replace(" ", "").replace("!", "")) for i in y]
x = [i for i in range(0, y.__len__())]
yscale = (max(y) - min(y)) / 8
plt.figure(0)
plt.plot(x, y)
plt.axis([-0.01, x.__len__(), min(y) - yscale, max(y) + yscale])
plt.xlabel('Iterations')
plt.ylabel('Cost function value')

blue_line = mlines.Line2D([], [], color='blue',
                          label="Cost function, initial: {}, final: {}".format(y[0], y[-1]))
extra = mpatches.Patch(color='none', label=r'Cost function delta: {}'.format((max(y) - min(y))))
plt.legend([blue_line, extra], [blue_line.get_label(), extra.get_label()], prop={'size': 10})
# plt.text((x.__len__()+10)*1/3, (max(y) + yscale)*4/5, , fontsize=14)
plt.grid(True)
plt.savefig(folder + "/cost_functional_dynamics.png")
# plt.show()
plt.close()

if sys.argv:
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
plt.plot(x, y1, "g-", x, y2, 'b-', x, y3, 'r-')
plt.axis([0, 1.0, -0.02, 0.52])
plt.xlabel('x-coord')
plt.ylabel('Value')

green_legend = mlines.Line2D([], [], color='green', label="Optimal control function $u$", )
blue_legend = mlines.Line2D([], [], color='blue', label="Initial control function $u$")
red_legend = mlines.Line2D([], [], color='red', label="Final control function $u$")
plt.legend(handles=[green_legend, blue_legend, red_legend], prop={'size': 10})

plt.grid(True)
plt.savefig(folder + "/control_initial_optimal_final.png")
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

green_legend = mlines.Line2D([], [], color='green', label=input_data[0][0], )
blue_legend = mlines.Line2D([], [], color='blue', label=input_data[1][0])
yellow_legend = mlines.Line2D([], [], color='yellow', label=input_data[2][0])
red_legend = mlines.Line2D([], [], color='red', label=input_data[3][0])

plt.legend(handles=[green_legend, blue_legend, yellow_legend, red_legend], prop={'size': 10})

y = input_data[0][1:]
x = np.arange(0, 1.0, 1.0 / (y.__len__() - 1)).tolist() + [1.0]
plt.plot(x, y, "g-", x, input_data[1][1:], 'b-')
plt.plot(x, input_data[2][1:], 'y--')
plt.plot(x, input_data[3][1:], 'r-')
plt.xlabel('x-coord')
plt.ylabel('value')
plt.savefig(folder + "/theta_funcs.png")
plt.close()
