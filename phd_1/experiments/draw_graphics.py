import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

with open('scratch/theta_ans_gamma_dynamic.txt') as file:
    data = file.read().split()

x_list = list()
y_list = list()
val_list = list()

for i in range(len(data) // 3):
    x, y, val = data[i * 3: 3 * i + 3]
    x = x[1:-1]
    y = y[:-1]
    val = val[:-1]
    x_list.append(float(x))
    y_list.append(float(y))
    val_list.append(float(val))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = np.array(x_list)
y = np.array(y_list)
z = np.array(val_list)

X, Y = np.meshgrid(np.unique(x), np.unique(y))
Z = np.zeros_like(X)

for i in range(len(x)):
    xi = np.where(np.unique(x) == x[i])[0][0]
    yi = np.where(np.unique(y) == y[i])[0][0]
    Z[yi, xi] = z[i]

# SC = ax.pcolor(Z, cmap='viridis', vmin=Z.min(), vmax=Z.max())
#
# the_divider = make_axes_locatable(ax)
# color_axis = the_divider.append_axes("right", size="5%", pad=0.1)

# plt.colorbar(SC)
ax.set_xlabel(r"$ \gamma $")
ax.set_ylabel(r"$ \beta $")
ax.set_zlabel(r"$ \theta $")
p = ax.plot_surface(X, Y, Z, cmap='viridis')
# fig.colorbar(p)
plt.show()
