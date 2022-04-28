from os import listdir
from os.path import isfile, join
from mshr import *
from dolfin import *
import matplotlib.pyplot as plt

parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True

folder = 'exp2'
set_log_active(False)
domain = Rectangle(dolfin.Point(0., 0.), dolfin.Point(1., 1.)) - \
         Circle(dolfin.Point(0.5, 0.5), .2)
omega2d = generate_mesh(domain, 100)
finite_element = FiniteElement("CG", omega2d.ufl_cell(), 1)
square = FunctionSpace(omega2d, finite_element)
xml_files = [f for f in listdir(folder) if isfile(join(folder, f)) and f.split('.')[1] == 'xml']
for i in xml_files:
    print(i)
    target = i.split('.')[0]
    theta = Function(square, f'{folder}/{i}')
    c = plot(theta)
    plt.colorbar(c)
    plt.savefig(f'{folder}/{i.split(".")[0]}.eps')
    plt.savefig(f'{folder}/{i.split(".")[0]}.png')
    plt.close()

# with open('exp2/quality.txt', 'r') as f:
#     data = list(map(float, f.read().split()))
# draw_simple_graphic(data, name='quality', folder='exp2')
