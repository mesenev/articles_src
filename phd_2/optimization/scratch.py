import codecs
import json
from dolfin import *
from solver2d import SolveOptimization as Problem
from utilities import print_2d_isolines

answer = Function(Problem.state_space, 'exp4/state.xml')
print_2d_isolines(answer.split()[0])
