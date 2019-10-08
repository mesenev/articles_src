from dolfin import *
import matplotlib.pyplot as plt

mesh = UnitSquareMesh(10, 10)
bmesh = BoundaryMesh(mesh, 'exterior')


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

