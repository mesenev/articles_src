from dolfin import *

mesh = BoxMesh(Point(0, 0, 0), Point(1, 1, 1), 4, 4, 4)
bmesh = BoundaryMesh(mesh, 'exterior')
# Init w.r.t outward normal
bmesh.init_cell_orientations(Expression(('x[0]-0.5', 'x[1]-0.5', 'x[2]-0.5'), degree=3))

cell_orientations = bmesh.cell_orientations()
for o, cell in zip(cell_orientations, cells(bmesh)):
    x, y, z = cell.midpoint()

    if near(abs(x*(1-x)), 0):
        want = Point(-1 if near(x, 0) else 1, 0, 0)
    elif near(abs(y*(1-y)), 0):
        want = Point(0, -1 if near(y, 0) else 1, 0)
    else:
        want = Point(0, 0, -1 if near(z, 0) else 1)

    cn = cell.cell_normal()
    # Reorient
    if o: cn *= -1

    assert (cn - want).norm() < 1E-14, ((cn - want).norm(), (x, y, z), cn.array(), want.array())
