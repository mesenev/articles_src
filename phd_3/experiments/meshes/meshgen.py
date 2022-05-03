CUBE_CIRCLE = 'meshes/cube-circle-12.xml'


def cube_circle_gen():
    filename = CUBE_CIRCLE.split('/')[1:]
    from mshr import Sphere, Box, generate_mesh
    from dolfin import Point, File, Mesh
    domain = Box(Point(0, 0, 0), Point(1, 1, 1)) - Sphere(Point(0.5, 0.5, 0.5), .25)
    File(CUBE_CIRCLE) << generate_mesh(domain, 12)
    omega = Mesh(CUBE_CIRCLE)
    return