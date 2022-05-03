CUBE_CIRCLE = 'meshes/cube-circle-12.xml'


def cube_circle_gen():
    filename = CUBE_CIRCLE.split('/')[1]
    from mshr import Sphere, Box, generate_mesh
    from dolfin import Point, File, Mesh
    domain = Box(Point(0, 0, 0), Point(1, 1, 1)) - Sphere(Point(0.5, 0.5, 0.5), .25)
    File(filename) << generate_mesh(domain, 36)
    omega = Mesh(filename)
    return


if __name__ == '__main__':
    cube_circle_gen()
