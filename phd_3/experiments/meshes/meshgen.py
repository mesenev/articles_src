CUBE_CIRCLE = 'meshes/cube-circle-12.xml'
SQUARE_CIRCLE = 'meshes/square-circle-12.xml'


def cube_circle_gen():
    filename = CUBE_CIRCLE.split('/')[1]
    from mshr import Sphere, Box, generate_mesh
    from dolfin import Point, File, Mesh
    domain = (
            Box(Point(0, 0, 0), Point(1, 1, 1))
            - Sphere(Point(0.5, 0.5, 0.5), .15)
    )
    File(filename) << generate_mesh(domain, 33)
    omega = Mesh(filename)
    return


def square_circle_gen():
    filename = SQUARE_CIRCLE.split('/')[1]
    from mshr import Rectangle, Circle, generate_mesh
    from dolfin import Point, File, Mesh
    domain = (
            Rectangle(Point(0., 0.), Point(1., 1.)) -
            Circle(Point(0.5, 0.5), .2)
    )
    File(filename) << generate_mesh(domain, 200)
    omega = Mesh(filename)
    return


if __name__ == '__main__':
    # cube_circle_gen()
    # square_circle_gen()
    exit(0)
