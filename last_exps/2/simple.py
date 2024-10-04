from matplotlib import pyplot

def draw_eps(name):
    X = list()
    for eps in range(9):
        X.append(0.1 * (-1 + 2 * eps / 8))


    with open('result.txt') as f:
        Y = list(abs(float(xx)) for xx in f.read().split())

    pyplot.plot(X, Y)
    pyplot.xlabel(r"$\varepsilon$")
    pyplot.ylabel(r"$||\theta - \theta^\varepsilon||_{L^2(\Omega)}$")
    pyplot.subplots_adjust(left=0.15)
    pyplot.savefig(f'{name}.eps', )
    pyplot.savefig(f'{name}.png', )
