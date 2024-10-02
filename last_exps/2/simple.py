from matplotlib import pyplot

X = list()
for eps in range(9):
    X.append(0.1 * (-1 + 2 * eps / 8))


with open('result.txt') as f:
    Y = list(abs(float(xx)) for xx in f.read().split())

pyplot.plot(X, Y)
pyplot.xlabel(r"$\varepsilon$")
pyplot.ylabel(r"$||\theta - \theta^\varepsilon||_{L^2(\Omega)}$")
pyplot.subplots_adjust(left=0.15)
pyplot.savefig('deps.eps', )
pyplot.savefig('deps.png', )
