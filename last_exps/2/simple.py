from matplotlib import pyplot

X = list()
for eps in range(9):
    X.append(0.1 * (-1 + 2 * eps / 8))


with open('result.txt') as f:
    Y = list(abs(float(xx)) for xx in f.read().split())

fig, ax = pyplot.subplots()
f = ax.plot(X, Y)
fig.savefig('deps.svg')
