from matplotlib import pyplot


def draw_simple(ind: int):
    with open(f'theta_avg{ind}.txt') as f:
        Y = list(abs(float(xx)) for xx in f.read().split())

    X = list(eps * 1/len(Y) for eps in range(len(Y)))
    pyplot.plot(X, Y)
    pyplot.ylim(-0, 1.1)
    pyplot.xlabel(r"$t$")
    pyplot.ylabel(r"$||\theta||_{L^2(\Omega)}$")
    pyplot.subplots_adjust(left=0.15)
    pyplot.grid()
    pyplot.savefig(f'theta_dyn_time{ind}.eps', )
    pyplot.savefig(f'theta_dyn_time{ind}.png', )

# draw_simple(1)
# draw_simple(2)
