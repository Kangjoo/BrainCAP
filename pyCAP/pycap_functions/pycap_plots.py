import matplotlib.pyplot as plt

def plot_histogram(data, range, out):
    fig, axs = plt.subplots(1, 1)
    cnts, values, bars = axs.hist(data, bins=range[1]-range[0], range=range)

    #Color based on frequency
    cmap = plt.cm.viridis
    for i, (cnt, value, bar) in enumerate(zip(cnts, values, bars)):
        bar.set_facecolor(cmap(cnt/cnts.max()))

    plt.savefig(out)

    return