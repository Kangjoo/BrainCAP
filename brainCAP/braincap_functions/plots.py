import matplotlib.pyplot as plt
import numpy as np

def plot_histogram(data, range, out):
    fig, axs = plt.subplots(1, 1)
    cnts, values, bars = axs.hist(data, bins=range[1]-range[0], range=range)

    #Color based on frequency
    cmap = plt.cm.viridis
    for i, (cnt, value, bar) in enumerate(zip(cnts, values, bars)):
        bar.set_facecolor(cmap(cnt/cnts.max()))

    plt.savefig(out)
    plt.close()

    return

def plot_scree(c_vars, scores, c_var, final_k, out):

    fig, axs = plt.subplots(1, 1)

    x_line = np.arange(len(c_vars))

    axs.plot(x_line,scores,'o-',markersize=4,color='grey',linewidth=2)

    axs.scatter(x_line[c_vars.index(final_k)], scores[c_vars.index(final_k)], 150, color='lightgreen', clip_on=False)

    axs.spines[['right', 'top']].set_visible(False)
    axs.spines.bottom.set(linewidth=2.5)
    axs.spines.left.set(linewidth=2.5)
    axs.set_xlabel(f"{c_var}", size=12)
    axs.set_ylabel("Silhouette score", size=12)
    axs.set(xticklabels=[])
    axs.tick_params(bottom=False)
    axs.set_xlim(left=-2)

    #Label points
    for x,y,label in zip(x_line, scores, c_vars):
        axs.text(x+0.2, y+0.2, f"{label}", fontsize=8)

    plt.savefig(out)
    plt.close()
    return