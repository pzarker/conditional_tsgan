import os
import numpy as np
import matplotlib.pyplot as plt


def save_plot_sample(samples, title, file_dir, file_name, n_samples=6, ncol=2):
    assert n_samples <= samples.shape[0]
    assert n_samples % ncol == 0
    sample_length = samples.shape[1]

    samples = samples.detach().numpy()

    col = 'grey'

    x_points = np.arange(sample_length)

    nrow = int(n_samples/ncol)
    fig, axarr = plt.subplots(nrow, ncol, sharex=True, figsize=(6, 6))
    for m in range(nrow):
        for n in range(ncol):
            # first column
            sample = samples[n*nrow + m, :, 0]
            axarr[m, n].plot(x_points, sample, color=col)
            axarr[m, n].set_ylim(-1, 1)
    for n in range(ncol):
        axarr[-1, n].xaxis.set_ticks(range(0, sample_length, int(sample_length/4)))
    fig.suptitle(title)
    fig.subplots_adjust(hspace=0.15)
    fig.savefig(os.path.join(file_dir, f"{file_name}.png"))
    plt.clf()
    plt.close()
    return
