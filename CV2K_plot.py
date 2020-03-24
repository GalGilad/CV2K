from CV2K_cv import *
import os
import matplotlib.pyplot as plt

# plot nmf results
# results_dir is the nmf's results directory
# args is a dictionary of parameters
def plot(results_dir, V, ranks, args={}):
    n, m = V.shape
    # create directory where we save the plots
    save_dir = results_dir + "/plot"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    ranks = np.array(ranks)
    unique_ranks = np.unique(ranks)
    
    print(unique_ranks, flush=True)

    val_err = np.load(results_dir + "/val_err.npy")

    best_mean = unique_ranks[np.argmin(np.nanmean(val_err, axis=1).tolist())]
    best_median = unique_ranks[np.argmin(np.nanmedian(val_err, axis=1).tolist())]
    print(best_median, flush=True)

    title_str = args['data'] + ", %dx%d,\nrepeatsPerRank: %d, obj: %s, maxIter: %d" % (
            n, m, args['reps'], args['obj'], args['maxiter'])

    means = np.nanmean(val_err, axis=1).tolist()
    medians = np.nanmedian(val_err, axis=1).tolist()
    print(np.log(means), flush=True)
    print(np.log(medians), flush=True)

    plt.title(title_str)
    plt.xlabel('Rank')
    plt.ylabel('Imputation Error')
    plt.plot(ranks + randn(ranks.size) * .1, val_err.flatten(), '.k', markersize=1)  # scatter
    # curves
    plt.plot(unique_ranks, means, 'r-', zorder=-1, label='Validation mean (best: %d)' % best_mean)
    plt.plot(unique_ranks, medians, 'b--', zorder=-1, label='Validation median (best: %d)' % best_median)
    plt.legend(loc='best')
    plt.xticks(unique_ranks.astype(int))
    plt.savefig(save_dir + '/scatterNcurve.pdf')
    plt.clf()

    plt.title(title_str)
    plt.xlabel('Rank')
    plt.ylabel('log(Imputation Error)')
    plt.plot(ranks + randn(ranks.size) * .1, np.log(val_err.flatten()), '.k', markersize=1)  # scatter
    plt.plot(unique_ranks, np.log(means), 'r-', zorder=-1, label='Validation mean (best: %d)' % best_mean)
    plt.plot(unique_ranks, np.log(medians), 'b--', zorder=-1, label='Validation median (best: %d)' % best_median)
    plt.legend(loc='best')
    plt.xticks(unique_ranks.astype(int))
    plt.savefig(save_dir + '/scatterNcurve_log.pdf')
    plt.clf()