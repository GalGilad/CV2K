import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import sys
import collections
from multiprocessing import Pool
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import argparse
import scipy.stats
import scipy.optimize.nnls
from CV2K_cv import *


def rollback(errors):
    """
    Finds K after rolling backwards from the k with minimum median value, using Wilcoxon rank-sum test
    :param errors: numpy array of errors (Ks x repetitions)
    :return: index of K after rollback
    """
    best_k_arg = np.argmin(np.nanmedian(errors, axis=1))
    # rollback
    for k in range(best_k_arg):
        best_k_runs = errors[best_k_arg]
        next_k_runs = errors[k]
        u, p = scipy.stats.ranksums(next_k_runs, best_k_runs)
        if p > 0.05:
            best_k_arg = k
            break
    return best_k_arg


def NMF(V, k):
    """
    V ~ WH using CV2K_cv.py methods
    :param V: input matrix (n x m)
    :param k: rank of factorization
    :return: W (n x k) and H (k x m) matrices
    """
    W, H = init_factor_matrices(V, k, O=np.ones((V.shape), dtype=bool), eps=1e-6, obj='euc', reg=0)
    W, H, _ = optimize(V, W, H, O=np.ones((V.shape), dtype=bool), maxiter=args.maxiter, eps=1e-6, obj='euc', reg=0)

    return W, H


def nnls(V, H):
    """
    Given V and H, computes W matrix: V ~ WH using SciPy's NNLS
    :param V: input matrix (n x m)
    :param H: signature matrix (k x m)
    :return: exposure matrix (n x k)
    """
    n = V.shape[0]
    W = []
    for j in range(n):
        W.append(scipy.optimize.nnls(H.T, V[j])[0])

    return np.array(W)


def CV2K_x_compute_error(V, k, sample_fold, category_folds):
    """
    CV2K_x helper - computes overall error for a given k and sample_fold, by accumulating all category_fold errors.
    using parameters from main.
    :param V: input matrix (n x m)
    :param k: rank of factorization
    :param sample_fold: index of group of samples that correspond to the current fold
    :param category_folds: total number of category folds
    :return: float - total error for the given sample fold
    """
    np.random.seed()
    errors = 0

    # splitting samples to train and test
    train_samples_idx = np.concatenate([indices for i, indices in enumerate(fold_sample_indices) if i != sample_fold])
    test_samples_idx = fold_sample_indices[sample_fold]
    V_train = V[train_samples_idx]
    V_test = V[test_samples_idx]

    # producing H matrix from V using NMF
    _, H = NMF(V_train, k)
    # normalizing H
    H /= H.sum(1, keepdims=True)

    for category_fold in range(category_folds):
        # splitting categories to train and test
        train_categories_idx = np.concatenate([indices for i, indices in enumerate(fold_category_indices) if
                                               i != category_fold])
        test_categories_idx = fold_category_indices[category_fold]
        V_for_nnls = V_test.T[train_categories_idx].T

        # creating normalizing H for NNLS
        H_for_nnls = H.T[train_categories_idx].T.copy()
        H_for_nnls /= H_for_nnls.sum(1, keepdims=True)

        # getting final W using NNLS
        final_W = nnls(V_for_nnls, H_for_nnls)
        final_W = final_W / final_W.sum(1, keepdims=True)

        # getting final V, H
        final_H = H.T[test_categories_idx].T
        final_V_from_WH = final_W.dot(final_H)
        normalized_V_test = V_test / V_test.sum(1, keepdims=True)
        final_V = normalized_V_test.T[test_categories_idx].T

        # compute error and add to overall
        errors += np.sum(np.abs(final_V_from_WH - final_V)) / final_V_from_WH.size

    print("k = %d, fold: %d - completed." % (k, sample_fold))

    return errors


def CV2K_x():
    """
    runs CV2K_x method. using parameters from main.
    :return: numpy array - errors for each k and sample fold - error matrix (Ks x sample_folds)
    """
    # run method with pool
    with Pool(args.workers) as pool:
        res = [pool.apply_async(CV2K_x_compute_error, (V, k, sample_fold, category_folds)) for
               k in Ks for sample_fold in range(sample_folds)]
        pool.close()
        pool.join()
    # get results from pool
    res = [x.get() for x in res]
    errors = np.reshape([x for x in res], (len(Ks), sample_folds))

    return errors


def CV2K_standard():
    """
    runs CV2K standard method. using parameters from main.
    :return: numpy array - errors for each k and repetition - error matrix (Ks x repetitions)
    """
    rank_cycle = np.array([[i] * args.reps for i in Ks]).flatten()
    # create pool and run nmf
    with Pool(args.workers) as pool:
        res = [pool.apply_async(crossval_nmf, (V, rank, args.maxiter, eps, args.obj, args.reg, args.fraction))
               for rank in rank_cycle]
        pool.close()
        pool.join()
    # get results from pool
    res = [x.get() for x in res]
    Ws, Hs = [x[0] for x in res], [x[1] for x in res]  # ignoring
    errors = np.reshape(np.array([x[2] for x in res]), (len(np.unique(rank_cycle)), args.reps))

    return errors


def CV2K_auto_binary_search(variant='standard'):
    Ks_and_errors = {}
    global Ks
    Ks = [5, 10, 20]  # predetermined starting Ks
    low_bound, up_bound = 1, np.min([80, n, m])
    stop = False
    flag = 0

    while not stop:
        if flag == 1: stop = True
        if variant == 'standard':
            errors = CV2K_standard()
        else:  # variant == 'x'
            errors = CV2K_x()
        # store errors in dictionary and sort it by key
        for i in range(errors.shape[0]):
            Ks_and_errors[Ks[i]] = errors[i]
        Ks_and_errors = collections.OrderedDict(sorted(Ks_and_errors.items()))
        all_errors = np.vstack(list(Ks_and_errors.values()))
        # load next Ks
        sorted_visited_Ks = np.array(list(Ks_and_errors.keys()))
        best_k_idx = rollback(all_errors)
        best_k = sorted_visited_Ks[best_k_idx]
        best_k_global_arg = np.nonzero(sorted_visited_Ks == best_k)[0][0]
        if best_k_global_arg == 0: new_Ks = [max(low_bound, best_k//2), best_k + (sorted_visited_Ks[1]-best_k)//2]
        elif best_k_global_arg == len(sorted_visited_Ks)-1: new_Ks = [best_k - (best_k-sorted_visited_Ks[-2])//2,
                                                                      min(best_k*2, up_bound)]
        else: new_Ks = [best_k - (best_k-sorted_visited_Ks[best_k_global_arg-1])//2,
                    best_k + (sorted_visited_Ks[best_k_global_arg+1]-best_k)//2]
        Ks = np.setdiff1d(new_Ks, sorted_visited_Ks)
        if len(Ks) == 0:
            flag = 1  # indicates final loop -- 7 size window [k-3, k+3]
            Ks = np.setdiff1d(np.arange(max(low_bound, best_k-3), min(best_k+4, up_bound+1)), sorted_visited_Ks)
            if len(Ks) == 0: stop = True

    # update global Ks and final errors
    Ks = np.array(list(Ks_and_errors.keys()))
    final_errors = np.vstack(list(Ks_and_errors.values()))
    return final_errors


def produce_figure(rollback=True):
    """
    produces a figure that shows the results
    :param rollback: int - chosen k, after rolling backwards from the k with minimum median error
    """
    print("Tested Ks: ", Ks, flush=True)
    medians = np.nanmedian(errors, axis=1)
    print("Median values: ", np.log(medians), flush=True)
    print("Best K by median value: ", best_k, flush=True)

    title_str = args.data + "\n%dx%d, reps/folds: %d" % (n, m, args.reps)
    plt.title(title_str)
    plt.xlabel('Rank')
    plt.ylabel('log(Imputation Error)')
    rank_cycle = np.array([[i] * args.reps for i in Ks]).flatten()
    plt.plot(rank_cycle + randn(rank_cycle.size) * .1, np.log(errors.flatten()), '.k', markersize=1)  # scatter
    plt.plot(Ks, np.log(medians), 'b--', zorder=-1, label='Validation median (best: %d)' % best_k)
    plt.legend(loc='best')
    plt.xticks(Ks.astype(int))
    if rollback:
        plt.axvspan(best_k_after_rollback - 0.05, best_k_after_rollback + 0.05, color='gray', alpha=0.5)
    plt.savefig(run_dir + '/figure.pdf')
    plt.clf()


if __name__ == '__main__':
    # parse script parameters
    parser = argparse.ArgumentParser(description='CV2K')
    parser.add_argument('--version', type=str, default='standard', choices=['standard', 'x'], help='standard / x')
    parser.add_argument('--auto', type=str, default='n', choices=['y', 'n'], help='automatic search vs given range')
    parser.add_argument('--data', type=str, help='file_name.npy - n x m catalog: n rows are samples, '
                                                 'm columns are mutation types')
    parser.add_argument('--fraction', type=float, default=0.1, help='validation fraction')
    parser.add_argument('--reps', type=int, default=30, help='number of repeats per rank')
    parser.add_argument('--maxiter', type=int, default=2000, help='max number of iterations')
    parser.add_argument('--bottom_k', type=int, default=1, help='lower boundary')
    parser.add_argument('--top_k', type=int, default=10, help='upper boundary')
    parser.add_argument('--stride', type=int, default=1, help='stride')
    parser.add_argument('--workers', type=int, default=20, help='number of workers')
    parser.add_argument('--obj', type=str, default='kl', choices=['kl', 'euc', 'is'],
                        help='euclidean vs kl-divergence vs IS')
    parser.add_argument('--reg', type=float, default=0, help='regularization rate')
    args = parser.parse_args()

    V = np.load(args.data)
    n, m = V.shape
    # define convergence criterion
    eps = 1e-6
    # create rank cycle
    Ks = np.arange(args.bottom_k, args.top_k+1, args.stride)

    start = time.time()
    timestamp = datetime.datetime.now().strftime('%d-%m_%H-%M-%S')
    # create output directory
    run_dir = args.data + "_%dx%d_reps-%d_obj-%s_frac-%.4f-reg-%.2f_" % (n, m, args.reps, args.obj, args.fraction,
                                                                         args.reg) + timestamp
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    # main procedure
    if args.version == 'standard':
        if args.auto == 'n':
            errors = CV2K_standard()
        else:
            errors = CV2K_auto_binary_search(args.version)
    else:  # args.version == 'x'
        # define default number of category and sample folds
        category_folds = V.shape[1]  # m category folds
        sample_folds = 10
        # shuffling indices - shouldn't be necessary most of the time
        indices = np.arange(n)
        np.random.shuffle(indices)
        V = V[indices].copy()
        indices = np.arange(m)
        np.random.shuffle(indices)
        V = V.T[indices].T.copy()
        # splitting indices
        indices = np.arange(n)
        fold_sample_indices = np.array_split(indices, sample_folds)
        indices = np.arange(m)
        fold_category_indices = np.split(indices, category_folds)
        # get errors using our method
        if args.auto == 'n':
            errors = CV2K_x()
        else:
            errors = CV2K_auto_binary_search(args.version)

    end = time.time()
    print("Completed!")
    # save errors to output directory
    np.save(run_dir + "/errors.npy", errors)

    # redirect print to output file
    orig_stdout = sys.stdout
    f = open(run_dir + '/output.txt', 'w')
    sys.stdout = f

    # print parameters and runtime to output file
    print(args, flush=True)
    print("runtime: " + str(datetime.timedelta(seconds=end - start)))

    best_k_arg = np.argmin(np.nanmedian(errors, axis=1))
    best_k = Ks[best_k_arg]
    # rollback
    best_k_after_rollback_idx = rollback(errors)
    best_k_after_rollback = Ks[best_k_after_rollback_idx]
    
    # results figure
    produce_figure(rollback=True)
    print("Rollback from median: %d\n" % best_k_after_rollback)

    sys.stdout = orig_stdout
    f.close()
