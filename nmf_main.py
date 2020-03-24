import itertools
import os
from nmf_cv import *
from multiprocessing import Pool
import pickle
import nmf_plot
import sys
import datetime
from datetime import timedelta
import time
import argparse
import scipy.stats


def ncycle(iterable, repeat=1):
    for item in itertools.cycle(iterable):
        for _ in range(repeat):
            yield item


def create_iteration_cycle(low, high, stride, n_reps):
    rank_range = range(low, high + 1, stride)
    addition = range(high + 5, high + 4, 5)
    rank_range = [i for j in (rank_range, addition) for i in j]
    a = ncycle(rank_range, n_reps)
    rank_cyc = [next(a) for _ in range(len(rank_range) * n_reps)]
    return rank_cyc


if __name__ == '__main__':
    # parse script parameters
    parser = argparse.ArgumentParser(description='CV2K')
    parser.add_argument('--data', type=str, default='syn',
                        help='file_name.npy - nxm catalog: n rows are samples, m columns are mutation types')
    parser.add_argument('--reps', type=int, default=10, help='number of repeats per rank')
    parser.add_argument('--maxiter', type=int, default=2000, help='max number of iterations')
    parser.add_argument('--botbound', type=int, default=1, help='bottom rank boundary')
    parser.add_argument('--upbound', type=int, default=6, help='upper rank boundary')
    parser.add_argument('--stride', type=int, default=1, help='stride')
    parser.add_argument('--workers', type=int, default=6, help='number of workers')
    parser.add_argument('--obj', type=str, default='kl', choices=['kl', 'euc', 'is'], help='euclidean vs kl-divergence vs IS')
    parser.add_argument('--reg', type=float, default=0, help='regularization rate')
    parser.add_argument('--fraction', type=float, default=0.01, help='validation fraction')
    parser.add_argument('--best', type=int, default=100, help='best_of')
    args = parser.parse_args()

    if np.load(args.data).shape[1] == 96 or np.load(args.data).shape[1] == 1536:
        data = np.load(args.data)
        n, m = data.shape
    else:
        data = np.load(args.data).T
        n, m = data.shape

    # create iteration cycle
    rank_cyc = create_iteration_cycle(args.botbound, args.upbound, args.stride, args.reps)

    # convergence criterion
    eps = 1e-6

    start = time.time()
    # create name for output directory
    timestamp = datetime.datetime.now().strftime('%d-%m_%H-%M-%S')
    # create output directory
    run_dir = args.data + "_%dx%d_reps-%d_obj-%s_frac-%.4f-reg-%.2f_" % (n, m, args.reps, args.obj, args.fraction, args.reg) + timestamp
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    # save scripts
    os.system('cp nmf_main.py ' + run_dir + '/' + 'nmf_main.py.backup')
    os.system('cp nmf_cv.py ' + run_dir + '/' + 'nmf_cv.py.backup')

    # create pool and run nmf
    with Pool(args.workers) as pool:
        res = [pool.apply_async(crossval_nmf, (data, rank, args.maxiter, eps, args.obj, args.reg, args.fraction, args.best)) for
               rank in rank_cyc]
        pool.close()
        pool.join()

    # get results from pool
    res = [x.get() for x in res]
    Ws, Hs = [x[0] for x in res], [x[1] for x in res]
    val_err = np.reshape(np.array([x[2] for x in res]), (len(np.unique(rank_cyc)), args.reps))
    
    print("Completed!")
    # redirect print to output file
    orig_stdout = sys.stdout
    f = open(run_dir + '/output.txt', 'w')
    sys.stdout = f

    end = time.time()
    # print parameters and runtime to output file
    print(args, flush=True)
    print("runtime: " + str(timedelta(seconds=end - start)))
    
    # save output date to output directory
    np.save(run_dir + "/val_err.npy", val_err)


    # plot the results
    nmf_plot.plot(run_dir, data, ranks=rank_cyc, args=vars(args))

    # export W, H matrices
    mat_indices = [(k * args.reps + np.argmin(val_err[k])) for k in range(len(np.unique(rank_cyc)))]
    with open(run_dir + "/Hs", "wb") as fp:
        pickle.dump([Hs[i] for i in mat_indices], fp)
    with open(run_dir + "/Ws", "wb") as fp:
        pickle.dump([Ws[i] for i in mat_indices], fp)

    # rollback
    best_med = np.unique(rank_cyc)[np.argmin(np.nanmedian(val_err, axis=1).tolist())]
    best_k = np.argmin(np.nanmedian(val_err, axis=1))
    true_k = best_k
    for k in range(0, best_k):
        best_k_runs = val_err[best_k]
        next_k_runs = val_err[k]
        u, p = scipy.stats.ranksums(next_k_runs, best_k_runs)
        if p > 0.05:
            best_k = k
            break
    rollback = best_med - (true_k - best_k)
    print("Rollback from median: %d\n" % rollback)

    sys.stdout = orig_stdout
    f.close()